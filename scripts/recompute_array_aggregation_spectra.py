#!/usr/bin/env python3
"""Recompute array aggregation, R7 and signature spectra from verified beta data.

Inputs are a version-1 array cache built from the public signal matrices. No
published summary, prediction, eigenvalue or previously selected probe list is
an input to these calculations. The common-probe signatures are ranked here
using the six discovery cell types shared with the external cohort.

Example:
    python recompute_array_aggregation_spectra.py --cache-dir array_cache --out beta_summaries

The output directory must be new and outside the input cache. Optional
--write-probe-aggregation also exports the two per-probe transformations for
all seven populations; their summaries are always calculated.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np


POPULATIONS = ("CD4", "CD8", "Mono", "B", "NK", "Neu", "Eos")
COMMON_POPULATIONS = POPULATIONS[:-1]
SIGNATURE_SIZES = (10, 25, 50, 100, 250, 500, 1000, 5000, 10000)
ABSOLUTE_EIGENVALUE_THRESHOLD = 1e-10
RELATIVE_EIGENVALUE_THRESHOLD = 1e-12
REQUIRED_CACHE_FILES = (
    "discovery_beta.npy",
    "discovery_probe_ids.csv.gz",
    "discovery_samples.csv",
    "common_discovery_indices.npy",
    "common_probe_ids.csv.gz",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, columns: list[str], rows) -> None:
    """Write stable UTF-8 CSV; gzip headers have no filename or timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(columns)
                    writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)


def binary_entropy(values: np.ndarray) -> np.ndarray:
    """h(p) in bits, with both endpoint terms defined as zero."""
    p = np.asarray(values, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Beta values must be finite fractions in [0, 1].")
    result = np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    x = p[interior]
    result[interior] = -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)
    return result


def variance_ratio(class_means: np.ndarray, within_class_variances: np.ndarray) -> np.ndarray:
    """R = Var(class means)/(Var(class means) + mean within-class Var)."""
    between = np.var(class_means, axis=1, ddof=1, dtype=np.float64)
    within = np.mean(within_class_variances, axis=1, dtype=np.float64)
    total = between + within
    ratio = np.zeros_like(total)
    np.divide(between, total, out=ratio, where=total > 0)
    if not np.isfinite(ratio).all() or np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("The calculated variance ratio is outside [0, 1].")
    return ratio


def load_verified_cache(cache: Path):
    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "complete",
        "array_orientation": "probes_by_samples",
        "precision": "float64",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unsupported or incomplete cache: {key} must equal {value!r}.")
    verified = {}
    for name in REQUIRED_CACHE_FILES:
        declared = manifest["files"][name]
        path = cache / name
        digest = file_sha256(path)
        if path.stat().st_size != declared["bytes"] or digest != declared["sha256"]:
            raise ValueError(f"Cache checksum/size mismatch: {name}")
        verified[name] = {"bytes": path.stat().st_size, "sha256": digest}

    beta = np.load(cache / "discovery_beta.npy", mmap_mode="r", allow_pickle=False)
    if beta.shape != (481737, 42) or beta.dtype != np.dtype("float64"):
        raise ValueError("The discovery cache must contain float64 beta for 481737 probes by 42 samples.")
    probe_ids = np.asarray([row["probe_id"] for row in read_csv(cache / "discovery_probe_ids.csv.gz")])
    if len(probe_ids) != beta.shape[0] or len(set(probe_ids.tolist())) != len(probe_ids):
        raise ValueError("Discovery probe identities are incomplete or duplicated.")
    if not all(identifier.startswith("cg") for identifier in probe_ids):
        raise ValueError("The discovery input contains a non-cg probe identifier.")
    samples = read_csv(cache / "discovery_samples.csv")
    if len(samples) != beta.shape[1] or len({row["gsm"] for row in samples}) != len(samples):
        raise ValueError("Discovery sample identities are incomplete or duplicated.")
    if set(row["population"] for row in samples) != set(POPULATIONS):
        raise ValueError("The discovery sample classes differ from the specified seven populations.")
    groups = []
    for population in POPULATIONS:
        positions = [i for i, row in enumerate(samples) if row["population"] == population]
        donor_ids = [int(samples[i]["donor_id"]) for i in positions]
        if len(positions) != 6 or sorted(donor_ids) != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Expected one sample for each of six donors in {population}.")
        groups.append(np.asarray(positions, dtype=np.int64))

    common_indices = np.load(cache / "common_discovery_indices.npy", mmap_mode="r", allow_pickle=False)
    common_ids = np.asarray([row["probe_id"] for row in read_csv(cache / "common_probe_ids.csv.gz")])
    if common_indices.shape != (448510,) or common_indices.dtype.kind != "i" or common_indices.dtype.itemsize != 8:
        raise ValueError("Expected 448510 int64 discovery row indices for the common probes.")
    if np.any(common_indices < 0) or np.any(common_indices >= beta.shape[0]) or np.any(np.diff(common_indices) <= 0):
        raise ValueError("Common probes must be unique, in discovery source order and within the discovery array.")
    if not np.array_equal(probe_ids[common_indices], common_ids):
        raise ValueError("Common probe identities do not match the indexed discovery rows.")
    return beta, probe_ids, samples, groups, common_indices, common_ids, verified, file_sha256(manifest_path)


def calculate_aggregation(beta, probe_ids, samples, groups, out: Path, export_per_probe: bool):
    n_probes = beta.shape[0]
    means = np.empty((n_probes, len(POPULATIONS)), dtype=np.float64)
    variances = np.empty_like(means)
    per_probe_a = np.empty_like(means) if export_per_probe else None
    per_probe_b = np.empty_like(means) if export_per_probe else None
    population_rows = []
    donor_rows = []
    donor_identity_rows = []
    for c, (population, sample_indices) in enumerate(zip(POPULATIONS, groups)):
        values = np.asarray(beta[:, sample_indices], dtype=np.float64)
        h_values = binary_entropy(values)
        means[:, c] = values.mean(axis=1, dtype=np.float64)
        variances[:, c] = values.var(axis=1, ddof=1, dtype=np.float64)
        a = binary_entropy(means[:, c])
        b = h_values.mean(axis=1, dtype=np.float64)
        sum_a = float(a.sum(dtype=np.float64))
        sum_b = float(b.sum(dtype=np.float64))
        gap = sum_a - sum_b
        donor_means = h_values.mean(axis=0, dtype=np.float64)
        population_rows.append([
            population, n_probes, sum_a, sum_b, gap,
            gap / sum_a if sum_a else 0.0,
            float(np.median(a)), float(donor_means.mean(dtype=np.float64)),
            float(donor_means.std(ddof=1, dtype=np.float64)), gap / n_probes,
        ])
        for local, source_column in enumerate(sample_indices):
            sample = samples[int(source_column)]
            donor = int(sample["donor_id"])
            donor_rows.append([donor, population, float(donor_means[local])])
            donor_identity_rows.append([
                sample["gsm"], sample["sample_name"], population, donor,
                int(source_column), n_probes, float(donor_means[local]),
                float(h_values[:, local].sum(dtype=np.float64)),
            ])
        if export_per_probe:
            per_probe_a[:, c] = a
            per_probe_b[:, c] = b

    write_csv(out / "arrays/aggregation_population_summary.csv", [
        "population", "n_cpg", "sum_A_bits", "sum_B_bits", "jensen_gap_bits",
        "gap_fraction_of_A", "median_A_across_CpGs", "donor_mean_h_beta",
        "donor_sd_h_beta", "mean_jensen_gap_per_CpG",
    ], population_rows)
    write_csv(out / "arrays/donor_entropy_summary.csv", ["donor", "population", "mean_h_beta"], donor_rows)
    write_csv(out / "arrays/donor_entropy_sample_identities.csv", [
        "gsm", "sample_name", "population", "donor_id", "cache_column_index_zero",
        "n_cpg", "mean_h_beta", "sum_h_beta_bits",
    ], donor_identity_rows)

    r7 = variance_ratio(means, variances)
    q25, median, q75 = np.quantile(r7, [0.25, 0.5, 0.75], method="linear")
    write_csv(out / "arrays/discovery_R7_values.csv.gz", ["probe_id", "R7"],
              ([probe, float(value)] for probe, value in zip(probe_ids, r7)))
    write_csv(out / "arrays/discovery_R7_summary.csv", [
        "n_cpg", "median_R7", "q25_R7", "q75_R7", "fraction_R7_gt_0_5", "fraction_R7_gt_0_8",
    ], [[n_probes, float(median), float(q25), float(q75),
         float(np.mean(r7 > 0.5)), float(np.mean(r7 > 0.8))]])

    if export_per_probe:
        columns = ["probe_id"]
        for population in POPULATIONS:
            columns.extend([f"{population}_A_h_mean_beta", f"{population}_B_mean_h_beta"])
        def rows():
            for j, probe in enumerate(probe_ids):
                row = [probe]
                for c in range(len(POPULATIONS)):
                    row.extend([float(per_probe_a[j, c]), float(per_probe_b[j, c])])
                yield row
        write_csv(out / "arrays/per_probe_population_aggregation.csv.gz", columns, rows())
    return means, variances


def calculate_spectra(beta, common_indices, common_ids, means, variances, groups, out: Path):
    # These moments were calculated above from the six donors in each class.
    # Only the six shared DISCOVERY classes participate in ranking or spectra.
    common_means = means[common_indices, :len(COMMON_POPULATIONS)]
    common_variances = variances[common_indices, :len(COMMON_POPULATIONS)]
    r6 = variance_ratio(common_means, common_variances)
    ranking = np.lexsort((common_ids, -r6))
    write_csv(out / "arrays/common_discovery_R6_ranking.csv.gz", ["probe_id", "R_discovery", "discovery_rank"],
              ([common_ids[j], float(r6[j]), rank] for rank, j in enumerate(ranking, 1)))
    write_csv(out / "spectral/signature_discovery_ranked_probes.csv.gz", ["probe_id", "R_discovery", "discovery_rank"],
              ([common_ids[j], float(r6[j]), rank] for rank, j in enumerate(ranking[:max(SIGNATURE_SIZES)], 1)))
    write_csv(out / "spectral/signature_membership.csv.gz", ["n_cpg", "within_signature_rank", "probe_id"],
              ([size, rank, common_ids[j]] for size in SIGNATURE_SIZES
               for rank, j in enumerate(ranking[:size], 1)))

    discovery_columns = np.sort(np.concatenate(groups[:len(COMMON_POPULATIONS)]))
    if len(discovery_columns) != 36:
        raise ValueError("Spectra require exactly 36 discovery samples from six shared populations.")
    summary_rows = []
    eigen_rows = []
    diagnostic_rows = []
    class_sum_rows = []
    for size in SIGNATURE_SIZES:
        selected = ranking[:size]
        selected_discovery_rows = common_indices[selected]
        values = np.asarray(beta[np.ix_(selected_discovery_rows, discovery_columns)], dtype=np.float64)
        # Exact constant rows are identified before centering, so rounding of
        # the mean of identical numbers cannot create spurious variance.
        constant = np.all(values == values[:, :1], axis=1)
        variable = values[~constant]
        if variable.shape[0] == 0:
            raise ValueError(f"All rows are constant for signature size {size}.")
        centered = variable - variable.mean(axis=1, keepdims=True, dtype=np.float64)
        sample_sd = variable.std(axis=1, ddof=1, keepdims=True, dtype=np.float64)
        if not np.isfinite(sample_sd).all() or np.any(sample_sd <= 0):
            raise ValueError("A nonconstant selected row has invalid sample variance.")
        z = centered / sample_sd
        gram = (z.T @ z) / (values.shape[1] - 1)
        eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        largest = float(eigenvalues[0])
        threshold = max(ABSOLUTE_EIGENVALUE_THRESHOLD, RELATIVE_EIGENVALUE_THRESHOLD * largest)
        positive = eigenvalues[eigenvalues > threshold]
        rank_cap = min(variable.shape[0], values.shape[1] - 1)
        if len(positive) == 0 or len(positive) > rank_cap:
            raise ValueError("Retained spectral rank is inconsistent with the centered data dimensions.")
        weights = positive / positive.sum(dtype=np.float64)
        participation_ratio = float(1.0 / np.sum(weights * weights, dtype=np.float64))
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights), dtype=np.float64)))

        # The marginal sum includes all selected probes. Constant rows, if any,
        # are excluded only from correlation standardization, not from this sum.
        class_sums = binary_entropy(common_means[selected]).sum(axis=0, dtype=np.float64)
        balanced_sum = float(class_sums.mean(dtype=np.float64))
        for population, population_sum in zip(COMMON_POPULATIONS, class_sums):
            class_sum_rows.append([size, population, 6, float(population_sum)])
        summary_rows.append([size, balanced_sum, participation_ratio, effective_rank,
                             len(positive), int(constant.sum())])
        for index, (eigenvalue, weight) in enumerate(zip(positive, weights), 1):
            eigen_rows.append([size, index, float(eigenvalue), float(weight)])
        diagnostic_rows.append([
            size, values.shape[1], variable.shape[0], int(constant.sum()), rank_cap,
            threshold, largest, float(eigenvalues[-1]), float(np.trace(gram)),
            len(positive), float(positive.sum(dtype=np.float64)),
        ])

    write_csv(out / "spectral/signature_summary_current.csv", [
        "n_cpg", "mean_marginal_sum_bits", "participation_ratio_dimension",
        "effective_rank", "numerical_rank", "n_constant_excluded",
    ], summary_rows)
    write_csv(out / "spectral/signature_full_eigenvalue_spectra.csv", [
        "n_cpg", "eigen_index", "eigenvalue", "normalized_eigenvalue",
    ], eigen_rows)
    write_csv(out / "spectral/signature_population_marginal_sums.csv", [
        "n_cpg", "population", "n_donors", "sum_h_mean_beta_bits",
    ], class_sum_rows)
    write_csv(out / "spectral/signature_rank_diagnostics.csv", [
        "n_cpg", "n_discovery_samples", "n_nonconstant_probes", "n_constant_excluded",
        "centered_rank_cap", "positive_eigenvalue_threshold", "largest_eigenvalue",
        "smallest_gram_eigenvalue", "gram_trace", "retained_rank", "retained_eigenvalue_sum",
    ], diagnostic_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--write-probe-aggregation", action="store_true")
    args = parser.parse_args()
    cache, out = args.cache_dir.resolve(), args.out.resolve()
    if out.exists():
        parser.error("The output directory must be new.")
    if out == cache or cache in out.parents:
        parser.error("The output directory must be outside the input cache.")

    loaded = load_verified_cache(cache)
    beta, probe_ids, samples, groups, common_indices, common_ids, verified, cache_manifest_sha = loaded
    out.mkdir(parents=True)
    for directory in ("arrays", "spectral", "metadata"):
        (out / directory).mkdir()
    shutil.copyfile(cache / "discovery_samples.csv", out / "metadata/discovery_samples.csv")
    means, variances = calculate_aggregation(beta, probe_ids, samples, groups, out, args.write_probe_aggregation)
    calculate_spectra(beta, common_indices, common_ids, means, variances, groups, out)
    print(json.dumps({"status": "complete", "discovery_probes": beta.shape[0],
                      "common_probes": len(common_indices), "signature_sizes": list(SIGNATURE_SIZES),
                      "outputs": sum(1 for path in out.rglob("*") if path.is_file())}, indent=2))


if __name__ == "__main__":
    main()
