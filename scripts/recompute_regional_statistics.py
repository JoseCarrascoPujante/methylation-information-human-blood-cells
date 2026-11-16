#!/usr/bin/env python3
"""Recalculate regional estimands from nine pairs of observed PAT count CSVs.

Python >=3.8, standard library only; no network, aggregate inputs or simulation.
The finite
hypergeometric expectation is evaluated using integer binomial coefficients.
Input layout: SAMPLES_ROOT/GSMxxxx/counts/GSMxxxx_candidate_{site,window}_counts.csv.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from functools import lru_cache
import gzip
import io
import json
import math
import os
from pathlib import Path
import tempfile

START, END, WIDTH = 100001, 120000, 4
N_SITES, N_WINDOWS = END - START + 1, (END - START + 1) // WIDTH
MIN_SITE_DEPTH, MIN_WINDOW_DEPTH, RAREFACTION_DEPTH = 10, 20, 20
TOL = 1e-12
PATTERNS = tuple(format(i, "04b").replace("0", "T").replace("1", "C") for i in range(16))
SITE_HEADER = ["gsm", "population", "donor_id_from_metadata", "cpg_index",
               "methylated_count", "total_count", "sample_depth_ge_10"]
WINDOW_HEADER = ["gsm", "population", "donor_id_from_metadata", "block_start_index",
                 "block_end_index", "complete_depth", "sample_depth_ge_20"] + ["count_" + p for p in PATTERNS]
METRIC_NAMES = ["observed_joint", "observed_marginal_sum", "observed_total_correlation",
                "depth20_joint", "depth20_marginal_sum", "depth20_total_correlation"]


@dataclass(frozen=True)
class Sample:
    gsm: str
    population: str
    cell_population: str
    donor: int
    sample_name: str


SAMPLES = (
    Sample("GSM5652279", "CD4", "CD4 T cells", 54, "Blood-T-CD4-Z000000TT"),
    Sample("GSM5652280", "CD4", "CD4 T cells", 51, "Blood-T-CD4-Z000000U7"),
    Sample("GSM5652281", "CD4", "CD4 T cells", 49, "Blood-T-CD4-Z000000UM"),
    Sample("GSM5652316", "B", "B cells", 54, "Blood-B-Z000000TX"),
    Sample("GSM5652317", "B", "B cells", 51, "Blood-B-Z000000UB"),
    Sample("GSM5652318", "B", "B cells", 49, "Blood-B-Z000000UR"),
    Sample("GSM5652302", "Mono", "Monocytes", 54, "Blood-Monocytes-Z000000TP"),
    Sample("GSM5652303", "Mono", "Monocytes", 51, "Blood-Monocytes-Z000000U3"),
    Sample("GSM5652304", "Mono", "Monocytes", 49, "Blood-Monocytes-Z000000UH"),
)
CELL_ORDER = ("CD4 T cells", "B cells", "Monocytes")


def h_bin(p):
    if not 0 <= p <= 1:
        raise ValueError("A methylation fraction is outside [0,1].")
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def entropy_counts(counts):
    n = sum(counts)
    if n <= 0:
        raise ValueError("Entropy is undefined for zero depth.")
    return math.fsum(-(c / n) * math.log2(c / n) for c in counts if c > 0)


@lru_cache(maxsize=65536)
def expected_entropy_component(total, category_count, draw_depth):
    """E[-(K/n)log2(K/n)], K~Hypergeom(total, category_count, n)."""
    if category_count == 0 or category_count == total:
        return 0.0
    denominator = math.comb(total, draw_depth)
    low = max(1, draw_depth - (total - category_count))
    high = min(draw_depth, category_count)
    return math.fsum(
        (math.comb(category_count, k) * math.comb(total - category_count, draw_depth - k) / denominator)
        * (-(k / draw_depth) * math.log2(k / draw_depth))
        for k in range(low, high + 1)
    )


def expected_plugin_entropy(counts, draw_depth=RAREFACTION_DEPTH):
    """Exact finite-sample expectation, evaluated in floating point; no RNG."""
    if draw_depth < 1 or any(not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError("Invalid rarefaction counts or depth.")
    total = sum(counts)
    if total < draw_depth:
        raise ValueError("Rarefaction depth exceeds the complete-fragment depth.")
    return math.fsum(expected_entropy_component(total, c, draw_depth) for c in counts if c)


def window_estimands(counts):
    depth = sum(counts)
    joint = entropy_counts(counts)
    methylated = [sum(counts[i] for i in range(16) if (i >> (3 - pos)) & 1)
                  for pos in range(WIDTH)]
    marginal = math.fsum(h_bin(c / depth) for c in methylated)
    rare_joint = expected_plugin_entropy(counts)
    rare_marginal = math.fsum(expected_plugin_entropy([depth - c, c]) for c in methylated)
    if not (-TOL <= joint <= marginal + TOL and marginal <= WIDTH + TOL):
        raise ValueError("Observed entropy bounds failed.")
    if not (-TOL <= rare_joint <= rare_marginal + TOL and rare_marginal <= WIDTH + TOL):
        raise ValueError("Rarefied entropy bounds failed.")
    if rare_joint > joint + TOL or rare_marginal > marginal + TOL:
        raise ValueError("The expected rarefied entropy exceeds the full empirical entropy.")
    return dict(zip(METRIC_NAMES, (joint, marginal, marginal - joint,
                                  rare_joint, rare_marginal, rare_marginal - rare_joint)))


def read_input_csv(path, expected_header):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames != expected_header:
        raise ValueError(f"Unexpected CSV schema: {path}")
    return list(reader)


def check_labels(row, sample):
    if (row["gsm"], row["population"], int(row["donor_id_from_metadata"])) != (
            sample.gsm, sample.population, sample.donor):
        raise ValueError(f"Sample, population or donor label mismatch for {sample.gsm}.")


def load_counts(root):
    paths = {}
    for sample in SAMPLES:
        for kind in ("site", "window"):
            paths[(sample.gsm, kind)] = root / sample.gsm / "counts" / f"{sample.gsm}_candidate_{kind}_counts.csv"
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError("All nine sample pairs are required; missing files: " + "; ".join(missing))
    data = {}
    for sample in SAMPLES:
        sr = read_input_csv(paths[(sample.gsm, "site")], SITE_HEADER)
        wr = read_input_csv(paths[(sample.gsm, "window")], WINDOW_HEADER)
        if len(sr) != N_SITES or len(wr) != N_WINDOWS:
            raise ValueError(f"Expected {N_SITES} sites and {N_WINDOWS} windows for {sample.gsm}.")
        methylated, totals, patterns, complete_depths = [], [], [], []
        for offset, row in enumerate(sr):
            check_labels(row, sample)
            m, n = int(row["methylated_count"]), int(row["total_count"])
            if int(row["cpg_index"]) != START + offset or not 0 <= m <= n:
                raise ValueError(f"Invalid site coordinates or integer counts for {sample.gsm}.")
            if row["sample_depth_ge_10"] != str(n >= MIN_SITE_DEPTH):
                raise ValueError(f"Invalid individual site-depth flag for {sample.gsm}.")
            methylated.append(m)
            totals.append(n)
        for offset, row in enumerate(wr):
            check_labels(row, sample)
            start = START + WIDTH * offset
            counts = tuple(int(row["count_" + p]) for p in PATTERNS)
            depth = sum(counts)
            if (int(row["block_start_index"]), int(row["block_end_index"])) != (start, start + WIDTH - 1):
                raise ValueError(f"Invalid aligned window coordinates for {sample.gsm}.")
            if min(counts) < 0 or int(row["complete_depth"]) != depth:
                raise ValueError(f"Invalid pattern counts or complete depth for {sample.gsm}.")
            if row["sample_depth_ge_20"] != str(depth >= MIN_WINDOW_DEPTH):
                raise ValueError(f"Invalid individual window-depth flag for {sample.gsm}.")
            for pos in range(WIDTH):
                c = sum(counts[i] for i in range(16) if (i >> (3 - pos)) & 1)
                j = WIDTH * offset + pos
                if c > methylated[j] or depth - c > totals[j] - methylated[j]:
                    raise ValueError(f"Complete-pattern marginals exceed site counts for {sample.gsm} at {start}.")
            patterns.append(counts)
            complete_depths.append(depth)
        data[sample.gsm] = {"methylated": methylated, "total": totals,
                            "patterns": patterns, "complete_depth": complete_depths}
    return data


@contextmanager
def output_text(path):
    if path.suffix == ".gz":
        # Finish the compression stream and validate its CRC before publishing.
        # The buffer contains only this output table, not raw PAT input data.
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=6, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                yield handle
        payload = buffer.getvalue()
        gzip.decompress(payload)
        fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as raw:
                if raw.write(payload) != len(payload):
                    raise OSError("Incomplete compressed-table write.")
                raw.flush()
                os.fsync(raw.fileno())
            if temporary.read_bytes() != payload:
                raise OSError("Compressed-table readback differs from the completed stream.")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            yield handle


def write_csv(path, header, rows):
    with output_text(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise ValueError(f"A nonfinite result would be written to {path.name}.")
            writer.writerow(row)


def mean(values):
    return math.fsum(values) / len(values)


def build_outputs(out, data):
    site_depths = [[data[s.gsm]["total"][i] for s in SAMPLES] for i in range(N_SITES)]
    window_depths = [[data[s.gsm]["complete_depth"][i] for s in SAMPLES] for i in range(N_WINDOWS)]
    common_sites = [i for i, depths in enumerate(site_depths) if min(depths) >= MIN_SITE_DEPTH]
    common_windows = [i for i, depths in enumerate(window_depths) if min(depths) >= MIN_WINDOW_DEPTH]
    if not common_sites or not common_windows:
        raise ValueError(f"An entropy summary is undefined: {len(common_sites)} common sites and {len(common_windows)} common windows.")
    common_site_set = set(common_sites)
    if any(WIDTH * i + pos not in common_site_set for i in common_windows for pos in range(WIDTH)):
        raise ValueError("A common complete window is inconsistent with the common-site filter.")

    def mask_rows(depth_vectors, threshold, is_window):
        for i, depths in enumerate(depth_vectors):
            row = ({"block_start_index": START + WIDTH * i, "block_end_index": START + WIDTH * i + WIDTH - 1}
                   if is_window else {"cpg_index": START + i})
            row.update(minimum_depth=min(depths), samples_passing=sum(n >= threshold for n in depths),
                       retained_all_nine=min(depths) >= threshold)
            row.update({s.gsm + "_depth": n for s, n in zip(SAMPLES, depths)})
            yield row

    tail = ["minimum_depth", "samples_passing", "retained_all_nine"] + [s.gsm + "_depth" for s in SAMPLES]
    write_csv(out / "pat_site_common_mask.csv.gz", ["cpg_index"] + tail,
              mask_rows(site_depths, MIN_SITE_DEPTH, False))
    write_csv(out / "pat_window_common_mask.csv.gz", ["block_start_index", "block_end_index"] + tail,
              mask_rows(window_depths, MIN_WINDOW_DEPTH, True))
    (out / "pat_common_sites.txt").write_text("".join(f"{START + i}\n" for i in common_sites))
    (out / "pat_common_blocks.txt").write_text("".join(f"{START + WIDTH * i}\n" for i in common_windows))
    write_csv(out / "pat_samples.csv", ["gsm", "cell_population", "donor", "sample_name"],
              ({"gsm": s.gsm, "cell_population": s.cell_population, "donor": s.donor,
                "sample_name": s.sample_name} for s in SAMPLES))

    fractions, site_h, donor_site = {}, {}, []
    for sample in SAMPLES:
        d = data[sample.gsm]
        fractions[sample.gsm] = [d["methylated"][i] / d["total"][i] for i in common_sites]
        site_h[sample.gsm] = [h_bin(p) for p in fractions[sample.gsm]]
        donor_site.append({"gsm": sample.gsm, "cell_population": sample.cell_population,
                           "donor": sample.donor, "n_sites": len(common_sites),
                           "mean_site_entropy": mean(site_h[sample.gsm])})
    write_csv(out / "pat_donor_site_entropy_points.csv", list(donor_site[0]), donor_site)

    def site_rows():
        for sample in SAMPLES:
            d = data[sample.gsm]
            for j, i in enumerate(common_sites):
                yield {"gsm": sample.gsm, "cell_population": sample.cell_population, "donor": sample.donor,
                       "cpg_index": START + i, "methylated_count": d["methylated"][i], "total_count": d["total"][i],
                       "fraction": fractions[sample.gsm][j], "h_fraction": site_h[sample.gsm][j]}
    write_csv(out / "pat_site_counts_common.csv.gz", ["gsm", "cell_population", "donor", "cpg_index",
              "methylated_count", "total_count", "fraction", "h_fraction"], site_rows())

    population_site, site_detail = [], []
    for cell in CELL_ORDER:
        samples = [s for s in SAMPLES if s.cell_population == cell]
        a_values, b_values = [], []
        for j, i in enumerate(common_sites):
            ps = [fractions[s.gsm][j] for s in samples]
            hs = [site_h[s.gsm][j] for s in samples]
            p_bar = mean(ps)
            a, b = h_bin(p_bar), mean(hs)
            if a < b - TOL:
                raise ValueError("Site-level Jensen inequality failed.")
            a_values.append(a)
            b_values.append(b)
            row = {"cell_population": cell, "cpg_index": START + i, "mean_fraction": p_bar,
                   "h_mean_fraction": a, "mean_donor_h_fraction": b, "jensen_gap": a - b}
            for s, p, h in zip(samples, ps, hs):
                row["fraction_donor_" + str(s.donor)] = p
                row["h_fraction_donor_" + str(s.donor)] = h
            site_detail.append(row)
        a_bar, b_bar = mean(a_values), mean(b_values)
        population_site.append({"cell_population": cell, "n_donors": len(samples), "n_sites": len(common_sites),
                                "mean_h_mean_fraction": a_bar, "mean_donor_h_fraction": b_bar, "jensen_gap": a_bar - b_bar})
    write_csv(out / "pat_population_aggregation_summary.csv", list(population_site[0]), population_site)
    write_csv(out / "pat_site_population_values.csv.gz", list(site_detail[0]), site_detail)

    window_detail, donor_window = [], []
    for sample in SAMPLES:
        metrics = {name: [] for name in METRIC_NAMES}
        for i in common_windows:
            counts = data[sample.gsm]["patterns"][i]
            result = window_estimands(counts)
            for name in METRIC_NAMES:
                metrics[name].append(result[name])
            row = {"gsm": sample.gsm, "cell_population": sample.cell_population, "donor": sample.donor,
                   "block_start_index": START + WIDTH * i, "block_end_index": START + WIDTH * i + WIDTH - 1,
                   "complete_depth": sum(counts), "joint_entropy": result["observed_joint"],
                   "marginal_sum": result["observed_marginal_sum"], "total_correlation": result["observed_total_correlation"],
                   "depth20_joint": result["depth20_joint"], "depth20_marginal_sum": result["depth20_marginal_sum"],
                   "depth20_total_correlation": result["depth20_total_correlation"]}
            row.update({"count_" + p: c for p, c in zip(PATTERNS, counts)})
            window_detail.append(row)
        row = {"gsm": sample.gsm, "cell_population": sample.cell_population, "donor": sample.donor,
               "n_blocks": len(common_windows)}
        row.update({name: mean(metrics[name]) for name in METRIC_NAMES})
        donor_window.append(row)
    write_csv(out / "pat_window_counts_common.csv.gz", list(window_detail[0]), window_detail)
    write_csv(out / "pat_donor_window_entropy_summary.csv", list(donor_window[0]), donor_window)
    population_window = []
    for cell in CELL_ORDER:
        donors = [r for r in donor_window if r["cell_population"] == cell]
        row = {"cell_population": cell, "n_donors": len(donors), "n_blocks": len(common_windows)}
        row.update({name: mean([r[name] for r in donors]) for name in METRIC_NAMES})
        population_window.append(row)
    write_csv(out / "pat_population_window_entropy_summary.csv", list(population_window[0]), population_window)

    design = {"genome_build": "hg38", "chromosome": "chr1", "coordinate_type": "one-based inclusive CpG indices",
              "cpg_index_start": START, "cpg_index_end": END, "n_samples": len(SAMPLES),
              "n_cell_populations": len(CELL_ORDER), "n_matched_donors": 3, "candidate_cpgs": N_SITES,
              "common_sites_depth_ge_10": len(common_sites), "fraction_sites_retained": len(common_sites) / N_SITES,
              "candidate_nonoverlap_4cpg_windows": N_WINDOWS,
              "windows_complete_fragment_depth_ge_20_all_samples": len(common_windows),
              "fraction_windows_retained": len(common_windows) / N_WINDOWS, "rarefaction_depth": RAREFACTION_DEPTH}
    write_csv(out / "regional_design_summary.csv", list(design), [design])
    return {"n_samples": len(SAMPLES), "common_sites": len(common_sites), "common_windows": len(common_windows),
            "population_site": population_site, "population_window": population_window}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory; existing directories are not overwritten.")
    args = parser.parse_args()
    root, out = args.samples_root.resolve(), args.out.resolve()
    if not root.is_dir():
        parser.error("The samples root is not a directory.")
    if out.exists() or out == root or root in out.parents or out in root.parents:
        parser.error("Choose a new output directory outside the input samples directory.")
    try:
        data = load_counts(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=out.name + "_staging_", dir=out.parent) as staging:
            summary = build_outputs(Path(staging), data)
            os.replace(staging, out)
        print(json.dumps(summary, indent=2))
    except (ValueError, KeyError, UnicodeError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
