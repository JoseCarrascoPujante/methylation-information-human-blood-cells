#!/usr/bin/env python3
"""Donor-block entropy comparison and donor-stratified methylome distances.

Inputs are numerical outputs of the array reconstruction, not manuscript values.
The entropy input contains one mean h(beta) per physical sample. The distance
input contains the mean absolute difference across the same 481,737 cg probes
for each of the 861 unordered pairs of the 42 discovery samples.

The Friedman statistic is calibrated by independent within-donor permutations
of the seven population labels. This is a Monte Carlo permutation test with
the plus-one p-value, not an asymptotic chi-square test. Sample pairs are used
only descriptively; shared pairs are not independent biological replicates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


POPULATIONS = ("CD4", "CD8", "B", "NK", "Mono", "Neu", "Eos")
DONORS = tuple(range(1, 7))
N_CPG = 481737
STRATA = (
    "same_type_different_donor",
    "different_type_same_donor",
    "different_type_different_donor",
)
EXPECTED_PAIRS = dict(zip(STRATA, (105, 126, 630)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_metadata(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    if len(rows) != 42 or len({r["gsm"] for r in rows}) != 42:
        raise ValueError("Metadata must contain 42 unique discovery GSMs")
    result = {}
    cells = set()
    for row in rows:
        item = dict(row)
        item["donor_id"] = int(row["donor_id"])
        item["source_sample_index_zero"] = int(row["source_sample_index_zero"])
        cell = (item["donor_id"], item["population"])
        if cell in cells:
            raise ValueError("Duplicate donor/population cell")
        cells.add(cell)
        result[row["gsm"]] = item
    if cells != set(itertools.product(DONORS, POPULATIONS)):
        raise ValueError("Expected six complete donor blocks and seven populations")
    if len({r["source_sample_index_zero"] for r in result.values()}) != 42:
        raise ValueError("Duplicate array source columns")
    return result


def average_ranks_twice(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Return twice the average ranks and sum(t^3-t) for exact tied groups."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.int64)
    tie_sum = 0
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        # One-based ranks start+1,...,stop have doubled average start+1+stop.
        ranks[order[start:stop]] = start + 1 + stop
        size = stop - start
        tie_sum += size ** 3 - size
        start = stop
    return ranks, tie_sum


def entropy_comparison(path: Path, metadata: dict[str, dict], out: Path,
                       permutations: int, seed: int, batch_size: int) -> dict:
    rows = read_csv(path)
    if len(rows) != 42 or {r["gsm"] for r in rows} != set(metadata):
        raise ValueError("Entropy rows do not match the 42 sample identities")
    matrix = np.full((6, 7), np.nan, dtype=np.float64)
    by_cell = {}
    for row in rows:
        meta = metadata[row["gsm"]]
        donor = int(row["donor_id"])
        population = row["population"]
        if donor != meta["donor_id"] or population != meta["population"]:
            raise ValueError("Entropy donor/population identity mismatch")
        if row["sample_name"] != meta["sample_name"] or int(row["n_cpg"]) != N_CPG:
            raise ValueError("Entropy sample name or probe count mismatch")
        value = float(row["mean_h_beta"])
        total = float(row["sum_h_beta_bits"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid donor entropy")
        if not math.isclose(total, N_CPG * value, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Donor mean and sum are inconsistent")
        cell = (donor, population)
        if cell in by_cell:
            raise ValueError("Duplicate donor entropy cell")
        by_cell[cell] = row
        matrix[DONORS.index(donor), POPULATIONS.index(population)] = value
    if not np.isfinite(matrix).all():
        raise ValueError("Incomplete entropy matrix")

    rank2 = np.empty_like(matrix, dtype=np.int64)
    ties = 0
    for donor_index, values in enumerate(matrix):
        rank2[donor_index], tie_sum = average_ranks_twice(values)
        ties += tie_sum
    n, k = matrix.shape
    tie_correction = 1 - ties / (n * (k ** 3 - k))
    if tie_correction <= 0:
        raise ValueError("All observations tied within every donor")
    rank_sums2 = rank2.sum(axis=0, dtype=np.int64)
    observed_integer = int(np.square(rank_sums2).sum(dtype=np.int64))

    def statistic(integer_value):
        return (3 * integer_value / (n * k * (k + 1)) - 3 * n * (k + 1)) / tie_correction

    observed_q = float(statistic(observed_integer))
    generator = np.random.Generator(np.random.PCG64(seed))
    exceedances = 0
    distribution = Counter()
    done = 0
    while done < permutations:
        size = min(batch_size, permutations - done)
        repeated = np.broadcast_to(rank2, (size, n, k))
        # permuted (not permutation) independently permutes every donor row
        # in every replicate, preserving the seven values within that donor.
        shuffled = generator.permuted(repeated, axis=2)
        null_sums2 = shuffled.sum(axis=1, dtype=np.int64)
        null_integer = np.square(null_sums2).sum(axis=1, dtype=np.int64)
        exceedances += int(np.count_nonzero(null_integer >= observed_integer))
        unique, counts = np.unique(null_integer, return_counts=True)
        distribution.update({int(t): int(c) for t, c in zip(unique, counts)})
        done += size
    p_value = (exceedances + 1) / (permutations + 1)

    observations = []
    for i, donor in enumerate(DONORS):
        for j, population in enumerate(POPULATIONS):
            row = by_cell[(donor, population)]
            observations.append({
                "donor_id": donor, "population": population, "gsm": row["gsm"],
                "n_cpg": N_CPG, "mean_h_beta_bits_per_cpg": float(matrix[i, j]),
                "within_donor_rank": float(rank2[i, j] / 2),
            })
    write_csv(out / "donor_entropy_observations_and_ranks.csv", observations)
    summaries = [{
        "population": population, "n_donors": n, "n_cpg": N_CPG,
        "mean_of_donor_mean_h_beta": float(matrix[:, j].mean()),
        "sd_of_donor_mean_h_beta_ddof1": float(matrix[:, j].std(ddof=1)),
        "median_of_donor_mean_h_beta": float(np.median(matrix[:, j])),
        "mean_within_donor_rank": float(rank2[:, j].mean() / 2),
    } for j, population in enumerate(POPULATIONS)]
    write_csv(out / "donor_entropy_population_summary.csv", summaries)
    write_csv(out / "friedman_permutation_null_distribution.csv", [{
        "sum_squared_doubled_rank_sums": integer,
        "friedman_statistic": float(statistic(integer)), "frequency": frequency,
    } for integer, frequency in sorted(distribution.items())])
    result = {
        "test": "Friedman statistic calibrated by within-donor label permutations",
        "response": "mean over 481737 cg probes of h(beta) for each physical sample",
        "n_independent_donor_blocks": n, "n_populations": k,
        "donor_order": list(DONORS), "population_order": list(POPULATIONS),
        "friedman_statistic": observed_q,
        "kendalls_w": observed_q / (n * (k - 1)),
        "sum_squared_doubled_rank_sums": observed_integer,
        "tie_term_sum_t3_minus_t": ties, "tie_correction": tie_correction,
        "permutations": permutations, "seed": seed, "generator": "PCG64",
        "batch_size": batch_size, "upper_tail_exceedances_including_ties": exceedances,
        "p_value_plus_one": p_value, "p_value_formula": "(exceedances + 1)/(permutations + 1)",
        "monte_carlo_p_resolution": 1 / (permutations + 1),
        "permutation_unit": "independent permutation of all seven population labels within each donor",
        "null": "population labels are exchangeable within each of the six donor blocks",
        "inference_scope": "exploratory global comparison; no pairwise post hoc tests; six donors are the replication units",
        "rank_comparison": "integer comparison of squared sums of doubled average ranks",
        "calibration": "Monte Carlo, not exhaustive enumeration and not the chi-square approximation",
    }
    (out / "friedman_donor_block_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def descriptive(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {"n_pairs": len(values), "mean_MAD": float(array.mean()),
            "median_MAD": float(median), "q1_MAD": float(q1), "q3_MAD": float(q3)}


def donor_distances(path: Path, metadata: dict[str, dict], out: Path) -> dict:
    rows = read_csv(path)
    expected = {frozenset(pair) for pair in itertools.combinations(metadata, 2)}
    observed = set()
    strata = {key: [] for key in STRATA}
    by_donor = {donor: [] for donor in DONORS}
    enriched = []
    for row in rows:
        gsm_a, gsm_b = row["gsm_a"], row["gsm_b"]
        pair = frozenset((gsm_a, gsm_b))
        if pair not in expected or pair in observed:
            raise ValueError("Unexpected or duplicate sample pair")
        observed.add(pair)
        a, b = metadata[gsm_a], metadata[gsm_b]
        for suffix, sample in (("a", a), ("b", b)):
            if (row[f"population_{suffix}"] != sample["population"] or
                    int(row[f"donor_suffix_{suffix}"]) != sample["donor_id"] or
                    row[f"sample_title_{suffix}"] != sample["sample_name"] or
                    int(row[f"sample_{suffix}"]) != sample["source_sample_index_zero"] + 1):
                raise ValueError("Pair identities disagree with verified sample metadata")
        same_type = a["population"] == b["population"]
        same_donor = a["donor_id"] == b["donor_id"]
        if row["same_population"] != str(same_type) or row["same_donor"] != str(same_donor):
            raise ValueError("Pair flags disagree with sample identities")
        if int(row["n_cpg"]) != N_CPG:
            raise ValueError("Distance probe count mismatch")
        mad = float(row["mean_absolute_beta_distance"])
        rms = float(row["root_mean_square_beta_distance"])
        if not (math.isfinite(mad) and math.isfinite(rms) and 0 <= mad <= rms + 1e-12 <= 1 + 1e-12):
            raise ValueError("Invalid pairwise distance")
        if same_type:
            if same_donor:
                raise ValueError("A same-type, same-donor pair duplicates a physical sample")
            stratum = STRATA[0]
        elif same_donor:
            stratum = STRATA[1]
            by_donor[a["donor_id"]].append(mad)
        else:
            stratum = STRATA[2]
        strata[stratum].append(mad)
        enriched.append({
            "gsm_a": gsm_a, "gsm_b": gsm_b, "population_a": a["population"],
            "population_b": b["population"], "donor_a": a["donor_id"],
            "donor_b": b["donor_id"], "stratum": stratum, "n_cpg": N_CPG,
            "mean_absolute_beta_distance": mad,
        })
    if observed != expected:
        raise ValueError("Distance table does not contain all 861 unordered sample pairs")
    for key, expected_count in EXPECTED_PAIRS.items():
        if len(strata[key]) != expected_count:
            raise ValueError(f"Unexpected number of pairs in {key}")
    if any(len(values) != 21 for values in by_donor.values()):
        raise ValueError("Each donor must contribute 21 distinct cell-type pairs")
    summaries = [{"stratum": key, "n_cpg": N_CPG, **descriptive(strata[key])} for key in STRATA]
    donor_rows = [{"donor_id": donor, "n_types": 7, "n_cpg": N_CPG, **descriptive(by_donor[donor])}
                  for donor in DONORS]
    write_csv(out / "Table_S8A_donor_stratified_MAD.csv", summaries)
    write_csv(out / "Table_S8B_within_donor_MAD.csv", donor_rows)
    write_csv(out / "sample_pairs_with_donor_strata.csv", enriched)
    donor_means = np.asarray([row["mean_MAD"] for row in donor_rows], dtype=np.float64)
    result = {
        "n_samples": 42, "n_donors": 6, "n_types": 7, "n_cpg": N_CPG,
        "n_unordered_pairs": len(observed), "strata": summaries,
        "within_donor_means": donor_rows,
        "mean_of_six_within_donor_mean_MADs": float(donor_means.mean()),
        "sd_of_six_within_donor_mean_MADs_ddof1": float(donor_means.std(ddof=1)),
        "definition": "MAD is the mean absolute beta difference across the 481737 eligible cg probes for one sample pair",
        "aggregation": "Table S8A summarizes pair-specific MADs; Table S8B summarizes 21 between-type pairs separately in each donor",
        "quantiles": "linear interpolation (NumPy method='linear')",
        "inference": "descriptive; sample pairs share donors and samples and are not treated as independent replicates",
    }
    (out / "donor_stratified_MAD_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-values", type=Path, required=True)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    if args.permutations < 1 or args.batch_size < 1:
        parser.error("Permutation count and batch size must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = sample_metadata(args.sample_metadata)
    distances = donor_distances(args.distances, metadata, args.out)
    friedman = entropy_comparison(args.donor_values, metadata, args.out,
                                  args.permutations, args.seed, args.batch_size)
    print(json.dumps({"friedman": friedman, "MAD": distances}, indent=2))


if __name__ == "__main__":
    main()
