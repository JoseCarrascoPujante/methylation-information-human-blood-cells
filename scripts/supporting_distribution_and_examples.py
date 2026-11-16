#!/usr/bin/env python3
"""Create the R7 histogram and exact, explicitly specified entropy examples.

R7 values are read from the probe-identified output of the array reconstruction.
The constructed examples enumerate their probability distributions exactly;
they do not represent biological observations or simulated sample estimates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def h(probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("Probability outside [0, 1]")
    if probability in (0, 1):
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)


def entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if np.any(values < 0) or not math.isclose(float(values.sum()), 1, abs_tol=1e-14):
        raise ValueError("Expected a probability distribution")
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def r7_histogram(path: Path, out: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["probe_id"] for row in rows]
    values = np.asarray([float(row["R7"]) for row in rows], dtype=np.float64)
    if len(rows) != 481737 or len(set(ids)) != len(rows):
        raise ValueError("Expected one R7 value for each of 481737 unique cg probes")
    if not all(probe.startswith("cg") for probe in ids):
        raise ValueError("Expected cg probe identifiers")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("R7 values must be finite and lie in [0, 1]")
    edges = np.linspace(0, 1, 51, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != len(rows):
        raise ValueError("Histogram does not include every eligible probe")
    percentages = counts * 100.0 / len(rows)
    write_csv(out / "R7_histogram_bins.csv", [{
        "bin_index_one": index + 1, "left_edge_inclusive": float(edges[index]),
        "right_edge": float(edges[index + 1]), "right_edge_inclusive": index == 49,
        "n_cpg": int(count), "percent_of_eligible_cg_probes": float(percentages[index]),
    } for index, count in enumerate(counts)])
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    gt05, gt08 = int(np.count_nonzero(values > 0.5)), int(np.count_nonzero(values > 0.8))
    summary = {
        "n_cpg": len(rows), "n_cell_types": 7, "n_donors_per_type": 6,
        "n_bins": 50, "bin_width": 0.02,
        "bins": "left-inclusive, right-exclusive, except the final bin includes R7=1",
        "min_R7": float(values.min()), "q1_R7": float(q1), "median_R7": float(median),
        "q3_R7": float(q3), "max_R7": float(values.max()),
        "count_R7_gt_0_5": gt05, "percent_R7_gt_0_5": gt05 * 100.0 / len(rows),
        "count_R7_gt_0_8": gt08, "percent_R7_gt_0_8": gt08 * 100.0 / len(rows),
        "definition": "R7 = sample variance of seven cell-type mean beta values divided by that variance plus mean within-type sample variance across six donors; both variances use ddof=1",
        "eligibility": "481737 discovery cg probes passing detection-P eligibility in at least 40 of 42 samples",
        "unit": "one value per eligible assayed cg probe, not per donor",
    }
    (out / "R7_histogram_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11,
        "axes.titlesize": 12, "axes.linewidth": 0.8, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.25, 4.45))
    ax.bar(edges[:-1], percentages, width=np.diff(edges), align="edge",
           color="#47799F", edgecolor="white", linewidth=0.35)
    ax.axvline(median, color="#202020", linewidth=1.45,
               label=f"Median = {median:.4f}")
    ax.axvline(0.5, color="#A7571B", linestyle="--", linewidth=1.35,
               label=rf"$R_j > 0.5$: {gt05 * 100 / len(rows):.2f}%")
    ax.axvline(0.8, color="#7B3A63", linestyle="--", linewidth=1.35,
               label=rf"$R_j > 0.8$: {gt08 * 100 / len(rows):.2f}%")
    ax.set(xlim=(0, 1), ylim=(0, percentages.max() * 1.39),
           xlabel=r"Variance ratio $R_j$ (seven discovery cell types)",
           ylabel="Eligible cg probes per bin (%)")
    ax.set_title(r"Distribution of discovery $R_j$", loc="left", pad=13, fontweight="bold")
    ax.text(1, 1.035, "481,737 cg probes", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="white",
              framealpha=1, fontsize=10, handlelength=2.5)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.5)
    ax.yaxis.grid(True, color="#D8DEE4", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.16, top=0.87)
    fig.savefig(out / "S6_Fig.png", dpi=300, facecolor="white")
    fig.savefig(out / "S6_Fig.pdf", facecolor="white")
    fig.savefig(out / "S6_Fig.svg", facecolor="white")
    plt.close(fig)
    caption = (
        "S6 Fig. Distribution of the discovery ranking statistic. The histogram shows the "
        "descriptive seven-cell-type variance ratio R across 481,737 eligible discovery cg "
        "probes, using 50 equal-width bins of 0.02 from 0 to 1. The vertical axis is the "
        "percentage of probes in each bin. The median and thresholds 0.5 and 0.8 are "
        "indicated. These are probe-level summaries of the measured array subset, not counts "
        "of independent sites or a genome-wide sampling estimate. Bin counts and probe "
        "identifiers are supplied in S1 Data.\n"
    )
    (out / "S6_Fig_caption.txt").write_text(caption, encoding="utf-8")
    return summary


def toy_aggregation(out: Path) -> list[dict]:
    scenarios = {
        "polarized_donor_fractions": [0, 0, 0, 1, 1, 1],
        "identical_intermediate_donor_fractions": [0.5] * 6,
    }
    rows, fractions = [], []
    for name, values in scenarios.items():
        mean_fraction = math.fsum(values) / len(values)
        a = h(mean_fraction)
        b = math.fsum(h(value) for value in values) / len(values)
        rows.append({"example": name, "n_constructed_donors": len(values),
                     "donor_fractions": ";".join(str(value) for value in values),
                     "mean_fraction": mean_fraction, "h_mean_fraction_bits": a,
                     "mean_donor_h_fraction_bits": b, "jensen_gap_bits": a - b})
        fractions.extend({"example": name, "donor_id": i + 1,
                          "methylation_fraction": value, "h_fraction_bits": h(value)}
                         for i, value in enumerate(values))
    write_csv(out / "Table_S9A_donor_aggregation_examples.csv", rows)
    write_csv(out / "toy_donor_fractions.csv", fractions)
    return rows


def toy_label_information(out: Path) -> list[dict]:
    scenarios = {"equal_class_distributions": (0.5, 0.5),
                 "unequal_class_distributions": (0.5, 0.0)}
    rows, states = [], []
    for name, (p_a, p_b) in scenarios.items():
        p_c = np.array([0.5, 0.5])
        joint = p_c[:, None] * np.array([[1 - p_a, p_a], [1 - p_b, p_b]])
        marginal_x = joint.sum(axis=0)
        hx = entropy(marginal_x)
        conditional = 0.5 * h(p_a) + 0.5 * h(p_b)
        mi = 0.0
        for c, x in itertools.product(range(2), repeat=2):
            probability = float(joint[c, x])
            states.append({"example": name, "class": "A" if c == 0 else "B",
                           "binary_state_X": x, "joint_probability": probability})
            if probability > 0:
                mi += probability * math.log2(probability / (p_c[c] * marginal_x[x]))
        if not math.isclose(mi, hx - conditional, abs_tol=1e-14):
            raise ValueError("Constructed mutual-information identities disagree")
        rows.append({"example": name, "P_class_A": 0.5, "P_class_B": 0.5,
                     "P_X1_given_A": p_a, "P_X1_given_B": p_b,
                     "H_X_bits": hx, "H_X_given_class_bits": conditional,
                     "I_X_class_bits": mi})
    write_csv(out / "Table_S9B_label_information_examples.csv", rows)
    write_csv(out / "toy_binary_state_class_probabilities.csv", states)
    return rows


def pattern_distribution_summary(name: str, n_sites: int, probabilities: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    patterns = np.array(list(itertools.product((0, 1), repeat=n_sites)), dtype=np.float64)
    if len(probabilities) != len(patterns):
        raise ValueError("Pattern probability vector has the wrong length")
    marginal_p = probabilities @ patterns
    h_joint = entropy(probabilities)
    marginal_sum = math.fsum(h(float(value)) for value in marginal_p)
    rows = [{"example": name, "binary_pattern": "".join(str(int(x)) for x in pattern),
             "probability": float(probability)} for pattern, probability in zip(patterns, probabilities)]
    centered = patterns - marginal_p
    covariance = (centered.T * probabilities) @ centered
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale == 0):
        raise ValueError("Constructed spectral examples require nonzero marginal variances")
    correlation = covariance / np.outer(scale, scale)
    result = {"example": name, "n_binary_sites": n_sites,
              "n_positive_probability_patterns": int(np.count_nonzero(probabilities)),
              "marginal_probabilities": ";".join(str(float(x)) for x in marginal_p),
              "sum_marginal_entropies_bits": marginal_sum, "joint_entropy_bits": h_joint,
              "total_correlation_bits": marginal_sum - h_joint}
    return result, rows, correlation


def toy_linked_patterns(out: Path) -> list[dict]:
    uniform = np.full(16, 1 / 16, dtype=np.float64)
    concordant = np.zeros(16, dtype=np.float64)
    concordant[[0, 15]] = 0.5
    rows, states = [], []
    for name, probabilities in (("independent_four_fair_bits", uniform),
                                ("TTTT_or_CCCC_equal_probability", concordant)):
        result, state_rows, _ = pattern_distribution_summary(name, 4, probabilities)
        rows.append(result)
        for row in state_rows:
            row["CpG_pattern_T0_C1"] = row["binary_pattern"].translate(str.maketrans("01", "TC"))
        states.extend(state_rows)
    write_csv(out / "Table_S9C_joint_marginal_examples.csv", rows)
    write_csv(out / "toy_four_CpG_pattern_probabilities.csv", states)
    return rows


def toy_spectra(out: Path) -> list[dict]:
    independent = np.full(8, 1 / 8, dtype=np.float64)
    parity = np.zeros(8, dtype=np.float64)
    parity[[0, 3, 5, 6]] = 0.25  # 000, 011, 101, 110; X3 = X1 XOR X2.
    rows, states, matrices = [], [], []
    for name, probabilities in (("independent_three_fair_bits", independent),
                                ("XOR_three_pairwise_independent_bits", parity)):
        result, state_rows, correlation = pattern_distribution_summary(name, 3, probabilities)
        eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
        if np.any(eigenvalues <= 0):
            raise ValueError("Unexpected nonpositive eigenvalue in constructed example")
        weights = eigenvalues / eigenvalues.sum()
        result.update({"correlation_eigenvalues": ";".join(str(float(value)) for value in eigenvalues),
                       "participation_ratio": float(1 / np.square(weights).sum()),
                       "effective_rank": float(np.exp(-np.sum(weights * np.log(weights))))})
        rows.append(result)
        states.extend(state_rows)
        matrices.extend({"example": name, "site_a": i + 1, "site_b": j + 1,
                         "population_correlation": float(correlation[i, j])}
                        for i, j in itertools.product(range(3), repeat=2))
    write_csv(out / "Table_S9D_same_spectrum_different_joint_entropy.csv", rows)
    write_csv(out / "toy_three_bit_pattern_probabilities.csv", states)
    write_csv(out / "toy_three_bit_correlation_matrices.csv", matrices)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-values", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    histogram = r7_histogram(args.r_values, args.out)
    examples = {"A_donor_aggregation": toy_aggregation(args.out),
                "B_label_information": toy_label_information(args.out),
                "C_joint_and_marginal": toy_linked_patterns(args.out),
                "D_spectrum_and_joint_entropy": toy_spectra(args.out)}
    (args.out / "exact_examples_summary.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"histogram": histogram, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
