#!/usr/bin/env python3
"""Draw the documented study design without loading results or fitting models.

Usage:
    python scripts/plot_s4.py --out /path/to/output

This schematic uses only the explicit DESIGN parameters below; it does not read
data files. Requires matplotlib and Pillow.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Fixed parameters of the study design depicted here. Samples are donor/cell-type
# observations. External donor counts are intentionally not specified because the
# diagram describes the 36 external samples across the six shared cell types.
DESIGN = {
    "discovery": {
        "donors": 6,
        "cell_types": 7,
        "samples": 42,
        "outer_training_donors": 5,
        "outer_test_donors": 1,
        "inner_training_donors": 4,
        "inner_validation_donors": 1,
    },
    "external": {
        "shared_cell_types": 6,
        "discovery_samples": 36,
        "external_samples": 36,
        "shared_cg_probes": 448510,
        "signature_cpgs": 100,
    },
}

STEM = "S4_Fig_evaluation_design"
LIGHT_BLUE = "#EAF2F9"
LIGHT_GOLD = "#FFF3DA"
EDGE = "#687786"
INK = "#17212B"

LEGEND = """S4 Fig. Evaluation design.
(A) Donor-held-out nested evaluation in the discovery cohort. Each outer split uses five donors for training and one donor for testing. Within the five training donors, inner leave-one-donor-out validation ranks CpGs using the four inner training donors and selects signature size using mean balanced accuracy across the five inner splits, with ties resolved in favor of the smaller size. Using the selected size, CpGs are ranked and the model is fitted on the five outer training donors, then applied to the held-out donor.
(B) External transfer uses a 100-CpG signature selected by the discovery ranking from 448,510 shared cg probes. Centroids are estimated from 36 discovery samples representing the six shared cell types, and classification is evaluated in 36 external samples. The Euclidean analysis is primary, and the correlation-based comparison is exploratory.
"""


def box(ax, x, y, w, h, text, face="white", fontsize=9, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor=face, edgecolor=EDGE, linewidth=.8,
    ))
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=INK, linespacing=1.3, fontweight=weight,
    )


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=.9, color=EDGE, shrinkA=0, shrinkB=0,
    ))


def make_figure():
    disc = DESIGN["discovery"]
    ext = DESIGN["external"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.2, 6.6))
    a = fig.add_axes([.055, .51, .89, .43])
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis("off")
    a.text(0, 1.055, "A  Donor-held-out validation", fontsize=11,
           fontweight="bold", color=INK)
    a.text(0, 1.005,
           f"Discovery: {disc['donors']} donors, {disc['cell_types']} cell types, "
           f"{disc['samples']} samples", fontsize=9, color=INK)
    box(a, .012, .23, .625, .69, "", face=LIGHT_BLUE)
    a.text(.325, .845,
           f"Outer training set: {disc['outer_training_donors']} donors",
           ha="center", fontsize=9.5, fontweight="bold", color=INK)
    box(a, .035, .49, .58, .29,
        f"Inner loop: {disc['outer_training_donors']} leave-one-donor-out splits\n"
        f"{disc['inner_training_donors']} training + "
        f"{disc['inner_validation_donors']} validation donor per split\n"
        f"Rank CpGs on the {disc['inner_training_donors']} training donors\n"
        "Select size using mean balanced accuracy\n"
        "Resolve ties in favor of the smaller size", fontsize=8.5)
    box(a, .12, .28, .41, .145,
        "Re-rank CpGs and fit the model\n"
        f"on all {disc['outer_training_donors']} training donors", fontsize=8.7)
    box(a, .735, .665, .25, .17,
        f"Outer test set\n{disc['outer_test_donors']} held-out donor",
        face=LIGHT_GOLD, fontsize=9)
    box(a, .735, .28, .25, .17,
        "Predict labels for\nthe held-out donor", fontsize=9)
    arrow(a, (.325, .476), (.325, .437))
    arrow(a, (.545, .352), (.721, .352))
    arrow(a, (.86, .65), (.86, .464))
    a.text(0, .075,
           "Repeat for all six outer donors; aggregate held-out predictions.",
           fontsize=9, color=INK)

    b = fig.add_axes([.055, .065, .89, .31])
    b.set(xlim=(0, 1), ylim=(0, 1))
    b.axis("off")
    b.text(0, 1.12, "B  Transfer to an external cohort", fontsize=11,
           fontweight="bold", color=INK)
    box(b, .012, .49, .438, .42,
        f"Discovery-derived signature\n"
        f"{ext['signature_cpgs']} CpGs; {ext['shared_cg_probes']:,} shared cg probes\n"
        f"Centroids: {ext['shared_cell_types']} cell types\n"
        f"{ext['discovery_samples']} discovery samples",
        face=LIGHT_BLUE, fontsize=9)
    box(b, .565, .49, .42, .42,
        f"External cohort\n{ext['shared_cell_types']} shared cell types; "
        f"{ext['external_samples']} samples\n"
        "Apply the discovery-selected signature\nEvaluate classification",
        face=LIGHT_GOLD, fontsize=9)
    box(b, .245, .075, .74, .25,
        "Primary evaluation: Euclidean rule\n"
        "Exploratory comparison: correlation rule", fontsize=9)
    arrow(b, (.465, .70), (.55, .70))
    arrow(b, (.78, .475), (.78, .34))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for figures and legends.")
    args = parser.parse_args()
    figdir = args.out / "figures"
    legdir = args.out / "legends"
    figdir.mkdir(parents=True, exist_ok=True)
    legdir.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    stem = figdir / STEM
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Evaluation design", "Author": "",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tif"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    (legdir / f"{STEM}.txt").write_text(LEGEND, encoding="utf-8")
    tabledir=args.out/'tables'
    tabledir.mkdir(parents=True,exist_ok=True)
    with (tabledir/'S4_Fig_design_parameters.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['component','parameter','value'])
        for component,parameters in DESIGN.items():
            for parameter,value in parameters.items():
                writer.writerow([component,parameter,value])
    print(f"Saved {STEM} in PDF, SVG, PNG and TIFF formats to {figdir}")


if __name__ == "__main__":
    main()
