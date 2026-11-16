#!/usr/bin/env python3
"""Draw Figure 1 from explicit mathematical definitions; no empirical inputs.

The three-column organization and the blue/green/gold/purple quantity boxes
represent the conceptual quantities defined in the caption.  This is a vector
scientific diagram, not a generated illustration.

Requirements: Python 3, NumPy, Matplotlib and Pillow.
Usage: python plot_fig1.py --out /path/to/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


BLUE = "#123F70"
GREEN = "#416B40"
GOLD = "#9C7C05"
PURPLE = "#4D376F"
TEXT = "#17202A"
LIGHT_BLUE = "#EDF4F9"

CAPTION = """Figure 1. Information framework for DNA methylation profiles.
(A) A specified representation contains N binary variables, with X_j=0 and
X_j=1 denoting unmethylated and methylated states. A probability distribution
over these states has joint Shannon entropy H(X) <= N bits. This upper bound
does not require independence. Equality requires a uniform distribution over
all 2^N configurations, equivalently independent variables with
P(X_j=1)=1/2. N refers to the chosen representation; the diagram does not assign
a per-cell value without a convention for genomic sites, alleles and strands.
(B) The binary-entropy function h(p) = -p log2(p) - (1-p) log2(1-p), with
h(0)=h(1)=0, reaches one bit at p=1/2. For methylation arrays, this function is
applied operationally to the signal-derived beta value. It does not by itself
estimate a joint molecular distribution. In particular, applying h to the
equally weighted donor-mean beta and averaging donor-specific h(beta) values
are different operations; their difference is non-negative by concavity.
(C) Four complementary quantities are distinguished. The binary-state-space
bound concerns a specified joint distribution. The marginal-profile
transformation applies h to the equally weighted mean beta value across the
D_c donors for CpG j and cell type c. The descriptive ratio R_j compares
the sample variance among cell-type mean beta values with the sum of that
variance and the mean within-cell-type sample variance among donors. It is
dimensionless and does not describe variation among measured single cells.
Empirical predictive information I(C; C-hat) is mutual information between
true cell-type labels and predictions obtained for held-out samples; it is
calculated from their confusion-matrix frequencies and depends on the class
set and evaluation protocol. These quantities are not successive numerical
corrections to a common estimate of methylome capacity.
"""



def rounded_box(fig, xywh, color=BLUE, face="white", lw=1.1):
    x, y, w, h = xywh
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=fig.transFigure, linewidth=lw, edgecolor=color,
        facecolor=face, zorder=1,
    )
    fig.add_artist(p)
    return p


def ft(fig, x, y, text, size=15, color=TEXT, weight="normal", ha="left", **kwargs):
    return fig.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center", zorder=3, **kwargs)


def quantity_box(fig, y, h, number, color, face):
    rounded_box(fig, (0.658, y, 0.314, h), color=color, face=face)
    # Ellipse dimensions are corrected for the figure aspect ratio so that the
    # numbered marker is circular in physical output, including the TIFF.
    from matplotlib.patches import Ellipse
    e = Ellipse((0.674, y+h-0.034), width=0.025, height=0.025*13.02/7.47,
                transform=fig.transFigure, facecolor=color, edgecolor=color,
                zorder=3)
    fig.add_artist(e)
    ft(fig, 0.674, y+h-0.035, str(number), size=17, color="white",
       weight="bold", ha="center")


def draw_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavuserif",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.02, 7.47), facecolor="white")

    for xpos in (0.326, 0.642):
        fig.add_artist(plt.Line2D([xpos, xpos], [0.09, 0.95],
                       transform=fig.transFigure, color="#D7DDE2", lw=0.9))
    for x, label in [(0.022, "A"), (0.342, "B"), (0.657, "C")]:
        ft(fig, x, 0.931, label, size=28, color=BLUE, weight="bold")
    ft(fig, 0.187, 0.925, "CpG methylation as a binary\nstate space",
       size=15.5, color=BLUE, weight="bold", ha="center", linespacing=1.3)
    ft(fig, 0.506, 0.925, "Binary entropy of a CpG\nmethylation variable",
       size=15.5, color=BLUE, weight="bold", ha="center", linespacing=1.3)
    ft(fig, 0.832, 0.925, "Complementary quantities\nfor cell-type recognition",
       size=15.5, color=BLUE, weight="bold", ha="center", linespacing=1.3)

    # A: illustrative DNA and a mathematically defined representation.
    ft(fig, 0.038, 0.834,
       "Each modeled CpG state is binary:\nunmethylated (0) or methylated (1).",
       size=14.4, linespacing=1.4)
    dna = fig.add_axes([0.032, 0.616, 0.276, 0.153], zorder=2)
    dna.set_xlim(0, 10)
    dna.set_ylim(-0.45, 1.5)
    dna.axis("off")
    x = np.linspace(0, 10, 801)
    strand = 0.28*np.sin(2*np.pi*x/2.45)
    dna.plot(x, strand, color="#7995B6", lw=2.4)
    dna.plot(x, -strand, color="#7995B6", lw=2.4)
    for v in np.arange(0.1, 10, 0.34):
        s = 0.28*np.sin(2*np.pi*v/2.45)
        dna.plot([v, v], [-s, s], color="#A9BACD", lw=1.1)
    for pos, state in [(1.2, 0), (3.1, 1), (6.5, 0), (8.4, 1)]:
        anchor = abs(0.28*np.sin(2*np.pi*pos/2.45))
        dna.plot([pos, pos], [anchor, 0.88], color="#8B8B8B", lw=1.0)
        dna.scatter([pos], [0.95], s=98, facecolor=BLUE if state else "white",
                    edgecolor=BLUE if state else "#42484D", linewidth=1.05,
                    zorder=5)
        dna.text(pos, 1.2, str(state), ha="center", va="center",
                 fontsize=15, fontweight="bold", color=TEXT)
    dna.text(9.6, 0.98, "…", fontsize=17, color=TEXT, va="center")
    ft(fig, 0.170, 0.613, "Binary CpG states", size=14, color=BLUE,
       weight="bold", ha="center")
    rounded_box(fig, (0.033, 0.415, 0.274, 0.154), face=LIGHT_BLUE)
    ft(fig, 0.170, 0.538, "Specified binary representation",
       size=14.2, weight="bold", ha="center")
    ft(fig, 0.170, 0.491, r"$\mathbf{X}=(X_1,\ldots,X_N)$",
       size=20, ha="center")
    ft(fig, 0.170, 0.444, r"At most $2^N$ configurations", size=14.7, ha="center")
    fig.add_artist(FancyArrowPatch((0.170, 0.404), (0.170, 0.368),
                   transform=fig.transFigure, arrowstyle="-|>",
                   mutation_scale=18, color="#7590AD", lw=1.2))
    ft(fig, 0.170, 0.341, "Entropy bound for this\nrepresentation",
       size=15.4, color=BLUE, weight="bold", ha="center", linespacing=1.25)
    rounded_box(fig, (0.033, 0.116, 0.274, 0.171), face="white", lw=1.3)
    ft(fig, 0.170, 0.245, r"$H(\mathbf{X})\leq N$ bits",
       size=23, ha="center")
    ft(fig, 0.170, 0.192, "Equality for independent variables",
       size=13.6, ha="center")
    ft(fig, 0.170, 0.151, r"with $P(X_j=1)=1/2$ for every $j$.",
       size=13.5, ha="center")

    # B: exact binary entropy curve. Endpoints are handled analytically.
    rounded_box(fig, (0.349, 0.770, 0.280, 0.080), face="white", lw=1.15)
    ft(fig, 0.489, 0.809,
       r"$h(p)=-p\log_2p-(1-p)\log_2(1-p)$",
       size=14.5, ha="center")
    ft(fig, 0.510, 0.696, "Maximum at $p=0.5$\n$h(p)=1$ bit",
       size=15.7, color=BLUE, weight="bold", ha="center", linespacing=1.3)
    ax = fig.add_axes([0.390, 0.235, 0.235, 0.396])
    p = np.linspace(0, 1, 1001)
    h = np.zeros_like(p)
    inner = (p > 0) & (p < 1)
    h[inner] = -(p[inner]*np.log2(p[inner]) +
                 (1-p[inner])*np.log2(1-p[inner]))
    ax.plot(p, h, color=BLUE, lw=2.0)
    ax.axhline(1, color="#8D8D8D", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.vlines(0.5, 0, 1, color="#8D8D8D", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.annotate("", xy=(0.5, 1.005), xytext=(0.5, 1.13),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    ax.set(xlim=(0, 1), ylim=(0, 1.16))
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"Methylation probability $p$", labelpad=9)
    ax.set_ylabel(r"$h(p)$ (bits)", labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)
    ax.tick_params(length=4, width=.9)
    ft(fig, 0.499, 0.123,
       "For arrays, $h(\\beta)$ is an\noperational transformation.",
       size=14.0, color="#354B60", ha="center", linespacing=1.3)

    # C: complementary quantities, with no arrows implying numerical conversion.
    quantity_box(fig, .722, .137, 1, BLUE, "#EDF4F9")
    ft(fig, .704, .828, "Theoretical upper bound", size=15.5,
       color=BLUE, weight="bold")
    ft(fig, .704, .794, "for a binary state space", size=14.5, color=BLUE)
    ft(fig, .822, .753, r"$H(\mathbf{X})\leq N$ bits", size=19.0, ha="center")

    quantity_box(fig, .529, .173, 2, GREEN, "#F1F6F0")
    ft(fig, .704, .670, "Marginal-profile transform", size=15.0,
       color=GREEN, weight="bold")
    ft(fig, .815, .614,
       r"$h(\overline{\beta}_{jc}),\quad\overline{\beta}_{jc}=\frac{1}{D_c}\sum_d\beta_{jcd}$",
       size=17.0, ha="center")
    ft(fig, .815, .559, "Equal donor weights; bits per CpG", size=13.4,
       color=GREEN, ha="center")

    quantity_box(fig, .321, .188, 3, GOLD, "#FCF9EF")
    ft(fig, .704, .477, "Variance ratio (dimensionless)", size=14.4,
       color="#7C650A", weight="bold")
    ft(fig, .816, .426,
       r"$R_j=\frac{V_{\mathrm{between},j}}{V_{\mathrm{between},j}+V_{\mathrm{within},j}}$",
       size=18.5, ha="center")
    ft(fig, .815, .372, "Between: cell-type means", size=12.8,
       color="#655A2E", ha="center")
    ft(fig, .815, .343, "Within: donors of the same type", size=12.8,
       color="#655A2E", ha="center")

    quantity_box(fig, .116, .184, 4, PURPLE, "#F4F1F8")
    ft(fig, .704, .269, "Predictive information", size=15.3,
       color=PURPLE, weight="bold")
    ft(fig, .704, .235, "about cell identity", size=14.5, color=PURPLE)
    ft(fig, .815, .187,
       r"$I(C;\widehat C)=H(C)-H(C\mid\widehat C)$",
       size=17.0, ha="center")
    ft(fig, .815, .140, "True vs. held-out predicted labels; bits", size=12.7,
       color=PURPLE, ha="center")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1]/"figures")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_figure()
    # Deliberately avoid bbox_inches='tight': the dimensions/aspect ratio are
    # fixed and all drawing elements live within these explicit bounds.
    files = []
    for suffix, dpi, extra in [
        ("png", 300, {}), ("pdf", 300, {}), ("svg", 300, {}),
        ("tif", 600, {"pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        p = out/f"Fig1.{suffix}"
        fig.savefig(p, dpi=dpi, **extra)
        files.append({"file": p.name, "bytes": p.stat().st_size})
    plt.close(fig)
    # The numerical curve is included as a CSV as well as a mathematical
    # definition. Panels A and C are definitions, not empirical estimates.
    curve = out/"Fig1_plot_data.csv"
    with curve.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["p", "binary_entropy_bits"])
        for p in np.linspace(0, 1, 1001):
            h = 0.0 if p in (0, 1) else -(p*np.log2(p)+(1-p)*np.log2(1-p))
            writer.writerow([float(p), float(h)])
    files.append({"file": curve.name, "bytes": curve.stat().st_size})
    if (out/"Fig1.tif").stat().st_size >= 10_000_000:
        raise ValueError("TIFF exceeds the specified 10 MB limit")
    (out/"Fig1_caption_EN.txt").write_text(CAPTION, encoding="utf-8")
    print(json.dumps({"output": str(out), "files": files}, indent=2))


if __name__ == "__main__":
    main()
