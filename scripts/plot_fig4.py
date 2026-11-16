#!/usr/bin/env python3
"""Draw Figure 4 from explicit array, spectral and prediction summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


BLUE = '#1766a6'
NAVY = '#143e68'
ORANGE = '#e58a24'
GREEN = '#378044'
PURPLE = '#72528d'
GREY = '#536372'


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def mutual_information(predictions):
    joint = Counter((r['true'], r['predicted']) for r in predictions)
    a = Counter(r['true'] for r in predictions)
    b = Counter(r['predicted'] for r in predictions)
    n = len(predictions)
    return sum(c / n * math.log2(c * n / (a[t] * b[p]))
               for (t, p), c in joint.items())


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#66717b')
    ax.tick_params(colors='#303b45', width=.6, length=3)
    ax.set_axisbelow(True)


def heading(ax, letter, title, y=1.055):
    ax.text(-.115, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', color=NAVY, va='bottom')
    ax.text(0, y, title, transform=ax.transAxes, fontsize=9.4,
            fontweight='bold', color=NAVY, va='bottom')


def make(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ar = {r['population']: r for r in rows(args.arrays)}
    sp = sorted(rows(args.spectral), key=lambda r: int(r['n_cpg']))
    external = rows(args.external)
    pred = rows(args.nested)
    assert len(pred) == 42
    assert all(int(r['n_cpg']) == 481737 for r in ar.values())
    s100 = next(r for r in sp if int(r['n_cpg']) == 100)
    e100 = next(r for r in external if int(r['n_cpg']) == 100 and r['method'] == 'euclidean')
    mi_nested = mutual_information(pred)
    mi_external = float(e100['empirical_mutual_information_bits'])

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(7.2, 5.65), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.095, right=.985, bottom=.075, top=.925,
                           hspace=.43, wspace=.32, height_ratios=[1, 1.08])
    axa = fig.add_subplot(grid[0, 0]); axa.axis('off')
    heading(axa, 'A', 'Quantities and their units')
    axa.text(0, .94, '100 CpGs; 36 discovery samples; six cell types',
             color=GREY, fontsize=8, transform=axa.transAxes)
    lines = [
        ('Marginal-profile sum', f"{float(s100['mean_marginal_sum_bits']):.4f} bits", BLUE),
        ('Participation ratio', f"{float(s100['participation_ratio_dimension']):.4f}", ORANGE),
        ('Spectral effective rank', f"{float(s100['effective_rank']):.4f}", GREEN),
    ]
    for y, (label, value, color) in zip([.79, .64, .49], lines):
        axa.plot([0, 1], [y-.055, y-.055], color='#dce3e8', lw=.7, transform=axa.transAxes)
        axa.text(.015, y, label, va='center', color=color, transform=axa.transAxes)
        axa.text(.985, y, value, va='center', ha='right', fontweight='bold',
                 color=color, transform=axa.transAxes)
    axa.text(.985, .37, 'Both spectral dimensions are unitless', fontsize=8,
             ha='right', color=GREY, transform=axa.transAxes)
    axa.text(0, .235, 'Information between true and predicted labels',
             color=GREY, fontsize=8, transform=axa.transAxes)
    for y, label, value in [(.12, 'Nested discovery; seven types', mi_nested),
                            (-.01, 'External primary; six types', mi_external)]:
        axa.text(.015, y, label, fontsize=8, transform=axa.transAxes, va='center')
        axa.text(.985, y, f'{value:.4f} bits', color=PURPLE, fontweight='bold',
                 ha='right', transform=axa.transAxes, va='center')

    axb = fig.add_subplot(grid[0, 1]); style(axb)
    heading(axb, 'B', 'Array-wide profile sums')
    order = ['CD4', 'CD8', 'B', 'NK', 'Mono', 'Neu', 'Eos']
    sums = [float(ar[p]['sum_A_bits']) / 1e6 for p in order]
    axb.bar(range(7), sums, width=.69, color=BLUE)
    for i, value in enumerate(sums):
        axb.text(i, value + .006, f'{value:.4f}', ha='center', va='bottom',
                 fontsize=8, rotation=45)
    axb.set_xticks(range(7), order)
    axb.set_ylim(0, .35)
    axb.set_yticks([0, .1, .2, .3])
    axb.set_ylabel(r'Sum of $h(\overline{\beta})$ (Mbit)')
    axb.grid(axis='y', color='#e2e7eb', linewidth=.6)
    axb.text(.5, -.18, '481,737 CpGs; six donors per cell type', ha='center',
             fontsize=8, color=GREY, transform=axb.transAxes)

    sub = grid[1, 0].subgridspec(2, 1, hspace=.12, height_ratios=[1, 1])
    axc1 = fig.add_subplot(sub[0]); axc2 = fig.add_subplot(sub[1], sharex=axc1)
    for ax in [axc1, axc2]:
        style(ax)
        ax.set_xscale('log')
        ax.grid(axis='y', color='#e2e7eb', linewidth=.6)
        ax.axvline(100, color='#7d9fb9', linewidth=.8, ls='--', zorder=0)
        ax.set_xlim(8, 13000)
    heading(axc1, 'C', 'Profile sum and spectral dimensions', y=1.16)
    x = np.array([int(r['n_cpg']) for r in sp])
    marginal = np.array([float(r['mean_marginal_sum_bits']) for r in sp])
    pr = np.array([float(r['participation_ratio_dimension']) for r in sp])
    er = np.array([float(r['effective_rank']) for r in sp])
    axc1.plot(x, marginal, 'o-', color=BLUE, markersize=3, linewidth=1.15)
    axc1.set_yscale('log')
    axc1.set_ylim(3, 13000)
    axc1.set_yticks([10, 100, 1000, 10000])
    axc1.set_ylabel('Profile sum\n(bits)', labelpad=3)
    axc1.tick_params(labelbottom=False)
    axc1.text(.02, .82, '100-CpG signature marked', fontsize=8,
              color=GREY, transform=axc1.transAxes)
    axc2.plot(x, pr, 's-', color=ORANGE, markersize=3, linewidth=1.1, label='PR')
    axc2.plot(x, er, '^-', color=GREEN, markersize=3.2, linewidth=1.1, label='Effective rank')
    axc2.set_ylim(.85, 3.15)
    axc2.set_yticks([1, 2, 3])
    axc2.set_ylabel('Dimension\n(unitless)', labelpad=9)
    axc2.set_xlabel('CpGs in discovery-ranked signature', labelpad=3)
    axc2.legend(loc='lower right', fontsize=8, frameon=False,
                handlelength=1.8, handletextpad=.4)

    axd = fig.add_subplot(grid[1, 1]); axd.axis('off')
    heading(axd, 'D', 'Interpretation of the quantities')
    boxes = [
        ('Binary-state model', r'$N$ binary variables: $H(\mathbf{X})\leq N$ bits', NAVY, '#f0f5fa'),
        ('Array profile', 'h of donor means over specified CpGs', BLUE, '#f1f7fb'),
        ('Spectral structure', 'Concentration of standardized variation', GREEN, '#f3f7f1'),
        ('Cell identity', 'Empirical information about evaluated labels', PURPLE, '#f5f2f8'),
    ]
    for i, (title, subtitle, color, fill) in enumerate(boxes):
        y = .82 - .245*i
        axd.add_patch(FancyBboxPatch((-.005, y-.09), 1.005, .20,
                      boxstyle='round,pad=0.007,rounding_size=0.018',
                      linewidth=.75, edgecolor=color, facecolor=fill,
                      transform=axd.transAxes, clip_on=False))
        axd.text(.03, y+.04, title, color=color, fontweight='bold', fontsize=8.2,
                 transform=axd.transAxes, va='center')
        axd.text(.03, y-.035, subtitle, fontsize=8,
                 transform=axd.transAxes, va='center')

    for suffix in ['png', 'pdf', 'svg', 'tif']:
        kw = {'dpi': 600 if suffix == 'tif' else 300}
        if suffix == 'tif':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(out / f'Fig4.{suffix}', **kw)
    plt.close(fig)

    data = []
    for row in sp:
        for key, unit in [('mean_marginal_sum_bits','bits'),
                          ('participation_ratio_dimension','dimensionless'),
                          ('effective_rank','dimensionless')]:
            data.append({'panel':'A/C' if int(row['n_cpg']) == 100 else 'C',
                         'group':row['n_cpg'], 'quantity':key, 'value':row[key], 'unit':unit})
    for p in order:
        data.append({'panel':'B','group':p,'quantity':'sum_h_donor_mean_beta',
                     'value':ar[p]['sum_A_bits'],'unit':'bits'})
    for group, value in [('nested_discovery_seven_classes', mi_nested),
                         ('external_primary_six_classes', mi_external)]:
        data.append({'panel':'A','group':group,'quantity':'empirical_label_mutual_information',
                     'value':value,'unit':'bits'})
    with (out/'Fig4_plot_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','group','quantity','value','unit'])
        writer.writeheader();writer.writerows(data)
    caption = ('Fig 4. Distinct quantitative descriptions of methylation profiles and cell identity. '
        '(A) The class-balanced marginal-profile sum, participation-ratio dimension and spectral effective '
        'rank for the 100-CpG signature are evaluated across 36 discovery samples from the six shared '
        'cell types. The class-balanced sum is the equally weighted mean of six cell-type sums of '
        'h applied to donor-mean beta values. Spectral dimensions are dimensionless. Empirical label '
        'mutual information is reported separately for the seven-class nested discovery evaluation '
        '(42 samples; signature size selected within each training partition) and the six-class '
        '100-CpG primary Euclidean external evaluation (36 samples). '
        '(B) Sums of h applied to the equally weighted donor-mean beta profile over 481,737 discovery '
        'cg probes, shown separately for seven cell populations. These are operational marginal-profile '
        'sums, expressed in millions of bits; the beta profiles use six donors per population. '
        '(C) Class-balanced marginal-profile sums (upper axis, bits) and spectral dimensions (lower axis, '
        'dimensionless) across discovery-ranked signature sizes. Each spectrum uses the same 36 discovery '
        'samples and six shared cell types; the 100-CpG signature is marked. '
        '(D) Theoretical binary-state bounds, operational profile summaries, spectral structure and '
        'information about evaluated labels have distinct definitions. The bound H(X) <= N bits refers '
        'to a model containing N binary variables. These quantities are interpreted within their stated '
        'variables, units and analysis sets. PR, participation ratio. Data are provided in S1 Data; '
        'numerical summaries are in Tables S1 and S5 in S1 Text.')
    (out/'Fig4_caption.txt').write_text(caption+'\n')
    print(json.dumps({'outdir':str(out),'plot_rows':len(data),
        'files':{p.name:p.stat().st_size for p in out.glob('Fig4.*')}},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['arrays','spectral','external','nested','outdir']:
        parser.add_argument('--'+name,required=True)
    make(parser.parse_args())
