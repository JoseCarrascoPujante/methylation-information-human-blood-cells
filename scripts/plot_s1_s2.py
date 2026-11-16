"""Generate classification figures and tables from individual prediction exports.

Usage:
    python scripts/plot_s1_s2.py --data data --out rebuilt_classification

Inputs are the supplied CSV prediction tables. The script calculates empirical
metrics and figure coordinates; it does not fit classifiers.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parents[1]
ORDER = ['CD4', 'CD8', 'B', 'NK', 'Mono', 'Neu', 'Eos']
BLUE = '#28639A'
GRAY = '#59636F'
ORANGE = '#CB7E15'
INK = '#17212B'


def confusion_and_metrics(frame, labels):
    observed = set(frame['true']) | set(frame['predicted'])
    if not observed.issubset(set(labels)):
        raise ValueError(f'Unexpected cell types: {observed - set(labels)}')
    cm = pd.crosstab(frame['true'], frame['predicted']).reindex(
        index=labels, columns=labels, fill_value=0)
    counts = cm.to_numpy(dtype=float)
    n = counts.sum()
    if n <= 0 or np.any(counts.sum(axis=1) == 0):
        raise ValueError('Each studied true class must have observations.')
    joint = counts / n
    true_p = joint.sum(axis=1)
    predicted_p = joint.sum(axis=0)
    mi = sum(joint[i, j] * math.log2(joint[i, j] /
             (true_p[i] * predicted_p[j]))
             for i in range(len(labels)) for j in range(len(labels))
             if joint[i, j] > 0)
    metrics = {
        'n_samples': int(n),
        'n_cell_types': len(labels),
        'correct': int(np.trace(counts)),
        'accuracy': float(np.trace(counts) / n),
        'balanced_accuracy': float(np.mean(np.diag(counts) / counts.sum(axis=1))),
        'empirical_mutual_information_bits': float(mi),
        'true_label_entropy_bits': float(-sum(p * math.log2(p) for p in true_p if p > 0)),
    }
    return cm, metrics


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': .7,
        'savefig.facecolor': 'white',
    })


def export_figure(fig, directory, name):
    p = directory / name
    fig.savefig(p.with_suffix('.pdf'), metadata={'Title': name.replace('_', ' '), 'Author': ''})
    fig.savefig(p.with_suffix('.svg'))
    fig.savefig(p.with_suffix('.png'), dpi=300)
    fig.savefig(p.with_suffix('.tif'), dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def add_title(ax, title, subtitle):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=10.3, pad=23)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, ha='left',
            va='bottom', color=GRAY, fontsize=8.7)


def figure_2(nested_cm, nested_metrics, external_cm, external_metrics, figdir):
    fig = plt.figure(figsize=(7.2, 4.5))
    panels = [
        ([.105, .255, .335, .536], nested_cm, nested_metrics,
         'A  Nested donor validation'),
        ([.56, .255, .335, .536], external_cm, external_metrics,
         'B  External Euclidean classification'),
    ]
    cmap = LinearSegmentedColormap.from_list('methylation_counts', ['#FFFFFF', BLUE])
    image = None
    for bounds, cm, metrics, title in panels:
        ax = fig.add_axes(bounds)
        a = cm.to_numpy()
        image = ax.imshow(a, cmap=cmap, vmin=0, vmax=6,
                          interpolation='nearest', aspect='equal')
        labels = cm.index.tolist()
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(axis='both', length=0, pad=4)
        ax.set_xlabel('Predicted cell type', labelpad=5)
        ax.set_ylabel('True cell type', labelpad=5)
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which='minor', color='#DCE1E6', linewidth=.55)
        ax.tick_params(which='minor', bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#AAB3BC')
            spine.set_linewidth(.65)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(a[i, j])
                ax.text(j, i, str(val), ha='center', va='center',
                        color='white' if val >= 4 else (INK if val else '#7A838C'),
                        fontsize=9.2 if val else 8.2,
                        fontweight='bold' if val else 'normal')
        fig.text(bounds[0], .885, title, fontsize=9.8, fontweight='bold', color=INK)
        fig.text(bounds[0], .835,
                 f"{metrics['n_cell_types']} cell types; {metrics['n_samples']} samples",
                 fontsize=8.7, color=GRAY)
        pct = 100 * metrics['accuracy']
        acc = '100%' if pct == 100 else f'{pct:.1f}%'
        caption = (f"Accuracy = {acc} ({metrics['correct']}/{metrics['n_samples']})\n"
                   f"Empirical MI = {metrics['empirical_mutual_information_bits']:.3f} bits")
        fig.text(bounds[0] + bounds[2]/2, .07, caption, ha='center', va='center',
                 fontsize=8.5, color=INK, linespacing=1.35)
    cax = fig.add_axes([.927, .255, .014, .536])
    cb = fig.colorbar(image, cax=cax, ticks=[0, 3, 6])
    cb.ax.tick_params(labelsize=8.2, length=2)
    cb.ax.set_title('Count', fontsize=8, pad=6)
    cb.outline.set_linewidth(.6)
    export_figure(fig, figdir, 'S2_Fig_cell_type_recognition')


def figure_s1(fixed_metrics, fold_selection, external_metrics, figdir):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0))
    fig.subplots_adjust(left=.13, right=.965, bottom=.075, top=.925, hspace=.62)
    fmt = FuncFormatter(lambda x, pos: f'{int(x):,}')
    # A: each point is the observed accuracy for a fixed size.
    ax = axes[0]
    z = fixed_metrics.sort_values('signature_size_cpgs')
    ax.plot(z.signature_size_cpgs, z.accuracy, color=BLUE, marker='o',
            markersize=4.5, linewidth=1.5, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks(z.signature_size_cpgs)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, 1.055)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Fixed signature size (CpGs; log scale)', labelpad=5)
    ax.set_ylabel('Observed accuracy')
    add_title(ax, 'A  Fixed-size discovery evaluation',
              f"{int(z.n_samples.iloc[0])} samples; {int(z.n_cell_types.iloc[0])} cell types")
    ax.grid(axis='y', color='#E5E8EC', linewidth=.6, zorder=0)
    # B: fold indices identify partitions, not published donor identifiers.
    ax = axes[1]
    z = fold_selection.sort_values('outer_fold_index')
    ax.scatter(z.outer_fold_index, z.selected_signature_cpgs,
               color=GRAY, s=30, zorder=3)
    for row in z.itertuples():
        ax.annotate(f'{int(row.selected_signature_cpgs):,}',
                    (row.outer_fold_index, row.selected_signature_cpgs),
                    textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=8.5, color=INK)
    ax.set_yscale('log')
    ax.set_ylim(70, 11000)
    ax.set_yticks([100, 250, 1000, 5000])
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(z.outer_fold_index)
    ax.set_xlim(.6, 6.4)
    ax.set_xlabel('Outer fold index', labelpad=5)
    ax.set_ylabel('Selected CpGs\n(log scale)')
    add_title(ax, 'B  Signature size selected by inner validation',
              'Six outer partitions; indices identify folds')
    ax.grid(axis='y', color='#E5E8EC', linewidth=.6, zorder=0)
    # C: external sizes are sensitivity analyses, never a selection rule.
    ax = axes[2]
    for method, color, style, marker, label in [
            ('euclidean', BLUE, '-', 'o', 'Euclidean (primary rule)'),
            ('correlation', ORANGE, '--', '^', 'Correlation (exploratory)')]:
        z = external_metrics[external_metrics['method'] == method].sort_values('signature_size_cpgs')
        ax.plot(z.signature_size_cpgs, z.accuracy, color=color, linestyle=style,
                marker=marker, markersize=4.4, linewidth=1.5, label=label, zorder=3)
    ax.axvline(100, ymin=.24, color=GRAY, linewidth=.8, linestyle=':', zorder=1)
    ax.set_xscale('log')
    ax.set_xticks(sorted(external_metrics.signature_size_cpgs.unique()))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, 1.055)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Signature size (CpGs; log scale)', labelpad=5)
    ax.set_ylabel('Observed accuracy')
    add_title(ax, 'C  External sensitivity to signature size',
              '36 samples; 6 cell types; dotted line marks the 100-CpG analysis')
    ax.grid(axis='y', color='#E5E8EC', linewidth=.6, zorder=0)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, .015), ncol=2,
              fontsize=8.2, frameon=False, columnspacing=1.5, handlelength=2.7)
    export_figure(fig, figdir, 'S1_Fig_signature_size')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=BASE/'data',
                        help='The data directory of S2.')
    parser.add_argument('--out', type=Path, default=BASE,
                        help='Output package directory.')
    args = parser.parse_args()
    data = args.data.resolve()
    out = args.out.resolve()
    figdir = out/'figures'
    tabdir = out/'tables'
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    inputs = {
        'nested': data/'model_fits/nested_outer_predictions.csv',
        'fixed': data/'model_fits/discovery_fixed_size_predictions.csv.gz',
        'external': data/'model_fits/external_individual_predictions.csv.gz',
    }
    frames = {key: pd.read_csv(path) for key, path in inputs.items()}
    nested, fixed, external = frames['nested'], frames['fixed'], frames['external']
    labels7 = [label for label in ORDER if label in set(nested['true'])]
    labels6 = [label for label in ORDER if label in set(external['true'])]
    if len(labels7) != 7 or len(labels6) != 6:
        raise ValueError('These figures require seven discovery and six external classes.')
    if nested.groupby('donor').selected_signature_size.nunique().max() != 1:
        raise ValueError('A nested fold has inconsistent selected signature sizes.')

    groups = [('nested_discovery', 'euclidean', None, nested, labels7)]
    groups += [('fixed_size_discovery', 'euclidean', int(n), g, labels7)
               for n, g in fixed.groupby('n_cpg')]
    groups += [('external', method, int(n), g, labels6)
               for (n, method), g in external.groupby(['n_cpg', 'method'])]
    summaries, long_cm, class_rows, sample_rows = [], [], [], []
    matrices = {}
    for protocol, method, n_cpg, frame, labels in groups:
        cm, metrics = confusion_and_metrics(frame, labels)
        matrices[(protocol, method, n_cpg)] = (cm, metrics)
        scope = {'protocol': protocol, 'method': method, 'signature_size_cpgs': n_cpg}
        if protocol == 'external':
            role = 'primary_100_cpg' if method == 'euclidean' and n_cpg == 100 else 'exploratory_correlation' if method == 'correlation' else 'external_size_sensitivity'
        else:
            role = 'nested_validation' if protocol == 'nested_discovery' else 'fixed_size_evaluation'
        scope['analysis_role'] = role
        summaries.append({**scope, **metrics})
        counts = cm.to_numpy()
        for i, truth in enumerate(labels):
            true_count = int(counts[i, :].sum())
            predicted_count = int(counts[:, i].sum())
            tp = int(counts[i, i])
            class_rows.append({**scope, 'cell_type': truth,
                               'true_samples': true_count, 'predicted_samples': predicted_count,
                               'correct': tp, 'recall': tp/true_count,
                               'precision': tp/predicted_count if predicted_count else np.nan})
            for j, prediction in enumerate(labels):
                long_cm.append({**scope, 'true_label': truth, 'predicted_label': prediction,
                                'count': int(counts[i, j])})
        for record in frame.to_dict('records'):
            fold_index = record.get('donor')
            sample_rows.append({
                **scope,
                'prediction_index': record.get('outer_prediction_index', record.get('prediction_index')),
                'outer_fold_index': fold_index,
                'external_sample_name': record.get('sample_name', '') if scope['protocol'] == 'external' else '',
                'true_label': record['true'], 'predicted_label': record['predicted'],
                'correct': record['true'] == record['predicted'],
                'selected_signature_cpgs': record.get('selected_signature_size', n_cpg),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(tabdir/'classification_metrics.csv', index=False)
    pd.DataFrame(long_cm).to_csv(tabdir/'classification_confusion_counts.csv', index=False)
    pd.DataFrame(class_rows).to_csv(tabdir/'classification_metrics_by_cell_type.csv', index=False)
    pd.DataFrame(sample_rows).to_csv(tabdir/'classification_predictions_tidy.csv.gz',
                                    index=False, compression={'method': 'gzip', 'mtime': 0})
    selected = nested.groupby('donor', as_index=False).agg(
        selected_signature_cpgs=('selected_signature_size', 'first'),
        evaluated_samples=('true', 'size')).rename(columns={'donor': 'outer_fold_index'})
    selected.to_csv(tabdir/'classification_nested_signature_selection.csv', index=False)

    nested_cm, nested_metrics = matrices[('nested_discovery', 'euclidean', None)]
    euclidean_cm, euclidean_metrics = matrices[('external', 'euclidean', 100)]
    if nested_metrics['n_samples'] != 42 or euclidean_metrics['n_samples'] != 36:
        raise ValueError('Unexpected sample counts for the requested primary panels.')
    if max(nested_cm.to_numpy().max(), euclidean_cm.to_numpy().max()) > 6:
        raise ValueError('Counts exceed the requested shared 0–6 color scale.')
    nested_cm.to_csv(tabdir/'classification_nested_confusion.csv')
    euclidean_cm.to_csv(tabdir/'classification_external_euclidean_100_confusion.csv')
    matrices[('external', 'correlation', 100)][0].to_csv(
        tabdir/'classification_external_correlation_100_confusion.csv')

    configure()
    figure_2(nested_cm, nested_metrics, euclidean_cm, euclidean_metrics, figdir)
    figure_s1(summary[summary.protocol == 'fixed_size_discovery'], selected,
              summary[summary.protocol == 'external'], figdir)

    captions = '''S2 Fig. Cell-type recognition across donors and an external cohort.
(A) Confusion matrix for donor-held-out nested validation in the discovery cohort, comprising 42 samples and seven cell types. (B) Confusion matrix for the primary Euclidean external evaluation of the 100-CpG signature, comprising 36 samples and six shared cell types. Rows indicate true labels, columns indicate predicted labels, and cell annotations give sample counts. Both panels use the same color scale from zero to six. Accuracy is the fraction of correctly classified samples. Empirical mutual information (MI) is calculated between true and predicted labels from the normalized confusion-matrix counts and refers to the classes and frequencies of the respective evaluation. The two tasks have different class sets. CD4 and CD8 denote T-cell populations, B denotes B cells, NK denotes natural killer cells, Mono denotes monocytes, Neu denotes neutrophils, and Eos denotes eosinophils.

S1 Fig. Signature-size evaluation and nested selection.
(A) Observed accuracy for fixed signature sizes in the discovery evaluation of 42 samples and seven cell types. Each point corresponds to the predictions for the stated fixed size. (B) Signature size selected by inner validation in each of six outer partitions. Horizontal positions are fold indices, not GEO donor identifiers. The vertical axis uses a logarithmic scale. (C) Descriptive external sensitivity curves for 36 samples and six shared cell types. Euclidean classification is the primary rule, and the correlation-based comparison is exploratory. The vertical dotted line identifies the 100-CpG analysis. External sensitivity curves do not select the signature size. Horizontal axes in A and C use logarithmic scales. Lines connect the evaluated sizes to guide the eye.
'''
    (out/'captions_classification.txt').write_text(captions, encoding='utf-8')
    print(json.dumps({'figures': 2, 'output': str(out), 'nested': nested_metrics,
                      'external_euclidean_100': euclidean_metrics}, indent=2))


if __name__ == '__main__':
    main()
