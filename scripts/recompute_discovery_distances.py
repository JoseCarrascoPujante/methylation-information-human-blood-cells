#!/usr/bin/env python3
"""Recompute descriptive pairwise distances from the public GSE35069 signals.

No classifiers are fitted. Inputs are never changed. The sample ordering is
explicitly exported and follows the locally supplied array-analysis definition.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ('CD4', 'CD8', 'Mono', 'B', 'NK', 'Neu', 'Eos')
URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE35nnn/GSE35069/suppl/GSE35069_Matrix_signal_intensities.txt.gz'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signals', required=True, type=Path)
    p.add_argument('--metadata', required=True, type=Path,
                   help='Official GEO sample metadata JSON, in series order.')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    source = args.signals.resolve()
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text())
    if len(metadata) != 60:
        raise ValueError('Expected metadata for all 60 GEO samples.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with gzip.open(source, 'rb') as f:
        for _ in iter(lambda: f.read(8 * 1024 * 1024), b''):
            pass
    with gzip.open(source, 'rt') as f:
        header = f.readline().rstrip('\r\n').split('\t')
    if len(header) != 181 or header[0] != 'ID_REF':
        raise ValueError('Unexpected signal-matrix header.')
    mapping = []
    columns = [header[0]]
    for i in range(42):
        sample = i + 19
        triplet = header[1 + 3 * (sample - 1):1 + 3 * sample]
        expected = [f'SAMPLE {sample} Unmethylated Signal',
                    f'SAMPLE {sample} Methylated signal',
                    f'SAMPLE {sample} Detection Pval']
        if triplet != expected:
            raise ValueError(f'Unexpected signal columns for sample {sample}.')
        columns.extend(triplet)
        sample_meta = metadata[sample - 1]
        title = sample_meta['Sample_title'][0]
        prefix, donor_suffix = title.rsplit('_', 1)
        population = {'CD4+': 'CD4', 'CD8+': 'CD8', 'CD14+': 'Mono',
                      'CD19+': 'B', 'CD56+': 'NK', 'Neu': 'Neu', 'Eos': 'Eos'}[prefix]
        if population != CLASSES[i // 6] or int(donor_suffix) != i % 6 + 1:
            raise ValueError('Official titles differ from the supplied sample ordering.')
        mapping.append({'array_column_sample': sample, 'gsm': sample_meta['gsm'],
                        'sample_title': title, 'source_cell_type': sample_meta['Sample_source_name_ch1'][0],
                        'population': population, 'donor_suffix': donor_suffix,
                        'mapping_basis': 'GEO series sample order and published sample-title suffix'})
    first, second = np.triu_indices(42, k=1)
    mad_sums = np.zeros(len(first), dtype=np.float64)
    squared_sums = np.zeros(len(first), dtype=np.float64)
    raw_rows = 0
    retained = 0
    probe_ids = []
    for frame in pd.read_csv(source, sep='\t', compression='gzip', usecols=columns,
                             dtype={header[0]: str}, chunksize=10000):
        raw_rows += len(frame)
        ids = frame.iloc[:, 0].astype(str).to_numpy()
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64).reshape(-1, 42, 3)
        keep = np.fromiter((x.startswith('cg') for x in ids), bool, count=len(ids))
        keep &= (values[:, :, 2] < 0.01).sum(axis=1) >= 40
        chosen = values[keep]
        beta = chosen[:, :, 1] / (chosen[:, :, 1] + chosen[:, :, 0] + 100.0)
        if not np.isfinite(beta).all() or np.any((beta < 0) | (beta > 1)):
            raise ValueError('Invalid beta values after the stated filter.')
        retained += len(beta)
        probe_ids.extend(ids[keep].tolist())
        # Only 41 temporary columns at a time; pair order matches triu_indices.
        offset = 0
        for i in range(41):
            delta = beta[:, i, None] - beta[:, i + 1:]
            n = 41 - i
            mad_sums[offset:offset + n] += np.abs(delta).sum(axis=0)
            squared_sums[offset:offset + n] += np.square(delta).sum(axis=0)
            offset += n
        if raw_rows % 100000 == 0:
            print(json.dumps({'rows_read': raw_rows, 'retained_cg': retained}), flush=True)
    if retained != 481737 or len(set(probe_ids)) != retained:
        raise ValueError(f'Unexpected retained CpG set: {retained}.')
    mad = mad_sums / retained
    rms = np.sqrt(squared_sums / retained)
    rows = []
    for n, (i, j) in enumerate(zip(first, second)):
        a, b = mapping[int(i)], mapping[int(j)]
        rows.append({'sample_a': a['array_column_sample'], 'sample_b': b['array_column_sample'],
                     'gsm_a': a['gsm'], 'gsm_b': b['gsm'],
                     'sample_title_a': a['sample_title'], 'sample_title_b': b['sample_title'],
                     'population_a': a['population'], 'population_b': b['population'],
                     'donor_suffix_a': a['donor_suffix'], 'donor_suffix_b': b['donor_suffix'],
                     'same_population': a['population'] == b['population'],
                     'same_donor': a['donor_suffix'] == b['donor_suffix'],
                     'n_cpg': retained, 'mean_absolute_beta_distance': float(mad[n]),
                     'root_mean_square_beta_distance': float(rms[n])})
    same_type = np.array([r['same_population'] for r in rows])
    same_donor = np.array([r['same_donor'] for r in rows])
    masks = {'within': same_type, 'between': ~same_type,
             'same_donor_between': (~same_type) & same_donor,
             'diff_donor_between': (~same_type) & (~same_donor)}
    summaries = []
    for metric, vector in (('mad', mad), ('rms', rms)):
        for group, mask in masks.items():
            selected = vector[mask]
            q25, median, q75 = np.quantile(selected, [.25, .5, .75], method='linear')
            summaries.append({'metric_group': f'{metric}_{group}', 'n': len(selected),
                              'mean': float(selected.mean()), 'median': float(median),
                              'q25': float(q25), 'q75': float(q75), 'n_cpg': retained})
    assert {k: int(v.sum()) for k, v in masks.items()} == {
        'within': 105, 'between': 756, 'same_donor_between': 126, 'diff_donor_between': 630}
    write_csv(out / 'sample_mapping.csv', mapping)
    write_csv(out / 'pairwise_distances.csv', rows)
    write_csv(out / 'pairwise_distance_summary.csv', summaries)
    with gzip.open(out / 'retained_probe_ids.txt.gz', 'wt') as f:
        f.write('\n'.join(probe_ids) + '\n')
    print(json.dumps({'complete': True, 'n_cpg': retained, 'pairs': len(rows), 'summary': summaries}), flush=True)


if __name__ == '__main__':
    main()
