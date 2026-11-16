#!/usr/bin/env python3
"""Export ranked CpG identifiers and evaluated cutoffs from fitted array models.

Every signature is a prefix of the ranking for its own training partition.
This utility reads fitted models without fitting or modifying them.
"""
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    models = json.loads((args.fits_dir / 'fitted_models.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rank_path = args.out / 'ranked_cpg_identifiers_by_fit.csv.gz'
    cutoff_path = args.out / 'evaluated_rank_cutoffs_by_fit.csv'
    tmp_rank = rank_path.with_suffix('.tmp.gz')
    tmp_cutoff = cutoff_path.with_suffix('.tmp.csv')
    rows = 0
    cutoff_count = 0
    with tmp_rank.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8', newline='') as txt, tmp_cutoff.open('w', newline='') as cout:
            wr = csv.writer(txt)
            wc = csv.writer(cout)
            wr.writerow(['model_id', 'phase', 'outer_fold_index', 'rank', 'probe_id', 'R'])
            wc.writerow(['model_id', 'phase', 'outer_fold_index', 'rank_cutoff', 'minimum_selected_R'])
            for model in models:
                with np.load(args.fits_dir / model['model_file'], allow_pickle=False) as fit:
                    probes = fit['probe_ids']
                    scores = fit['ranking_scores']
                    assert len(probes) == len(scores) == model['ranked_features_exported']
                    assert len(set(probes.tolist())) == len(probes)
                    assert np.all(scores[:-1] >= scores[1:])
                    prefix = [model['model_id'], model['phase'], model.get('outer_fold_index', '')]
                    for rank, (probe, score) in enumerate(zip(probes, scores), 1):
                        wr.writerow(prefix + [rank, str(probe), format(float(score), '.17g')])
                        rows += 1
                    cutoffs = sorted(map(int, model['ranking_ties_at_evaluated_cutoffs']))
                    for k in cutoffs:
                        assert 0 < k <= len(probes)
                        wc.writerow(prefix + [k, format(float(scores[k - 1]), '.17g')])
                        cutoff_count += 1
            cout.flush()
            os.fsync(cout.fileno())
    for temporary, final in [(tmp_rank, rank_path), (tmp_cutoff, cutoff_path)]:
        # Windows requires write access for fsync.
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, final)
    print(json.dumps({'models': len(models), 'ranked_rows': rows, 'cutoff_rows': cutoff_count}, indent=2))


if __name__ == '__main__':
    main()
