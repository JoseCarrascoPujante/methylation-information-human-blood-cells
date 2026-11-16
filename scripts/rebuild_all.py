#!/usr/bin/env python3
"""Rebuild numerical summaries and Figures 1-4/S1-S6 from package inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True,help='A new output directory.')
    args=ap.parse_args();root=Path(__file__).resolve().parents[1];out=args.out.resolve()
    if out.exists():ap.error('Use a new output directory.')
    if out==root or root in out.parents and any(part in ('data','regional','scripts') for part in out.relative_to(root).parts):
        ap.error('Output must not be inside an input directory.')
    out.mkdir(parents=True)
    def run(script,*arguments):
        print(f'Running {script}',flush=True)
        subprocess.run([sys.executable,str(root/'scripts'/script),*[str(v) for v in arguments]],check=True)
    run('recalculate_export_summaries.py','--data',root/'data','--out',out/'array_summaries')
    run('recompute_regional_statistics.py','--samples-root',root/'regional/samples','--out',out/'regional_results')
    run('plot_fig1.py','--out',out/'figures')
    run('plot_fig2_fig3.py','--data',root/'data','--out',out/'main_2_3')
    run('plot_fig4.py','--arrays',root/'data/arrays/aggregation_population_summary.csv',
        '--spectral',root/'data/spectral/signature_summary_current.csv',
        '--external',root/'data/model_fits/external_size_curve.csv',
        '--nested',root/'data/model_fits/nested_outer_predictions.csv','--outdir',out/'figures')
    run('plot_s1_s2.py','--data',root/'data','--out',out/'supporting')
    run('plot_s3.py','--results',out/'regional_results','--out',out/'supporting')
    run('plot_s4.py','--out',out/'supporting')
    run('plot_s5.py','--data',root/'data','--out',out/'supporting')
    run('donor_aware_comparisons.py',
        '--donor-values',root/'data/arrays/donor_entropy_sample_identities.csv',
        '--sample-metadata',root/'data/metadata/discovery_samples.csv',
        '--distances',root/'data/arrays/pairwise_distances.csv',
        '--out',out/'donor_comparisons','--permutations',999999,'--seed',20261104,'--batch-size',10000)
    run('supporting_distribution_and_examples.py',
        '--r-values',root/'data/arrays/discovery_R7_values.csv.gz',
        '--out',out/'supporting_distribution_and_examples')
    print(f'Completed numerical and figure rebuild in {out}',flush=True)


if __name__=='__main__':main()
