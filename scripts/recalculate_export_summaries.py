#!/usr/bin/env python3
"""Recalculate numerical summaries from the observation-level array exports.

This command does not read raw intensity matrices or train a classifier.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); data=a.data.resolve(); out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    ag=pd.read_csv(data/'arrays/aggregation_population_summary.csv')
    don=pd.read_csv(data/'arrays/donor_entropy_summary.csv')
    ag['mean_h_donor_mean_beta']=ag.sum_A_bits/ag.n_cpg
    ag['mean_donor_h_beta']=ag.sum_B_bits/ag.n_cpg
    ag['mean_jensen_gap_per_CpG']=(ag.sum_A_bits-ag.sum_B_bits)/ag.n_cpg
    for r in ag.itertuples():
        z=don.loc[don.population==r.population,'mean_h_beta']
        if len(z)!=6 or not np.isclose(z.mean(),r.mean_donor_h_beta,rtol=0,atol=1e-12):
            raise ValueError('Donor and population summaries disagree.')
    ag.to_csv(out/'aggregation_population_summary.csv',index=False)

    r7=pd.read_csv(data/'arrays/discovery_R7_values.csv.gz').R7.to_numpy()
    if not np.isfinite(r7).all(): raise ValueError('R7 must be finite.')
    pd.DataFrame([{'n_cpg':len(r7),'median_R7':np.median(r7),
       'q25_R7':np.quantile(r7,.25),'q75_R7':np.quantile(r7,.75),
       'fraction_R7_gt_0_5':np.mean(r7>.5),'fraction_R7_gt_0_8':np.mean(r7>.8)}]
       ).to_csv(out/'discovery_R7_summary.csv',index=False)

    common=pd.read_csv(data/'arrays/probe_reproducibility_common_cg.csv.gz')
    if common.probe_id.duplicated().any(): raise ValueError('Repeated common CpG.')
    if not np.isfinite(common[['R_discovery','R_validation']]).all().all():
        raise ValueError('Ranking ratios must be finite.')
    rd=common.sort_values(['R_discovery','probe_id'],ascending=[False,True],kind='mergesort')
    rv=common.sort_values(['R_validation','probe_id'],ascending=[False,True],kind='mergesort')
    ov=[]; pc=[]
    def pattern_row(name, frame):
        z=frame.pattern_corr[np.isfinite(frame.pattern_corr)]
        return {'discovery_top_cpgs':name,'median_pattern_corr':z.median(),
                'q25':z.quantile(.25),'q75':z.quantile(.75),'n_finite':len(z)}
    pc.append(pattern_row('all',common))
    for m in [100,250,500,1000,5000,10000]:
        n=len(set(rd.head(m).probe_id)&set(rv.head(m).probe_id)); e=m*m/len(common)
        ov.append({'top_cpgs':m,'overlap':n,'expected_uniform_overlap':e,
                   'fold_enrichment':n/e,'shared_fraction':n/m})
        pc.append(pattern_row(m,rd.head(m)))
    pd.DataFrame(ov).to_csv(out/'overlap_curve.csv',index=False)
    pd.DataFrame(pc).to_csv(out/'pattern_correlation_curve.csv',index=False,na_rep='')
    rd.head(100).assign(discovery_rank=np.arange(1,101)).to_csv(
        out/'signature_100_probe_ids_and_statistics.csv',index=False,na_rep='')
    rho=float(np.corrcoef(common.R_discovery.rank(method='average'),
                          common.R_validation.rank(method='average'))[0,1])
    pd.DataFrame([{'common_cg_probes':len(common),'finite_pattern_correlations':
         int(np.isfinite(common.pattern_corr).sum()),'spearman_R6':rho}]).to_csv(
         out/'cross_cohort_summary.csv',index=False)

    eig=pd.read_csv(data/'spectral/signature_full_eigenvalue_spectra.csv')
    supplied=pd.read_csv(data/'spectral/signature_summary_current.csv').set_index('n_cpg')
    spectra=[]
    for n, f in eig.groupby('n_cpg',sort=True):
        vals=f.sort_values('eigen_index').eigenvalue.to_numpy()
        if not np.all(vals>0): raise ValueError('Retained eigenvalues must be positive.')
        q=vals/vals.sum()
        spectra.append({'n_cpg':int(n),
            'mean_marginal_sum_bits':float(supplied.loc[n,'mean_marginal_sum_bits']),
            'participation_ratio_dimension':float(1/np.dot(q,q)),
            'effective_rank':float(np.exp(-np.dot(q,np.log(q)))),
            'numerical_rank':len(vals),'n_constant_excluded':int(supplied.loc[n,'n_constant_excluded'])})
    pd.DataFrame(spectra).to_csv(out/'signature_summary_current.csv',index=False)
    print(json.dumps({'common_probes':len(common),'finite_correlations':int(np.isfinite(common.pattern_corr).sum()),
                      'R7_probes':len(r7),'spectral_sizes':len(spectra),'output':str(out)}))


if __name__=='__main__': main()
