#!/usr/bin/env python3
"""Recompute common-probe cell-type means, ranking ratios and correlations.

No classifiers or spectral estimates are fitted. Signals are parsed in float64.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES=['CD4','CD8','Mono','B','NK','Neu']
DISC_MAP={'CD4+':'CD4','CD8+':'CD8','CD14+':'Mono','CD19+':'B','CD56+':'NK','Neu':'Neu','Eos':'Eos'}
SIZES=[10,25,50,100,250,500,1000,5000,10000]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def headers(path):
    with gzip.open(path,'rt') as f:return f.readline().rstrip('\r\n').split('\t')


def extract(path,samples,min_detected,cohort):
    head=headers(path);columns=[head[0]]
    for s in samples:
        pos=s['source_sample_index_zero']
        columns.extend(head[1+pos*3:1+(pos+1)*3])
    labels=np.array([s['population'] for s in samples])
    rows=[];read=0;retained=0
    for frame in pd.read_csv(path,sep='\t',compression='gzip',usecols=columns,
                             dtype={head[0]:str},chunksize=20000):
        read+=len(frame);ids=frame.iloc[:,0].astype(str).to_numpy()
        values=frame.iloc[:,1:].to_numpy(dtype=np.float64).reshape(-1,len(samples),3)
        keep=np.fromiter((x.startswith('cg') for x in ids),bool,count=len(ids))
        keep&=(values[:,:,2]<.01).sum(1)>=min_detected
        chosen=values[keep]
        beta=chosen[:,:,1]/(chosen[:,:,1]+chosen[:,:,0]+100.)
        if not np.isfinite(beta).all():raise ValueError('Nonfinite beta after filtering.')
        means=np.column_stack([beta[:,labels==c].mean(1) for c in CLASSES])
        within=np.column_stack([beta[:,labels==c].var(1,ddof=1) for c in CLASSES]).mean(1)
        between=means.var(1,ddof=1);denominator=within+between
        ratio=np.divide(between,denominator,out=np.zeros_like(between),where=denominator>0)
        part=pd.DataFrame(means,columns=[f'{cohort}_mean_{c}' for c in CLASSES])
        part.insert(0,'probe_id',ids[keep]);part[f'R_{cohort}']=ratio
        part[f'{cohort}_within_variance']=within;part[f'{cohort}_between_variance']=between
        rows.append(part);retained+=len(part)
        if read%200000==0:print(json.dumps({'cohort':cohort,'rows_read':read,'retained_cg':retained}),flush=True)
    data=pd.concat(rows,ignore_index=True)
    if data.probe_id.duplicated().any():raise ValueError('Duplicate retained probe IDs.')
    return data,{'source_rows':read,'retained_cg':len(data),'minimum_detected_samples':min_detected,'selected_samples':len(samples),'gzip_read_to_eof_passed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--discovery',required=True,type=Path);p.add_argument('--external',required=True,type=Path)
    p.add_argument('--discovery-metadata',required=True,type=Path);p.add_argument('--external-metadata',required=True,type=Path)
    p.add_argument('--reference-common',required=True,type=Path);p.add_argument('--reference-top',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source_hashes={n:sha(getattr(a,n)) for n in ('discovery','external','discovery_metadata','external_metadata','reference_common','reference_top')}
    md=json.loads(a.discovery_metadata.read_text());assert len(md)==60
    discovery_samples=[]
    for i in range(18,60):
        title=md[i]['Sample_title'][0];prefix,suffix=title.rsplit('_',1)
        discovery_samples.append({'source_sample_index_zero':i,'gsm':md[i]['gsm'],'sample_title':title,
                                  'population':DISC_MAP[prefix],'donor_suffix':suffix})
    mh=headers(a.discovery)
    assert len(mh)==181 and all(mh[1+3*s['source_sample_index_zero']].startswith(f"SAMPLE {s['source_sample_index_zero']+1} ") for s in discovery_samples)
    me=pd.read_csv(a.external_metadata,dtype=str)
    selected=me[me.included.str.lower().eq('true')].copy();assert len(selected)==36
    eh=headers(a.external);array_order=[x.removesuffix(' Unmethylated signal') for x in eh[1::3]]
    assert len(array_order)==49 and len(set(array_order))==49
    external_samples=[]
    for r in selected.to_dict('records'):
        external_samples.append({'source_sample_index_zero':array_order.index(r['array_id']),
                                 'gsm':r['gsm'],'sample_title':r['sample_title'],'population':r['population'],
                                 'array_id':r['array_id']})
    # read_csv preserves source-column order, so align the label vector with that order.
    external_samples.sort(key=lambda r:r['source_sample_index_zero'])
    assert pd.Series([s['population'] for s in external_samples]).value_counts().to_dict()=={c:6 for c in CLASSES}
    discovery,dc=extract(a.discovery,discovery_samples,40,'discovery')
    external,ec=extract(a.external,external_samples,35,'validation')
    combined=discovery.merge(external,on='probe_id',how='inner',validate='one_to_one')
    ref=pd.read_csv(a.reference_common,usecols=['probe_id'])
    if len(discovery)!=481737 or len(combined)!=448510:raise ValueError('Unexpected filtered/common probe counts.')
    if set(combined.probe_id)!=set(ref.probe_id):raise ValueError('Common probe identities differ from reference.')
    x=combined[[f'discovery_mean_{c}' for c in CLASSES]].to_numpy()
    y=combined[[f'validation_mean_{c}' for c in CLASSES]].to_numpy()
    xc=x-x.mean(1,keepdims=True);yc=y-y.mean(1,keepdims=True)
    norm=np.sqrt(np.square(xc).sum(1)*np.square(yc).sum(1))
    corr=np.divide((xc*yc).sum(1),norm,out=np.full(len(combined),np.nan),where=norm>0)
    combined['pattern_corr']=np.clip(corr,-1.,1.)
    common=combined[['probe_id','R_discovery','R_validation','pattern_corr']].copy()
    common=common.set_index('probe_id').loc[ref.probe_id].reset_index()
    constant=common.loc[common.pattern_corr.isna(),'probe_id'].tolist()
    if constant!=['cg24418853']:raise ValueError(f'Unexpected undefined correlations: {constant}')
    ranked=common.sort_values(['R_discovery','probe_id'],ascending=[False,True],kind='mergesort').reset_index(drop=True)
    ranked['discovery_rank']=np.arange(1,len(ranked)+1)
    validation_rank=common.sort_values(['R_validation','probe_id'],ascending=[False,True],kind='mergesort')
    top_ref=pd.read_csv(a.reference_top).sort_values('discovery_rank')
    compatibility=[]
    for m in SIZES:
        old=top_ref.head(m).probe_id.tolist();new=ranked.head(m).probe_id.tolist()
        compatibility.append({'n_cpg':m,'same_probe_set':set(old)==set(new),'same_order':old==new,
                              'gained_ids':sorted(set(new)-set(old)),'lost_ids':sorted(set(old)-set(new))})
    if not all(r['same_probe_set'] for r in compatibility):raise ValueError('A spectral signature set changed; stop for review.')
    pattern=[];overlap=[]
    for m in ['all',100,250,500,1000,5000,10000]:
        vals=(common if m=='all' else ranked.head(m)).pattern_corr.dropna()
        pattern.append({'discovery_top_cpgs':m,'median_pattern_corr':float(vals.median()),'q25':float(vals.quantile(.25)),
                        'q75':float(vals.quantile(.75)),'n_finite':len(vals)})
        if m!='all':
            observed=len(set(ranked.head(m).probe_id)&set(validation_rank.head(m).probe_id));expected=m*m/len(common)
            overlap.append({'top_cpgs':m,'overlap':observed,'expected_uniform_overlap':expected,
                            'fold_enrichment':observed/expected,'shared_fraction':observed/m})
    common.to_csv(out/'probe_reproducibility_common_cg.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    combined.to_csv(out/'common_cell_type_profiles.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    ranked.head(10000).to_csv(out/'external_signature_top10000.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(pattern).to_csv(out/'pattern_correlation_curve.csv',index=False)
    pd.DataFrame(overlap).to_csv(out/'overlap_curve.csv',index=False)
    summary=[]
    for name in ('discovery','validation'):
        vals=common['R_'+name]
        summary.append({'cohort':name,'n_cpg':len(vals),'median_R':float(vals.median()),'q25':float(vals.quantile(.25)),
                        'q75':float(vals.quantile(.75)),'fraction_R_gt_0_5':float((vals>.5).mean()),'fraction_R_gt_0_8':float((vals>.8).mean())})
    pd.DataFrame(summary).to_csv(out/'common_R_summary.csv',index=False)
    pd.DataFrame(discovery_samples).to_csv(out/'discovery_samples.csv',index=False)
    pd.DataFrame(external_samples).to_csv(out/'external_samples.csv',index=False)
    assert all(sha(getattr(a,n))==h for n,h in source_hashes.items())
    print(json.dumps({'complete':True,'common_cg_probes':len(common),'undefined':constant,'all_pattern_summary':pattern[0],
                      'top_sets_all_preserved':all(r['same_probe_set'] for r in compatibility)}),flush=True)


if __name__=='__main__':main()
