#!/usr/bin/env python3
"""Render Figures 2–3 from explicit numerical tables.

Usage: python plot_fig2_fig3.py --data DATA_ROOT --out OUTPUT_ROOT
Requires numpy, pandas, matplotlib and Pillow. No fitting or network access.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from PIL import Image

BLUE='#1965A9'
DARK='#123F70'
ORANGE='#E58022'
GRAY='#5F6872'
LIGHT='#E4E8ED'
ORDER7=['CD4','CD8','B','NK','Mono','Neu','Eos']
MATRIX7=['B','CD4','CD8','Eos','Mono','NK','Neu']
MATRIX6=['B','CD4','CD8','Mono','NK','Neu']


def metrics(frame,order):
    cm=pd.crosstab(frame['true'],frame['predicted']).reindex(index=order,columns=order,fill_value=0)
    a=cm.to_numpy();n=int(a.sum());p=a/n;rows=p.sum(1);cols=p.sum(0)
    mi=math.fsum(p[i,j]*math.log2(p[i,j]/(rows[i]*cols[j])) for i in range(len(order)) for j in range(len(order)) if p[i,j]>0)
    return cm,{'n':n,'correct':int(np.trace(a)),'accuracy':float(np.trace(a)/n),
               'balanced_accuracy':float(np.mean(np.diag(a)/a.sum(1))),
               'empirical_mutual_information_bits':mi}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                        'axes.labelsize':8,'axes.titlesize':8.7,'xtick.labelsize':7.3,
                        'ytick.labelsize':7.3,'axes.linewidth':.65,
                        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                        'svg.hashsalt':'methylation_fig2_fig3',
                        'savefig.facecolor':'white'})


def tidy(ax,grid='y'):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(length=2.5,width=.6,pad=2)
    if grid:ax.grid(axis=grid,color=LIGHT,linewidth=.5,zorder=0)
    ax.set_axisbelow(True)


def panel(fig,letter,x,y,title,cx):
    fig.text(x,y,letter,color=DARK,fontsize=13,fontweight='bold',va='top')
    fig.text(cx,y-.004,title,color=DARK,fontsize=8.6,fontweight='bold',
             va='top',ha='center',linespacing=1.2)


def confusion(fig,bounds,cm):
    ax=fig.add_axes(bounds)
    cmap=LinearSegmentedColormap.from_list('counts',['#FFFFFF',BLUE,DARK])
    a=cm.to_numpy();ax.imshow(a,vmin=0,vmax=6,cmap=cmap,interpolation='nearest',aspect='equal')
    ax.set_xticks(np.arange(len(cm)),cm.columns,rotation=45,ha='right',rotation_mode='anchor')
    ax.set_yticks(np.arange(len(cm)),cm.index)
    ax.set_xlabel('Predicted cell type',fontsize=7.4,labelpad=3)
    ax.set_ylabel('True cell type',fontsize=7.4,labelpad=2)
    ax.tick_params(length=0,labelsize=7.0,pad=2)
    ax.set_xticks(np.arange(-.5,len(cm),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(cm),1),minor=True)
    ax.grid(which='minor',color='#D4DDE6',linewidth=.45)
    ax.tick_params(which='minor',length=0)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j,i,str(int(a[i,j])),ha='center',va='center',fontsize=7.6,
                    color='white' if a[i,j]>=4 else GRAY,
                    fontweight='bold' if a[i,j] else 'normal')
    for spine in ax.spines.values():spine.set_color('#A6B5C5');spine.set_linewidth(.5)
    return ax


def export(fig,out,name):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    f=folder/name
    fig.savefig(f.with_suffix('.png'),dpi=300)
    fig.savefig(f.with_suffix('.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(f.with_suffix('.pdf'),dpi=300,metadata={'Title':name,'Author':'','CreationDate':None,'ModDate':None})
    fig.savefig(f.with_suffix('.svg'),dpi=300)
    plt.close(fig)
    checks=[]
    for suffix in ('.png','.tif','.pdf','.svg'):
        p=f.with_suffix(suffix);r={'file':p.name,'bytes':p.stat().st_size}
        if suffix in ('.png','.tif'):
            with Image.open(p) as im:
                im.load();r.update({'width_px':im.width,'height_px':im.height,'dpi':[float(v) for v in im.info.get('dpi',[])],
                                   'compression':im.info.get('compression')})
        if suffix=='.tif' and p.stat().st_size>=10_000_000:raise ValueError('TIFF exceeds 10 MB.')
        checks.append(r)
    return checks


def figure2(d,out):
    ag=d['aggregation_population_summary'].set_index('population').loc[ORDER7]
    r=d['discovery_R7_summary'].iloc[0]
    distances=d['pairwise_distance_summary'].set_index('metric_group')
    nested_cm,nested=metrics(d['nested_outer_predictions'],MATRIX7)
    fixed=[]
    for size,group in d['discovery_fixed_size_predictions'].groupby('n_cpg'):
        _,m=metrics(group,MATRIX7);fixed.append({'n_cpg':int(size),**m})
    fixed=pd.DataFrame(fixed).sort_values('n_cpg')
    assert nested['n']==42 and nested['correct']==42
    assert int(r.n_cpg)==481737 and (distances.n_cpg==481737).all()
    fig=plt.figure(figsize=(7.2,5.1))
    panel(fig,'A',.012,.975,'Median binary-entropy transform\nacross cell-type mean profiles',.267)
    ax=fig.add_axes([.075,.595,.37,.285])
    y=ag.median_A_across_CpGs.to_numpy()
    ax.bar(range(7),y,color=BLUE,width=.65,zorder=3)
    ax.set_ylim(0,.65);ax.set_yticks([0,.2,.4,.6]);ax.set_xticks(range(7),ORDER7)
    ax.set_ylabel('Median h(mean beta) (bits per CpG)')
    for i,v in enumerate(y):ax.text(i,v+.014,f'{v:.3f}',ha='center',va='bottom',fontsize=7.2)
    tidy(ax)
    panel(fig,'B',.505,.975,'Cell-identity-associated variance',.764)
    axstats=fig.add_axes([.55,.615,.17,.24]);axstats.axis('off')
    axstats.text(.5,1.02,r'Distribution of $R_j$',ha='center',fontsize=8.0,color=DARK,fontweight='bold')
    for y0,label,value in ((.77,'Median',r.median_R7),(.48,'Q1',r.q25_R7),(.19,'Q3',r.q75_R7)):
        axstats.text(.0,y0,label,ha='left',va='center',fontsize=8)
        axstats.text(1.,y0,f'{value:.3f}',ha='right',va='center',fontsize=8.6,color=BLUE,fontweight='bold')
    ax=fig.add_axes([.8,.595,.18,.285])
    percentages=100*np.array([r.fraction_R7_gt_0_5,r.fraction_R7_gt_0_8])
    ax.bar([0,1],percentages,width=.58,color=BLUE,zorder=3)
    ax.set_xticks([0,1],[r'$R_j>0.5$',r'$R_j>0.8$']);ax.set_ylim(0,45)
    ax.set_ylabel('cg probes (%)',fontsize=7.5);ax.set_yticks([0,20,40])
    for i,v in enumerate(percentages):ax.text(i,v+1.5,f'{v:.2f}%',ha='center',fontsize=7.5)
    tidy(ax)
    fig.text(.075,.535,'GSE35069: 42 samples; 7 cell types; 481,737 eligible cg probes.',fontsize=7.6,color=GRAY)
    panel(fig,'C',.012,.486,'Methylome distances',.193)
    ax=fig.add_axes([.075,.17,.25,.245])
    names=['mad_within','mad_between','rms_within','rms_between']
    ds=distances.loc[names];med=ds['median'].to_numpy();q1=ds.q25.to_numpy();q3=ds.q75.to_numpy()
    ax.bar(range(4),med,color=BLUE,width=.62,zorder=3)
    ax.errorbar(range(4),med,yerr=np.vstack((med-q1,q3-med)),fmt='none',ecolor='#1C2833',capsize=2.5,linewidth=.8,zorder=4)
    ax.axvline(1.5,color='#BBC3CD',lw=.6,ls='--')
    ax.set_xticks(range(4),['MAD\nwithin','MAD\nbetween','RMS\nwithin','RMS\nbetween'])
    ax.tick_params(axis='x',labelsize=7.0);ax.set_ylim(0,.14);ax.set_yticks([0,.05,.10])
    ax.set_ylabel('Median beta distance',fontsize=7.8)
    for i,v in enumerate(med):ax.text(i,q3[i]+.006,f'{v:.4f}',ha='center',fontsize=7.1)
    tidy(ax)
    mad_ratio=float(distances.loc['mad_between','median']/distances.loc['mad_within','median'])
    rms_ratio=float(distances.loc['rms_between','median']/distances.loc['rms_within','median'])
    fig.text(.2,.065,f'Between/within: MAD {mad_ratio:.2f}×; RMS {rms_ratio:.2f}×',ha='center',fontsize=7.1,color=DARK)
    panel(fig,'D',.365,.486,'Nested donor-held-out\nclassification',.51)
    confusion(fig,[.413,.17,.225,.245],nested_cm)
    fig.text(.526,.026,'100.0% (42/42)',ha='center',fontsize=7.6,color=DARK,fontweight='bold')
    panel(fig,'E',.69,.486,'Accuracy vs\nsignature size',.858)
    ax=fig.add_axes([.74,.17,.24,.245])
    ax.plot(fixed.n_cpg,100*fixed.accuracy,color=BLUE,marker='o',ms=3.1,lw=1.15,zorder=3)
    ax.set_xscale('log');ax.set_xticks([10,100,1000,5000]);ax.set_xticklabels(['10','100','1,000','5,000'],fontsize=7)
    ax.xaxis.set_minor_locator(NullLocator());ax.set_ylim(50,108);ax.set_yticks([50,75,100])
    ax.set_xlabel('Signature size (CpGs)',fontsize=7.7);ax.set_ylabel('Accuracy (%)',fontsize=7.7)
    for row in fixed.itertuples():
        v=100*row.accuracy
        if row.n_cpg in (10,25,50,100,5000):
            ax.annotate(f'{v:.1f}',(row.n_cpg,v),xytext=(4,5) if row.n_cpg==10 else (0,5),
                        textcoords='offset points',ha='left' if row.n_cpg==10 else 'center',fontsize=6.9)
    tidy(ax)
    checks=export(fig,out,'Figure_2_discovery_profiles')
    ag[['median_A_across_CpGs']].reset_index().to_csv(out/'tables/Figure_2A_median_transform.csv',index=False)
    d['discovery_R7_summary'].to_csv(out/'tables/Figure_2B_R_summary.csv',index=False)
    ds.reset_index().to_csv(out/'tables/Figure_2C_distances.csv',index=False)
    nested_cm.to_csv(out/'tables/Figure_2D_nested_confusion.csv')
    fixed.to_csv(out/'tables/Figure_2E_fixed_size.csv',index=False)
    caption="""Figure 2. Methylation-profile summaries and cell-type recognition in GSE35069.
(A) Median, across 481,737 eligible cg probes, of h applied to the equally weighted mean beta value of the six donors in each cell type. Here h(x) = -x log2(x) - (1-x) log2(1-x), with h(0) = h(1) = 0; this is an operational transformation of array beta values. (B) Median and quartiles of the descriptive cell-identity variance ratio R_j are listed on the left; the two bars on the right give the percentages of cg probes with R_j greater than 0.5 or 0.8. R_j and percentages are displayed separately. (C) Median mean absolute beta-value distance (MAD) and root-mean-square beta-value distance (RMS) across the same 481,737 cg probes. Error bars show the interquartile range across 105 within-cell-type pairs or 756 between-cell-type pairs, not a confidence interval. Pairs share samples; no independence-based significance test is shown. (D) Confusion matrix from nested leave-one-donor-out evaluation of 42 samples and seven cell types. Rows denote true labels, columns predicted labels, and entries are sample counts. Signature size is selected inside each training fold; this matrix does not represent a single fixed signature size. (E) Descriptive fixed-size leave-one-donor-out accuracy from 10 to 5,000 CpGs; the horizontal axis is logarithmic. CD4 and CD8 denote T-cell populations, B denotes B cells, NK natural killer cells, Mono monocytes, Neu neutrophils and Eos eosinophils. All panel data and sample identities are provided in S1 Data.
"""
    (out/'captions/Figure_2_caption.txt').write_text(caption)
    return {'files':checks,'nested_metrics':nested,'mad_between_within_ratio':mad_ratio,'rms_between_within_ratio':rms_ratio}


def figure3(d,out):
    common=d['probe_reproducibility_common_cg'];overlap=d['overlap_curve'].sort_values('top_cpgs')
    patterns=d['pattern_correlation_curve'].copy();external=d['external_individual_predictions']
    assert len(common)==448510 and common.probe_id.nunique()==448510
    assert np.isfinite(common[['R_discovery','R_validation']]).all().all()
    assert common.pattern_corr.isna().sum()==1
    assert common.loc[common.pattern_corr.isna(),'probe_id'].tolist()==['cg24418853']
    rho=float(common.R_discovery.rank(method='average').corr(common.R_validation.rank(method='average')))
    rows=[]
    for (size,method),group in external.groupby(['n_cpg','method']):
        _,m=metrics(group,MATRIX6);rows.append({'n_cpg':int(size),'method':method,**m})
    curves=pd.DataFrame(rows).sort_values(['method','n_cpg'])
    cm,correlation=metrics(external[(external.n_cpg==100)&(external.method=='correlation')],MATRIX6)
    _,euclidean=metrics(external[(external.n_cpg==100)&(external.method=='euclidean')],MATRIX6)
    assert correlation['correct']==36 and euclidean['correct']==33
    fig=plt.figure(figsize=(7.2,5.1))
    panel(fig,'A',.012,.977,'Cross-cohort agreement of Rⱼ',.26)
    ax=fig.add_axes([.075,.604,.365,.277])
    ax.scatter(common.R_discovery,common.R_validation,s=.3,color=BLUE,alpha=.18,
               linewidths=0,rasterized=True)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    ax.set_xlabel('Discovery Rⱼ (GSE35069)',labelpad=2);ax.set_ylabel('External Rⱼ (GSE110554)',labelpad=2)
    ax.text(.04,.95,f'Spearman ρ = {rho:.3f}',transform=ax.transAxes,va='top',fontsize=7.5)
    tidy(ax,None)
    panel(fig,'B',.505,.977,'Overlap of independently ranked CpG sets',.752)
    ax=fig.add_axes([.57,.604,.407,.277])
    ax.plot(overlap.top_cpgs,overlap.overlap,color=BLUE,marker='o',ms=3.5,lw=1.2,label='Observed overlap')
    expected=overlap.top_cpgs.to_numpy()**2/len(common)
    assert np.allclose(expected,overlap.expected_uniform_overlap,rtol=0,atol=1e-10)
    ax.plot(overlap.top_cpgs,expected,color=ORANGE,marker='o',ms=3.3,lw=1.1,label='Uniform reference')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(80,13000);ax.set_ylim(.009,25000)
    ax.set_xticks([100,1000,10000]);ax.set_xticklabels(['100','1,000','10,000'])
    ax.set_xlabel('Top CpGs per cohort',labelpad=2);ax.set_ylabel('Shared CpGs',labelpad=2)
    for m,v,e in zip(overlap.top_cpgs,overlap.overlap,expected):
        ax.annotate(f'{int(v):,}',(m,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
    ax.legend(loc='upper left',frameon=False,fontsize=7,handlelength=1.4,handletextpad=.4)
    tidy(ax)
    fig.text(.075,.520,'448,510 shared cg probes; six matched cell types in each cohort.',fontsize=7.3,color=GRAY)
    panel(fig,'C',.012,.492,'Cell-type pattern\npreservation',.21)
    ax=fig.add_axes([.075,.16,.295,.265])
    x=np.arange(len(patterns));med=patterns.median_pattern_corr.to_numpy();q1=patterns.q25.to_numpy();q3=patterns.q75.to_numpy()
    ax.bar(x,med,color=BLUE,width=.70,zorder=3)
    ax.errorbar(x,med,yerr=np.vstack((med-q1,q3-med)),fmt='none',ecolor='#1C2833',capsize=2,linewidth=.7,zorder=4)
    labels=['All']+[f'Top\n{int(v)}' if int(v)<1000 else f'Top\n{int(v)//1000}k'
                   for v in patterns.discovery_top_cpgs.iloc[1:]]
    ax.set_xticks(x,labels);ax.tick_params(axis='x',labelsize=7.0)
    ax.set_ylim(-.2,1.13);ax.set_yticks([-.2,0,.5,1]);ax.set_ylabel('Median pattern correlation',fontsize=7.6)
    for i,v in enumerate(med):ax.text(i,.20,f'{v:.4f}',ha='center',va='bottom',color='white',fontsize=6.5,rotation=90)
    tidy(ax)
    panel(fig,'D',.405,.492,'External classification',.58)
    ax=fig.add_axes([.455,.16,.265,.265])
    for method,color,linestyle,label in [('euclidean',BLUE,'-','Euclidean (primary)'),
                                       ('correlation',ORANGE,'--','Correlation (exploratory)')]:
        z=curves[curves.method==method].sort_values('n_cpg')
        ax.plot(z.n_cpg,100*z.accuracy,color=color,marker='o',ms=3.0,lw=1.05,ls=linestyle,label=label,zorder=3)
    ax.axvline(100,color=GRAY,lw=.65,ls=':',zorder=1)
    ax.set_xscale('log');ax.set_xlim(8,13000);ax.set_ylim(45,106)
    ax.set_xticks([10,100,1000,10000]);ax.set_xticklabels(['10','100','1,000','10,000'],fontsize=6.9)
    ax.xaxis.set_minor_locator(NullLocator());ax.set_yticks([50,75,100])
    ax.set_xlabel('Signature size (CpGs)',fontsize=7.6);ax.set_ylabel('Accuracy (%)',fontsize=7.6)
    ax.annotate('33/36',(100,100*euclidean['accuracy']),xytext=(4,-13),textcoords='offset points',fontsize=7,color=DARK)
    ax.legend(loc='lower right',fontsize=6.6,frameon=False,handlelength=1.4,handletextpad=.4,borderpad=.1)
    tidy(ax)
    panel(fig,'E',.755,.492,'100-CpG correlation\n(exploratory)',.891)
    confusion(fig,[.803,.19,.173,.245],cm)
    fig.text(.89,.068,'100.0% (36/36)',ha='center',fontsize=7.5,color=DARK,fontweight='bold')
    fig.text(.075,.035,'GSE110554: 36 evaluated samples; 6 cell types. Bars in C show medians and IQRs.',fontsize=7.0,color=GRAY)
    checks=export(fig,out,'Figure_3_external_transfer')
    pd.DataFrame([{'common_cg_probes':len(common),'spearman_rho':rho}]).to_csv(out/'tables/Figure_3A_rank_correlation.csv',index=False)
    overlap.to_csv(out/'tables/Figure_3B_overlap.csv',index=False)
    patterns.to_csv(out/'tables/Figure_3C_pattern_summary.csv',index=False)
    curves.to_csv(out/'tables/Figure_3D_external_accuracy.csv',index=False)
    cm.to_csv(out/'tables/Figure_3E_exploratory_confusion.csv')
    caption="""Figure 3. Cross-cohort methylation-pattern agreement and external classification in GSE110554.
(A) Discovery and external-cohort R_j values for 448,510 common cg probes, calculated using the six shared cell types in each cohort. Each point is one probe; the displayed Spearman coefficient is descriptive. (B) The observed overlap between independently ranked top-CpG sets is compared with m squared divided by N, where m is the selected set size and N = 448,510. The uniform reference is descriptive, not a dependence-adjusted significance test. Both axes are logarithmic. (C) For each CpG, Pearson correlation compares the two six-component vectors of cell-type mean beta values. Bars show the median correlation and error bars the interquartile range across CpGs. The all-probe summary includes 448,509 finite correlations; cg24418853 has a constant external cell-type mean profile and its correlation is undefined. All ranked subsets shown have finite correlations. The CpG subsets are selected using only the discovery ranking. (D) External accuracy across signature sizes for Euclidean nearest-centroid classification (primary rule) and the exploratory correlation-distance comparison. The vertical dotted line identifies the 100-CpG signature; these curves do not select a signature size using external outcomes. (E) Confusion matrix for the exploratory correlation-distance rule with 100 CpGs. Rows denote true cell types, columns predicted types, and entries are sample counts. All 36 external samples were classified correctly by this exploratory rule; the primary Euclidean 100-CpG result is 33/36. CD4 and CD8 denote T-cell populations, B denotes B cells, Mono monocytes, NK natural killer cells and Neu neutrophils. Numerical panel data, prediction records and metadata are provided in S1 and S2 Data.
"""
    (out/'captions/Figure_3_caption.txt').write_text(caption)
    return {'files':checks,'external_euclidean_100':euclidean,'external_correlation_100':correlation,
            'common_cg_probes':len(common),'finite_pattern_correlations':int(common.pattern_corr.notna().sum()),'spearman_rho':rho}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=a.data.resolve();out=a.out.resolve()
    for sub in ('figures','tables','captions'):(out/sub).mkdir(parents=True,exist_ok=True)
    files={'aggregation_population_summary':'arrays/aggregation_population_summary.csv',
           'discovery_R7_summary':'arrays/discovery_R7_summary.csv',
           'pairwise_distance_summary':'arrays/pairwise_distance_summary.csv',
           'probe_reproducibility_common_cg':'arrays/probe_reproducibility_common_cg.csv.gz',
           'overlap_curve':'arrays/overlap_curve.csv','pattern_correlation_curve':'arrays/pattern_correlation_curve.csv',
           'nested_outer_predictions':'model_fits/nested_outer_predictions.csv',
           'discovery_fixed_size_predictions':'model_fits/discovery_fixed_size_predictions.csv.gz',
           'external_individual_predictions':'model_fits/external_individual_predictions.csv.gz'}
    d={k:pd.read_csv(data/v) for k,v in files.items()}
    style();result={'figure2':figure2(d,out),'figure3':figure3(d,out)}
    print(json.dumps({'figures':2,'out':str(out),
                      'figure2_nested':result['figure2']['nested_metrics'],
                      'figure3_primary':result['figure3']['external_euclidean_100']}),flush=True)


if __name__=='__main__':main()
