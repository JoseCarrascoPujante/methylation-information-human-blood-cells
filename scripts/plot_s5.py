"""Draw S5 Fig from the aggregation and spectral tables in data/.

Example:
    python scripts/plot_s5.py --data data --out rebuilt_aggregation_spectrum

Requires numpy, pandas, matplotlib and Pillow. Inputs are read-only.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

P=argparse.ArgumentParser()
P.add_argument('--data',type=Path,required=True,help='Data directory containing arrays and spectral')
P.add_argument('--out',type=Path,required=True,help='Output directory for figures, tables and legends')
A=P.parse_args()
DATA=A.data.resolve(); OUT=A.out.resolve()
FIG=OUT/'figures'; TAB=OUT/'tables'; LEG=OUT/'legends'
for d in (FIG,TAB,LEG):d.mkdir(parents=True,exist_ok=True)

BLUE='#28639A'; ORANGE='#CB7E15'; GRAY='#59636F'; LIGHT='#E5E9ED'; INK='#18222B'
ORDER7=['CD4','CD8','B','NK','Mono','Neu','Eos']
ORDER6=ORDER7[:-1]
def read(rel):
    p=DATA/rel
    return pd.read_csv(p)

def save_table(frame,name):
    frame.to_csv(TAB/name,index=False,compression='gzip' if name.endswith('.gz') else None)

def save_figure(fig,name):
    stem=FIG/name
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':name.replace('_',' '),'Author':'','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def title(ax,letter,main,sub):
    ax.set_title(f'{letter}  {main}',loc='left',fontweight='bold',fontsize=9.7,pad=22)
    ax.text(0,1.033,sub,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.1,color=GRAY)

def tidy(ax,grid_axis='y'):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8,length=3,width=.65)
    if grid_axis:ax.grid(axis=grid_axis,color=LIGHT,linewidth=.6,zorder=0)
    ax.set_axisbelow(True)

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','svg.hashsalt':'human_methylation_scientific_figures',
    'savefig.facecolor':'white',
})

# S5 Fig: aggregation on the complete eligible discovery set and a separately
# defined spectrum of the external-transfer signature.
ag=read('arrays/aggregation_population_summary.csv').set_index('population').loc[ORDER7].reset_index()
don=read('arrays/donor_entropy_summary.csv')
eig=read('spectral/signature_full_eigenvalue_spectra.csv')
sp=read('spectral/signature_summary_current.csv')
assert len(ag)==7 and ag.n_cpg.nunique()==1 and int(ag.n_cpg.iloc[0])==481737
assert len(don)==42 and set(don.population)==set(ORDER7)
assert all(don[don.population==p].donor.nunique()==6 for p in ORDER7)
assert np.allclose(ag.sum_A_bits-ag.sum_B_bits,ag.jensen_gap_bits,atol=1e-8,rtol=0)
assert (ag.jensen_gap_bits>=0).all()
N=int(ag.n_cpg.iloc[0]);x=np.arange(len(ORDER7))
ag['mean_h_donor_mean_beta']=ag.sum_A_bits/N
ag['mean_donor_h_beta']=ag.sum_B_bits/N
ag['mean_jensen_difference_bits_per_cpg']=ag.jensen_gap_bits/N
for _,r in ag.iterrows():
    assert np.isclose(don.loc[don.population==r.population,'mean_h_beta'].mean(),r.mean_donor_h_beta,rtol=0,atol=1e-12)

e100=eig[eig.n_cpg==100].sort_values('eigen_index').copy()
s100=sp[sp.n_cpg==100].iloc[0]
v=e100.eigenvalue.to_numpy();q=v/v.sum()
assert len(v)==35 and (v>0).all() and np.all(np.diff(v)<=0)
assert np.allclose(q,e100.normalized_eigenvalue,rtol=0,atol=1e-12)
dpr=float(1/np.dot(q,q));der=float(np.exp(-np.dot(q,np.log(q))))
assert np.isclose(dpr,s100.participation_ratio_dimension,atol=1e-10,rtol=0)
assert np.isclose(der,s100.effective_rank,atol=1e-10,rtol=0)

fig,axs=plt.subplots(2,2,figsize=(7.2,6.55))
fig.subplots_adjust(left=.125,right=.97,top=.905,bottom=.09,wspace=.48,hspace=.76)
ax=axs[0,0]
title(ax,'A','Donor aggregation','481,737 cg probes; 7 cell types')
for j,r in ag.iterrows():
    ax.plot([j-.12,j+.12],[r.mean_donor_h_beta,r.mean_h_donor_mean_beta],color=GRAY,lw=.8,zorder=2)
ax.scatter(x-.12,ag.mean_donor_h_beta,s=23,color=BLUE,marker='o',label=r'mean $h(\beta)$',zorder=3)
ax.scatter(x+.12,ag.mean_h_donor_mean_beta,s=26,color=ORANGE,marker='^',label=r'$h$(mean $\beta$)',zorder=3)
ax.set_xticks(x,ORDER7);ax.set_xlim(-.5,6.5)
ylow=float(don.mean_h_beta.min()-.012);yhigh=float(don.mean_h_beta.max()+.012)
ax.set_ylim(ylow,yhigh)
ax.set_ylabel('Mean value (bits per CpG)')
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.35),frameon=False,ncol=2,columnspacing=.9,handletextpad=.25)
tidy(ax)

ax=axs[0,1]
title(ax,'B','Donor-level profile summaries','481,737 cg probes; 7 cell types')
for j,pop in enumerate(ORDER7):
    z=don[don.population==pop].sort_values('donor')
    offsets=np.linspace(-.23,.23,len(z))
    ax.scatter(j+offsets,z.mean_h_beta,s=17,color=GRAY,alpha=.75,zorder=3,label='Donor' if j==0 else None)
    ax.scatter([j],[z.mean_h_beta.mean()],s=120,color=ORANGE,marker='_',linewidths=1.8,zorder=4,label='Mean' if j==0 else None)
ax.set_xticks(x,ORDER7);ax.set_xlim(-.5,6.5);ax.set_ylim(ylow,yhigh)
ax.set_ylabel(r'Mean $h(\beta)$ (bits per CpG)')
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.35),frameon=False,ncol=2,columnspacing=1.1,handletextpad=.3)
tidy(ax)

ax=axs[1,0]
title(ax,'C','Mean aggregation effect','481,737 cg probes; 7 cell types')
ax.bar(x,ag.mean_jensen_difference_bits_per_cpg,color=ORANGE,width=.62,zorder=3)
ax.set_xticks(x,ORDER7);ax.set_xlim(-.5,6.5)
ax.set_ylim(0,float(ag.mean_jensen_difference_bits_per_cpg.max()*1.2))
ax.set_ylabel('Mean Jensen difference\n(bits per CpG)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:.3f}'))
tidy(ax)

ax=axs[1,1]
title(ax,'D','Eigenvalue spectrum','100 CpGs; 36 samples; 6 cell types')
ax.plot(e100.eigen_index,q,color=BLUE,lw=1.1,marker='o',ms=2.5,zorder=3)
ax.set_yscale('log');ax.set_xlim(.5,35.5);ax.set_ylim(float(q.min()*.65),1.15)
ax.set_xticks([1,10,20,30,35]);ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Normalized eigenvalue')
ax.text(.97,.97,f'$d_{{\\mathrm{{PR}}}}={dpr:.4f}$\n$d_{{\\mathrm{{ER}}}}={der:.4f}$\nDimensionless',transform=ax.transAxes,ha='right',va='top',fontsize=8.2,linespacing=1.45)
tidy(ax)
save_figure(fig,'S5_Fig_aggregation_and_spectrum')
save_table(ag[['population','n_cpg','sum_A_bits','sum_B_bits','mean_h_donor_mean_beta','mean_donor_h_beta']], 'S5_FigA_population_aggregation.csv')
don_ordered=don.assign(population=pd.Categorical(don.population,categories=ORDER7,ordered=True)).sort_values(['population','donor'])
save_table(don_ordered, 'S5_FigB_donor_entropy.csv')
save_table(ag[['population','n_cpg','jensen_gap_bits','mean_jensen_difference_bits_per_cpg']], 'S5_FigC_mean_jensen_difference.csv')
save_table(e100, 'S5_FigD_spectrum_100_cpg.csv')
save_table(pd.DataFrame([dict(n_cpg=100,discovery_samples=36,cell_types=6,numerical_rank=len(v),participation_ratio_dimension=dpr,spectral_effective_rank=der,mean_marginal_sum_bits=float(s100.mean_marginal_sum_bits))]), 'S5_FigD_spectral_summary.csv')

legend3='''S5 Fig. Donor aggregation and spectral dimensionality.
(A-C) The discovery analysis includes 481,737 eligible cg probes, seven cell types, and six donors per cell type. The binary-entropy function h(x) = -x log2(x) - (1-x) log2(1-x), with h(0) = h(1) = 0, is applied operationally to array beta values. (A) The mean of donor-specific h(beta) values is compared with h applied to the equally weighted donor-mean beta, with both quantities then averaged across the eligible probes. Connecting segments pair the two quantities within a cell type. (B) Each gray point is one donor's mean h(beta) across the eligible probes; orange marks show the mean over the six donors. (C) Bars show the mean Jensen difference per CpG: h(mean beta) minus mean h(beta), averaged across the eligible probes. These are descriptive population summaries. (D) For the external-transfer signature of 100 CpGs, the CpG correlation spectrum is computed across the 36 discovery samples from six shared cell types. The 35 retained positive eigenvalues are normalized to sum to one and shown on a logarithmic vertical scale. With 36 samples, the centered data have rank at most 35. Participation-ratio dimension and spectral effective rank are calculated from that distribution and reported as dimensionless quantities. No error bars or confidence intervals are represented. CD4 and CD8: T-cell populations; B: B cells; NK: natural killer cells; Mono: monocytes; Neu: neutrophils; Eos: eosinophils.
'''
(LEG/'S5_Fig_aggregation_and_spectrum.txt').write_text(legend3,encoding='utf-8')


print(json.dumps({'figure':'S5_Fig_aggregation_and_spectrum','output':str(OUT)}))
