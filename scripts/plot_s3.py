#!/usr/bin/env python3
"""Create Figure S3 and summary tables from nine-sample regional calculations."""
import argparse
import csv
import json
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

POPULATIONS = ['CD4 T cells', 'B cells', 'Monocytes']
DONORS = [54, 51, 49]
COLORS = ['#28639A', '#CB7E15']
MARKERS = {54:'o', 51:'s', 49:'^'}


def read(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source=args.results;out=args.out
    for sub in ['figures','tables','captions']:(out/sub).mkdir(parents=True,exist_ok=True)
    sites=read(source/'pat_population_aggregation_summary.csv')
    windows=read(source/'pat_population_window_entropy_summary.csv')
    donor_sites=read(source/'pat_donor_site_entropy_points.csv')
    donor_windows=read(source/'pat_donor_window_entropy_summary.csv')
    sp={r['cell_population']:r for r in sites};wp={r['cell_population']:r for r in windows}
    ds={(r['cell_population'],int(r['donor'])):r for r in donor_sites}
    dw={(r['cell_population'],int(r['donor'])):r for r in donor_windows}
    assert set(sp)==set(wp)==set(POPULATIONS)
    assert len(ds)==len(dw)==9
    n_sites={int(r['n_sites']) for r in sites};n_windows={int(r['n_blocks']) for r in windows}
    assert len(n_sites)==len(n_windows)==1
    n_sites=n_sites.pop();n_windows=n_windows.pop()
    write(out/'tables/Table_S6A_regional_single_site.csv',sites)
    write(out/'tables/Table_S6B_regional_four_CpG.csv',windows)
    combined=[]
    for pop in POPULATIONS:
        for donor in DONORS:
            a,b=ds[(pop,donor)],dw[(pop,donor)]
            combined.append({**b,'n_sites':a['n_sites'],'mean_site_entropy':a['mean_site_entropy']})
    write(out/'tables/Table_S6C_regional_donor_values.csv',combined)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8.5,'axes.linewidth':0.6,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(7.2,4.4),sharey=True)
    fig.subplots_adjust(left=.14,right=.985,bottom=.33,top=.76,wspace=.27)
    ybase={pop:i*1.45 for i,pop in enumerate(POPULATIONS)}
    offsets={54:-.23,51:-.07,49:.09}
    data=[]
    def record(panel,pop,donor,quantity,value,level,gsm=''):
        data.append({'panel':panel,'cell_population':pop,'donor':donor,'gsm':gsm,
            'quantity':quantity,'value':float(value),'aggregation':level,
            'unit':'bits per CpG' if panel=='A' else 'bits per four-CpG window'})
    for i,(ax,panel) in enumerate(zip(axes,['A','B','C'])):
        for pop in POPULATIONS:
            base=ybase[pop]
            if panel=='A':
                for donor in DONORS:
                    row=ds[(pop,donor)];value=float(row['mean_site_entropy'])
                    ax.scatter(value,base+offsets[donor],s=15,marker=MARKERS[donor],color=COLORS[1],zorder=4)
                    record(panel,pop,donor,'h(p)',value,'donor mean over common sites',row['gsm'])
                values=[float(sp[pop][key]) for key in ['mean_h_mean_fraction','mean_donor_h_fraction']]
                names=['h(mean donor p)','mean donor h(p)']
            else:
                prefix='observed' if panel=='B' else 'depth20'
                keys=[prefix+'_joint',prefix+'_marginal_sum'];names=['Joint entropy','Sum of marginal entropies']
                for donor in DONORS:
                    row=dw[(pop,donor)];values_d=[float(row[k]) for k in keys]
                    y=base+offsets[donor]
                    ax.plot(values_d,[y,y],color='#B8B8B8',lw=.65,zorder=2)
                    for v,c,name in zip(values_d,COLORS,names):
                        ax.scatter(v,y,s=15,marker=MARKERS[donor],color=c,zorder=4)
                        record(panel,pop,donor,name,v,'donor mean over common windows',row['gsm'])
                values=[float(wp[pop][k]) for k in keys]
            y=base+.34
            ax.plot(values,[y,y],color='#777777',lw=.9,zorder=2)
            for value,color,name in zip(values,COLORS,names):
                ax.scatter(value,y,s=31,marker='D',color=color,edgecolor='white',lw=.45,zorder=5)
                record(panel,pop,'',name,value,'population summary with equal donor weights')
        ax.set_ylim(3.57,-.50)
        ax.set_yticks([ybase[p]+.03 for p in POPULATIONS],POPULATIONS)
        if i:ax.tick_params(axis='y',left=False)
        ax.tick_params(axis='both',length=2.5,color='#666666')
        ax.grid(axis='x',color='#E7E7E7',lw=.5);ax.set_axisbelow(True)
        for side in ['top','right','left']:ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#777777')
        ax.set_xlabel('Entropy (bits / CpG)' if panel=='A' else 'Entropy (bits / window)',labelpad=6)
        titles=['Donor aggregation','Observed patterns','Depth-20 expectation']
        subtitles=[f'{n_sites:,} common CpGs',f'{n_windows:,} common four-CpG windows',f'{n_windows:,} common four-CpG windows']
        ax.text(0,1.36,panel,transform=ax.transAxes,fontsize=11,weight='bold',va='bottom')
        ax.text(0,1.19,titles[i],transform=ax.transAxes,fontsize=9,weight='bold',va='bottom')
        ax.text(0,1.06,subtitles[i],transform=ax.transAxes,fontsize=7.7,color='#555555',va='bottom')
        labels=['h(mean p)','h(p)'] if panel=='A' else ['Joint entropy','Marginal sum']
        handles=[Line2D([],[],color=c,marker='o',ls='none',markersize=4.5,label=t) for c,t in zip(COLORS,labels)]
        ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False,
                  fontsize=8,handletextpad=.45,borderpad=0,labelspacing=.45)
    a_values=[r['value'] for r in data if r['panel']=='A']
    low=min(a_values);high=max(a_values);pad=max((high-low)*.12,.008)
    axes[0].set_xlim(low-pad,high+pad)
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    bc_values=[r['value'] for r in data if r['panel'] in ['B','C']]
    low=math.floor((min(bc_values)-.06)*10)/10;high=math.ceil((max(bc_values)+.06)*10)/10
    for ax in axes[1:]:
        ax.set_xlim(low,high);ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    handles=[Line2D([],[],color='#666666',marker=MARKERS[d],ls='none',markersize=4,label=f'Donor {d}') for d in DONORS]
    handles.append(Line2D([],[],color='#666666',marker='D',ls='none',markersize=5,label='Population summary'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.015),ncol=4,frameon=False,
               fontsize=8,handletextpad=.45,columnspacing=1.2)
    stem=out/'figures/S3_Fig_regional_sequencing'
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':'Regional methylation across three donors and three cell types'})
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    write(out/'tables/S3_Fig_plot_data.csv',data)
    caption=(
        'S3 Fig. Regional methylation summaries from nine sequencing samples. '
        'CD4 T cells, B cells and monocytes were available from the same three donors (54, 51 and 49). '
        'Small circles, squares and triangles represent individual donors; diamonds represent population summaries. '
        f'(A) Binary entropy over {n_sites:,} common CpGs retained from 20,000 candidates on chr1 '
        '(hg38 CpG indices 100,001–120,000 inclusive), requiring at least 10 C/T calls at each site in every sample. '
        'Orange donor points show the site mean of h(p) for each donor. Orange diamonds show their equally weighted mean; '
        'blue diamonds show the site mean of h applied to the equally weighted donor-mean methylated fraction. '
        'These values are in bits per CpG. '
        f'(B) Observed plug-in joint entropy of the 16 complete four-CpG patterns (blue) and the sum of four marginal '
        f'binary entropies (orange), calculated from the same complete fragments in {n_windows:,} common non-overlapping '
        f'windows ({100*n_windows/5000:.2f}% of 5,000 candidates). Every retained window has at least 20 complete '
        'fragments in each of the nine samples. (C) Conditional expected plug-in entropies after sampling 20 complete '
        'fragments without replacement from each observed window-pattern count vector, calculated analytically using '
        'hypergeometric probabilities. Donor points in B and C are unweighted means over the common windows; '
        'diamonds are unweighted means over the three donors. Connectors join quantities for the same donor or '
        'population summary. B and C use the same horizontal scale and units of bits per four-CpG window. '
        'The points describe the three donors; no uncertainty intervals are shown.\n')
    (out/'captions/S3_Fig_caption.txt').write_text(caption,encoding='utf-8')
    print(json.dumps({'figure':'S3_Fig_regional_sequencing','common_sites':n_sites,'common_windows':n_windows,'output':str(out)}))


if __name__=='__main__':main()
