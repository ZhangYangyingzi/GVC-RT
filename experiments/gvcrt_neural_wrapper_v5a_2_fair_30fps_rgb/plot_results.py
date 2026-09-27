"""Plot only measured V5-A.2 points with explicit minimum bitrate labels."""
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from audit_utils import *
LABELS={'original':'Original','v41_20000':'V4.1-20k','clip8':'clip8'}
COLORS={'original':'#333333','v41_20000':'#0072B2','clip8':'#D55E00'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('ulong','uvg'),required=True);args=p.parse_args()
    rows=read(f'{args.dataset}_all_qp_summary.csv');out=ROOT/'rd_curves'/args.dataset;out.mkdir(parents=True,exist_ok=True)
    records=[]
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(9,6));minimum=[];all_rates=[]
        for j,method in enumerate(METHODS):
            own=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['external_qp']))
            x=[float(r['mean_real_kbps']) for r in own];y=[float(r[metric]) for r in own];all_rates+=x
            ax.plot(x,y,'o-',label=LABELS[method],color=COLORS[method],markersize=4)
            i=min(range(len(x)),key=lambda n:x[n]);qp=int(own[i]['external_qp'])
            ax.annotate(f'QP{qp}\n{x[i]:.1f} kbps',(x[i],y[i]),xytext=(20,15-30*j),textcoords='offset points',
                        color=COLORS[method],fontsize=8,arrowprops=dict(arrowstyle='-',color=COLORS[method]))
            minimum.append(f'{LABELS[method]}: {x[i]:.1f} kbps (QP{qp})')
            records.append(dict(dataset=args.dataset,metric=metric,method=method,minimum_kbps=x[i],minimum_external_qp=qp))
        ax.set_xscale('log');low=min(all_rates);high=max(all_rates)
        ticks=[low]+[t for t in (10,20,50,100,200,500,1000,2000,5000,10000,20000,50000) if low*1.15<t<=high]
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off()
        ax.set_xlim(low*.85,high*1.08);ax.set_xlabel('Mean real bitrate (kbps, 30 fps)');ax.set_ylabel(metric)
        ax.set_title(('Fresh U-Long' if args.dataset=='ulong' else 'UVG')+' | 30 fps, 64 frames, canonical RGB')
        ax.grid(True,alpha=.25);ax.legend(loc='upper right')
        fig.text(.13,.015,'Minimum measured bitrate\n'+' | '.join(minimum),fontsize=9)
        fig.tight_layout(rect=(0,.09,1,1))
        for suffix in ('png','pdf'):fig.savefig(out/f'Rate_{metric}.{suffix}',dpi=180)
        plt.close(fig)
    dump(f'parts/{args.dataset}_plot_audit.json',dict(status='PASS',records=records,x_axis='Mean real bitrate (kbps, 30 fps)',scale='log'))
if __name__=='__main__':main()

