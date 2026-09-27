import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,MaxNLocator,NullFormatter
from audit_utils import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('ulong','uvg'),required=True);args=p.parse_args()
    rows=read(f'{args.dataset}_all_qp_summary.csv')
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(7,4.8))
        for method in METHODS:
            own=sorted((r for r in rows if r['method']==method),key=lambda r:int(r['external_qp']))
            assert [int(r['external_qp']) for r in own]==list(range(10))
            x=[float(r['mean_real_kbps']) for r in own];y=[float(r[metric]) for r in own]
            ax.plot(x,y,'o-',label=method)
            for r,xx,yy in zip(own,x,y):ax.annotate(r['external_qp'],(xx,yy),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_xscale('log');ax.set_xlabel('Mean real bitrate (kbps, log scale)');ax.set_ylabel(metric+' (lower is better)')
        lo,hi=ax.get_xlim();ticks=[v for v in MaxNLocator(nbins=5).tick_values(lo,hi) if lo<=v<=hi and v>0]
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(True,alpha=.25)
        ax.set_title(('Fresh U-Long' if args.dataset=='ulong' else 'UVG')+' | external QP 0–9')
        ax.legend(fontsize=8);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(ROOT/'rd_curves'/args.dataset/f'Rate_{metric}.{ext}',dpi=180)
        plt.close(fig)
    print(args.dataset,'PLOTS PASS',flush=True)

if __name__=='__main__':main()
