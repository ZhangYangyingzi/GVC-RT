import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
from v5_utils import ROOT,METHODS,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('validation','final'),required=True);args=p.parse_args()
    rows=read(f'{args.split}_qp_summary.csv')
    for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        for method in METHODS:
            ps=sorted((r for r in rows if r['method']==method),key=lambda r:float(r['kbps']))
            assert len(ps)==4
            x=[float(r['kbps']) for r in ps];y=[float(r[metric]) for r in ps]
            assert all(math.isfinite(v) for v in x+y)
            ax.plot(x,y,'o-',label=method)
        ax.set_xscale('log');ax.set_xlabel('Mean real bitrate (kbps, log scale)');ax.set_ylabel(metric+' (lower is better)')
        xmin,xmax=ax.get_xlim()
        ticks=[v for v in MaxNLocator(nbins=5).tick_values(xmin,xmax) if xmin<=v<=xmax and v>0]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(args.split.capitalize());ax.grid(True,alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(ROOT/'rd_curves'/f'{args.split}_Rate_{metric}.{ext}',dpi=180)
        plt.close(fig)
    print(args.split.upper(),'PLOTS PASS')

if __name__=='__main__':main()
