import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_utils import ROOT,read

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=('validation','final'),required=True); args=p.parse_args()
    rows=read(f'{args.split}_qp_summary.csv'); fids=read(f'fid_{args.split}.csv')
    for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        for method in dict.fromkeys(r['method'] for r in rows):
            subset=sorted((r for r in (fids if metric=='FID' else rows) if r['method']==method),key=lambda r:float(r['mean_real_kbps' if metric=='FID' else 'kbps']))
            assert len(subset)==4
            x=[float(r['mean_real_kbps' if metric=='FID' else 'kbps']) for r in subset]; y=[float(r[metric]) for r in subset]
            assert all(math.isfinite(v) for v in x+y)
            ax.plot(x,y,'o-',label=method)
        ax.set_xscale('log'); ax.set_xlabel('Mean real bitrate (kbps, log scale)'); ax.set_ylabel(metric+' (lower is better)')
        ax.set_title(args.split.capitalize()+' — '+metric); ax.grid(True,alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ('png','pdf'): fig.savefig(ROOT/'rd_curves'/f'{args.split}_Rate_{metric}.{suffix}',dpi=180)
        plt.close(fig)
    print(args.split.upper(),'PLOTS PASS')

if __name__=='__main__': main()
