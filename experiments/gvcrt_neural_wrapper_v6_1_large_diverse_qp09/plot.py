import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v61_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);args=p.parse_args();root=ROOT/'results'/args.split;rows=read(root/'all_qp_summary.csv');out=root/'rd_curves';out.mkdir(exist_ok=True)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(8,5))
        for method in dict.fromkeys(r['method'] for r in rows):
            own=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['external_qp']))
            ax.plot([float(r['kbps']) for r in own],[float(r[metric]) for r in own],'o-',label='V6.1 selected' if method=='selected' else method)
        ax.set_xscale('log');ax.set_xlabel('Real kbps @20fps' if args.split=='virat720' else 'Real kbps @30fps' if args.split!='validation' else 'Real kbps @source fps');ax.set_ylabel(metric);ax.set_title(args.split+' | force_zero_thres=0.12');ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'Rate_{metric}.{ext}',dpi=180)
        plt.close(fig)
    dump(out/'manifest.json',dict(status='PASS',files={p.name:sha(p) for p in out.glob('Rate_*')}))
if __name__=='__main__':main()
