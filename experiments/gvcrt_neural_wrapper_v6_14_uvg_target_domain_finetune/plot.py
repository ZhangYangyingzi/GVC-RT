"""Predetermined measured RD and training/validation trajectories only."""
import statistics
from v614_io import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    for sp in SPLITS:
        folder=ROOT/'results'/sp;macro=read(folder/'all_qp_summary.csv');fid=read(folder/'fid_pooled.csv');methods=list(dict.fromkeys(r['method'] for r in macro))
        for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
            fig,ax=plt.subplots(figsize=(9,6))
            for m in methods:
                rows=sorted([r for r in (fid if metric=='FID' else macro) if r['method']==m],key=lambda r:float(r['bpp']))
                ax.plot([float(r['bpp']) for r in rows],[float(r[metric]) for r in rows],'o-',markersize=3,label=m)
            ax.set(xlabel='Bitrate (actual bpp)',ylabel=metric+' (lower is better)',title=sp+' / '+('training_exposed' if sp=='train_fit' else 'held_out_from_this_training'));ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(folder/f'Rate_{metric}.png',dpi=180);plt.close(fig)
    training=read(ROOT/'training_logs/uvg_only.csv')
    for metric in ('loss','LPIPS','DISTS','rate_bpp','beta_rate','proxy_L1','gradient_norm'):
        fig,ax=plt.subplots(figsize=(9,5));ax.plot([int(r['adaptation_step']) for r in training],[float(r[metric]) for r in training],linewidth=.5,alpha=.6)
        ax.set(xlabel='Adaptation updates (source step 1000)',ylabel=metric,title='uvg_only / raw per-update values');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(ROOT/'results'/f'training_{metric}.png',dpi=150);plt.close(fig)
    for sp in ('validation','train_fit'):
        rows=read(ROOT/'results'/sp/'adaptation_history.csv')
        for metric in ('LPIPS','DISTS','FloLPIPS','bpp'):
            fig,ax=plt.subplots(figsize=(9,5))
            for q in (0,4,9):
                rs=[r for r in rows if int(r['QP'])==q];steps=sorted({int(r['adaptation_step']) for r in rs})
                ax.plot(steps,[statistics.mean(float(r[metric]) for r in rs if int(r['adaptation_step'])==s) for s in steps],'o-',label=f'QP {q}')
            ax.set(xlabel='Adaptation updates',ylabel=metric,title=sp+' / real RANS evaluation');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(ROOT/'results'/sp/f'History_{metric}.png',dpi=160);plt.close(fig)
    print('PLOTS PASS',flush=True)
if __name__=='__main__':main()
