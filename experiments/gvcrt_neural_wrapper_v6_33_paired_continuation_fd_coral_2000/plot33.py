"""Objective measured RD and separate three-QP diagnostic figures."""
from pathlib import Path
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'all_qp_summary.csv').open()))
branches=['B_C1_lr2','C_C1_lr2_FD','D_C1_lr2_mean_CORAL'];main=['original','B1_initial']+[f'{b}_{s}' for s in (1000,2000) for b in branches]
for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
 for phase in ('main','diagnostic_3QP'):
  methods=main if phase=='main' else main+[f'{b}_1500' for b in branches]
  for metric in ('LPIPS','DISTS','FloLPIPS','FID','D_mu','D_cov'):
   fig,ax=plt.subplots(figsize=(9,5.5))
   for m in methods:
    rr=sorted([r for r in rows if r['dataset']==d and r['method']==m and (phase=='main' or int(r['QP']) in (0,5,9))],key=lambda r:float(r['bpp']))
    assert len(rr)==(10 if phase=='main' else 3)
    ax.plot([float(r['bpp']) for r in rr],[float(r[metric]) for r in rr],marker='o',markersize=3,linewidth=1,label=m)
   ax.set_xlabel('Actual bpp (dataset mean)');ax.set_ylabel(metric);ax.set_title(d+' / '+metric+(' / measured 10 QPs' if phase=='main' else ' / diagnostic: QP0,5,9 only'));ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2);fig.tight_layout()
   p=ROOT/'results'/d/('' if phase=='main' else phase)/f'Rate_{metric}.png';p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=150);plt.close(fig)
