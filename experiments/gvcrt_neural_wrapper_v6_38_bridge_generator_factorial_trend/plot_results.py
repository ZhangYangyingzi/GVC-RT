"""Compact descriptive plots from completed CSVs; no point selection."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 rows=list(csv.DictReader((ROOT/'results/evaluation.csv').open()))
 fig,axes=plt.subplots(2,3,figsize=(12,7));methods=['O','B','G','BG']
 for ax,metric in zip(axes.flat,['LPIPS','DISTS','FloLPIPS','PSNR','L1']):
  for mi,m in enumerate(methods):
   for vi,v in enumerate([0,1]):
    rr=sorted([r for r in rows if r['method']==m and int(r['video'])==v],key=lambda r:float(r['actual_bpp']))
    ax.plot([float(r['actual_bpp']) for r in rr],[float(r[metric]) for r in rr],['-o','--s'][vi],color=f'C{mi}',label=f"{m}: {rr[0]['sequence']}",markersize=3)
  ax.set(xlabel='Actual full RANS bpp',ylabel=metric+(' (higher better)' if metric=='PSNR' else ' (lower better)'));ax.grid(alpha=.2)
 axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=8)
 fig.suptitle('Single seed; two HEVC-B videos; identical streams; O0 / B,G,BG100 updates');fig.tight_layout();fig.savefig(out/'same_stream_rd.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11,3))
 for b in ['B','G','BG']:
  rr=list(csv.DictReader((ROOT/f'results/train_{b}.csv').open()));xs=[int(r['step']) for r in rr]
  axes[0].plot(xs,[json.loads(r['losses'])['total_loss'] for r in rr],label=b)
  axes[1].plot(xs,[json.loads(r['losses'])['DISTS'] for r in rr],label=b)
  axes[2].plot(xs,[float(r['preclip_norm']) for r in rr],label=b)
 for ax,label in zip(axes,['Training total objective','Training DISTS','Combined gradient norm before clip']):ax.set(xlabel='Committed optimizer updates',ylabel=label);ax.grid(alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(out/'training_trends.png',dpi=140);plt.close(fig)
 print('PLOTS_COMPLETE')
