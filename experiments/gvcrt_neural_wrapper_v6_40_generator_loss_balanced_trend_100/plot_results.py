"""Descriptive real-stream RD and common-component training curves only."""
from pathlib import Path
import csv,json
try:
 import matplotlib
except ModuleNotFoundError:
 import os,sys
 alternative='/data1/anaconda3_new/anaconda_program/bin/python'
 os.execv(alternative,[alternative,'-B',str(Path(__file__).resolve())])
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 out=ROOT/'figures';out.mkdir(exist_ok=True);rows=list(csv.DictReader((ROOT/'evaluation.csv').open()));methods=['O_original','G_basic','G_dists']
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for ax,metric in zip(axes.flat,['LPIPS','DISTS','FloLPIPS','PSNR','L1']):
  for mi,m in enumerate(methods):
   for vi,v in enumerate([0,1]):
    rr=sorted([r for r in rows if r['method']==m and int(r['video'])==v],key=lambda r:float(r['actual_bpp']))
    ax.plot([float(r['actual_bpp']) for r in rr],[float(r[metric]) for r in rr],['-o','--s'][vi],color=f'C{mi}',label=f"{m}: {rr[0]['sequence']}",markersize=3)
  ax.set(xlabel='Actual complete RANS bpp',ylabel=metric+(' (higher better)' if metric=='PSNR' else ' (lower better)'));ax.grid(alpha=.2)
 axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=7)
 fig.suptitle('Basic loss versus +0.5 DISTS; single seed; two heldout videos');fig.tight_layout();fig.savefig(out/'same_stream_rd.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,6))
 for b in ['G_basic','G_dists']:
  rr=list(csv.DictReader((ROOT/f'results/train_{b}.csv').open()));xs=[int(r['step']) for r in rr]
  for ax,k in zip(axes.flat[:3],['LPIPS','DISTS','reconstruction_L1']):ax.plot(xs,[json.loads(r['losses'])[k] for r in rr],label=b);ax.set_ylabel(k)
  for ax,k in zip(axes.flat[3:],['preclip_norm','postclip_norm','Generator update norm']):
   values=[json.loads(r['parameter_delta_norms'])['generator']['norm'] if k=='Generator update norm' else float(r[k]) for r in rr]
   ax.plot(xs,values,label=b);ax.set_ylabel(k)
 for ax in axes.flat:ax.set_xlabel('Committed updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Shared loss observations and gradients; total objectives are not compared');fig.tight_layout();fig.savefig(out/'training_common_components.png',dpi=140);plt.close(fig)
 print('PLOTS_PASS',flush=True)
