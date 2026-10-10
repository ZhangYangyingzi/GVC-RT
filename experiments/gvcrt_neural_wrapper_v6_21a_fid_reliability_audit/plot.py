"""Objective numeric charts from completed paired diagnostics."""
from io21 import *
import numpy as np
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def finish(fig,name):
 p=ROOT/'plots'/name;p.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(p,dpi=140);plt.close(fig)
def main():
 pooled=load(ROOT/'results/fid_pooled_expanded.json');sample=load(ROOT/'results/fid_sample_size_summary.json');boot=load(ROOT/'results/fid_bootstrap_delta.json')
 for d in DATASETS:
  for metric in ['FID','KID']:
   fig,ax=plt.subplots(figsize=(7,4))
   for m in METHODS:
    rows=sorted([r for r in pooled if r['dataset']==d and r['method']==m],key=lambda x:x['bpp']);ax.plot([r['bpp']for r in rows],[r[metric]for r in rows],marker='.',label=m)
   ax.set(xlabel='Actual bpp',ylabel=metric,title=f'{d} | {metric} | extended frame pool');ax.legend(fontsize=7);ax.grid(alpha=.2);finish(fig,f'{d}_{metric.lower()}_vs_bpp.png')
   eq=load(ROOT/f'results/equal_rate_{metric.lower()}.json');fig,ax=plt.subplots(figsize=(7,4))
   for m in METHODS[1:]:
    rows=[r for r in eq if r['dataset']==d and r['method']==m];ax.plot([r['bpp']for r in rows],[r['delta']for r in rows],label=m+' − original')
   ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Actual bpp',ylabel=f'{metric} delta',title=f'{d} | {metric} delta | common rate interval');ax.legend(fontsize=7);ax.grid(alpha=.2);finish(fig,f'{d}_{metric.lower()}_delta_vs_bpp.png')
  for q in [0,4,9]:
   fig,ax=plt.subplots(figsize=(7,4))
   for m in METHODS:
    rr=sorted([r for r in sample if r['dataset']==d and r['method']==m and r['QP']==q],key=lambda x:x['sample_size']);ax.errorbar([r['sample_size']for r in rr],[r['mean']for r in rr],yerr=[r['std']for r in rr],marker='.',capsize=3,label=m)
   ax.set(xlabel='Sample size',ylabel='FID mean ± std',title=f'{d} | QP{q} | 100 paired resamples');ax.legend(fontsize=7);ax.grid(alpha=.2);finish(fig,f'{d}_qp{q}_fid_sample_size.png')
  for q in range(10):
   fig,ax=plt.subplots(figsize=(7,4))
   for m in METHODS[1:]:ax.hist([r['delta']for r in boot if r['dataset']==d and r['method']==m and r['QP']==q],bins=35,alpha=.5,label=m+' − original')
   ax.set(xlabel='FID delta',ylabel='Bootstrap count',title=f'{d} | QP{q} | 1000 paired bootstraps');ax.axvline(0,color='k',lw=.7);ax.legend(fontsize=7);finish(fig,f'{d}_qp{q}_bootstrap_delta_histogram.png')
 print('PLOTS PASS',len(list((ROOT/'plots').glob('*.png'))))
if __name__=='__main__':main()
