"""Fixed numeric visualizations; existing plotting environment, no metric changes."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def rows(n):return list(csv.DictReader((root/n).open()))
points=rows('intervention_points.csv');pooled=rows('intervention_fid_pooled.csv')
for d in ('hevc_b','mcljcv'):
 for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
  fig,ax=plt.subplots(figsize=(7,5))
  for m in ('R00','R10','R01','R11','R11_center'):
   curve=[]
   for q in (0,5,9):
    rr=[r for r in (pooled if metric=='FID' else points) if r.get('pool',r.get('dataset'))==d and r['method']==m and int(r['QP'])==q]
    if rr:curve.append((sum(float(r['bpp']) for r in rr)/len(rr),sum(float(r[metric]) for r in rr)/len(rr)))
   curve.sort()
   if curve:ax.plot([x for x,y in curve],[y for x,y in curve],marker='o',label=m)
  ax.set_xlabel('Actual bitrate (bpp)');ax.set_ylabel(metric+' (lower is better)');ax.set_title(d+' fixed cross cells');ax.grid(alpha=.25)
  if ax.lines:ax.legend()
  p=root/'plots/intervention'/d/f'Rate_{metric}.png';p.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(p,dpi=160);plt.close(fig)
for d in ('HEVC','MCL30','MCL29'):
 rr=[r for r in rows('correction_gap.csv') if r['pool']==d]
 fig,ax=plt.subplots(figsize=(7,5))
 for typ in ('RAW','BIAS','AFFINE'):
  r=sorted([r for r in rr if r['type']==typ],key=lambda x:int(x['QP']))
  if r:ax.plot([int(x['QP']) for x in r],[float(x['gap_corrected']) for x in r],marker='o',label=typ)
 ax.set_xlabel('External QP');ax.set_ylabel('FID B1 minus original');ax.set_title(d+' oracle P32-63 diagnostic');ax.grid(alpha=.25)
 if ax.lines:ax.legend()
 p=root/'plots/correction'/f'{d}_oracle_gap.png';p.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(p,dpi=160);plt.close(fig)
