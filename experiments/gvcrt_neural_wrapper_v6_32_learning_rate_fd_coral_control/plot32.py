"""Raw RD points; unavailable branches explicit in legend; no curve smoothing."""
from io32 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 points=read(ROOT/'evaluation_points.csv');pools=read(ROOT/'pooled_fid.csv')
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  base='mcljcv' if d=='mcljcv29' else d;vs=load(ROOT/'eval_manifest.json')[base]
  if d=='mcljcv29':vs=[v for v in vs if 'videoSRC17' not in v['name']]
  ids={v['video_index'] for v in vs}
  for phase,step in (('main',1000),('early_diagnostic',250)):
   models=['original','B1_initial']+[f'{b}_{step}' for b in BRANCHES]
   for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
    fig,ax=plt.subplots(figsize=(7.5,5))
    for m in models:
     curve=[]
     for q in ((0,5,9) if step==250 else range(10)):
      if metric=='FID':
       r=next((x for x in pools if x['dataset']==d and x['method']==m and int(x['QP'])==q and x['status']=='available'),None)
       if r:curve.append((float(r['bpp']),float(r['FID'])))
      else:
       rs=[r for r in points if r['dataset']==base and r['method']==m and int(r['QP'])==q and int(r['video']) in ids]
       if len(rs)==len(ids):curve.append((sum(float(r['bpp']) for r in rs)/len(rs),sum(float(r[metric]) for r in rs)/len(rs)))
     if curve:
      curve.sort();ax.plot([x for x,y in curve],[y for x,y in curve],'o-',label=m,markersize=4)
     else:ax.plot([],[],label=m+' (unavailable)',linestyle='--')
    ax.set_xlabel('Actual bitrate (bpp)');ax.set_ylabel(metric+' (lower is better)');ax.set_title(d+' — '+phase);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();p=ROOT/'results'/d/phase/f'Rate_{metric}.png';p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=160);plt.close(fig)
 preview=load(ROOT/'audits/fixed_previews.json')['records']
 for d in ('hevc_b','mcljcv'):
  models=['GT','original','B1_initial']+[f'{b}_1000' for b in BRANCHES]
  fig,axes=plt.subplots(3,7,figsize=(21,7))
  for i,frame in enumerate((16,32,63)):
   for j,m in enumerate(models):
    ax=axes[i,j];r=next((r for r in preview if r['dataset']==d and r['method']==m and r.get('frame')==frame and r['status']=='available'),None)
    if r:ax.imshow(plt.imread(r['path']));label=m+('' if m=='GT' else f'\n{r["bpp"]:.6f} bpp')
    else:ax.text(.5,.5,'unavailable',ha='center',va='center',transform=ax.transAxes);label=m
    ax.set_title(label+f'\nframe {frame}',fontsize=8);ax.axis('off')
  fig.suptitle(d+' first video, QP5 (fixed preview IDs)');fig.tight_layout();fig.savefig(ROOT/'previews'/d/'fixed_comparison.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
