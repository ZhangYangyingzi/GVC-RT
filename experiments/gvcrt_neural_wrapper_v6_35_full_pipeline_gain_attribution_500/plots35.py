import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
points=rows(R/'results/all_rd_points.csv');grid=rows(R/'results/common_rate_grid.csv');pooled=rows(R/'results/pooled_fid.csv');bounds=rows(R/'results/common_rate_bounds.csv')
manifest=json.load(open(R/'eval_manifest.json'));methods=['original','N_oursLoss_noP_500','P_oursLoss_withP_500'];metrics=['LPIPS','DISTS','FloLPIPS']
(R/'plots').mkdir(exist_ok=True)
for ds in ('ulong','uvg_holdout','hevc_b'):
 for metric in metrics:
  fig,ax=plt.subplots(figsize=(8,5))
  for v in manifest[ds]:
   for method in methods:
    rr=sorted([p for p in points if p['dataset']==ds and int(p['video'])==int(v['video_index']) and p['method']==method],key=lambda x:float(x['actual_bpp']))
    ax.plot([float(p['actual_bpp']) for p in rr],[float(p[metric]) for p in rr],marker='o',alpha=.34,label=method if int(v['video_index'])==int(manifest[ds][0]['video_index']) else None)
  ax.set_xscale('log');ax.set_xlabel('Actual RANS bpp');ax.set_ylabel(metric+' (lower is better)');ax.set_title(f'{ds}: per-video measured RD points');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{metric}_rd.png',dpi=150);plt.close(fig)
  b=next(x for x in bounds if x['dataset']==ds)
  if b['status']!='available':continue
  fig,ax=plt.subplots(figsize=(8,5))
  for method in methods:
   rr=[x for x in grid if x['dataset']==ds and x['method']==method and x['metric']==metric]
   if not rr:continue
   rr.sort(key=lambda x:int(x['grid_index']));ax.plot([float(x['actual_bpp']) for x in rr],[float(x['value']) for x in rr],marker='o',label=method)
  ax.set_xscale('log');ax.set_xlabel('Actual RANS bpp (dataset-wide common interval)');ax.set_ylabel(metric+' (lower is better)');ax.set_title(f'{ds}: equal-video mean, piecewise-linear ln-bpp grid');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{metric}_common_rate.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5))
 for method in methods:
  rr=sorted([x for x in pooled if x['dataset']==ds and x['method']==method],key=lambda x:float(x['mean_actual_bpp']))
  ax.plot([float(x['mean_actual_bpp']) for x in rr],[float(x['FID']) for x in rr],marker='o',label=method)
 ax.set_xscale('log');ax.set_xlabel('Dataset mean actual bpp');ax.set_ylabel('Pooled FID (lower is better)');ax.set_title(f'{ds}: pooled FID anchors');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_pooled_fid.png',dpi=150);plt.close(fig)
print('PLOTS_WRITTEN')
