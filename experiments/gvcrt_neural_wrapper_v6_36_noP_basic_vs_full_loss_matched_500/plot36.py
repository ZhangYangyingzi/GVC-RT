import csv,json,platform,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
with open(R/'results/all_rd_points.csv',newline='') as f:points=list(csv.DictReader(f))
with open(R/'eval_manifest.json') as f:manifest=json.load(f)
methods=['O_original','B_basicLoss_noP_500','F_fullLoss_noP_500'];datasets=['ulong','uvg_holdout','hevc_b'];metrics=['LPIPS','DISTS','FloLPIPS']
(R/'plots').mkdir(exist_ok=True)
for ds in datasets:
 for met in metrics:
  fig,ax=plt.subplots(figsize=(9,5.5))
  for v in manifest[ds]:
   for m in methods:
    rr=sorted([x for x in points if x['dataset']==ds and int(x['video'])==v['video_index'] and x['method']==m],key=lambda x:float(x['actual_bpp']))
    ax.plot([float(x['actual_bpp']) for x in rr],[float(x[met]) for x in rr],marker='o',alpha=.55,label=f'{m} / video {v["video_index"]}' if v['video_index']==manifest[ds][0]['video_index'] else None)
  ax.set_xscale('log');ax.set_xlabel('Actual RANS bpp');ax.set_ylabel(met+' (lower is better)');ax.set_title(f'{ds}: per-video measured points, QP 0/5/9');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{met}_per_video.png',dpi=150);plt.close(fig)
 # equal video mean over each video's own shared ln-rate interval, x is normalized position.
 for met in metrics:
  fig,ax=plt.subplots(figsize=(8,5))
  for m in methods:
   x=[];vals=[]
   for gi in range(20):
    row=[z for z in csv.DictReader(open(R/'results/common_rate_grid.csv')) if z['dataset']==ds and z['method']==m and z['metric']==met and int(z['grid_index'])==gi]
    if row:vals.append(sum(float(z['value']) for z in row)/len(row));x.append(gi/19)
   ax.plot(x,vals,marker='o',label=m)
  ax.set_xlabel('Normalized position within each video’s own common ln(bpp) interval');ax.set_ylabel(met+' (lower is better)');ax.set_title(f'{ds}: equal-video common-rate curves');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{met}_equal_video.png',dpi=150);plt.close(fig)
with open(R/'results/pooled_fid.csv',newline='') as f:fid=list(csv.DictReader(f))
for ds in datasets:
 fig,ax=plt.subplots(figsize=(8,5))
 for m in methods:
  rr=sorted([x for x in fid if x['dataset']==ds and x['method']==m],key=lambda x:float(x['mean_actual_bpp']))
  ax.plot([float(x['mean_actual_bpp']) for x in rr],[float(x['FID']) for x in rr],marker='o',label=m)
 ax.set_xscale('log');ax.set_xlabel('Dataset mean actual bpp');ax.set_ylabel('Pooled FID (lower is better)');ax.set_title(f'{ds}: pooled FID, 3 measured QP anchors');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_pooled_fid.png',dpi=150);plt.close(fig)
json.dump({'python':sys.version,'platform':platform.platform(),'matplotlib':matplotlib.__version__,'numpy':__import__('numpy').__version__,'plots':len(list((R/'plots').glob('*.png')))},open(R/'audits/plot_environment.json','w'),indent=2)
print('plots created',len(list((R/'plots').glob('*.png'))))
