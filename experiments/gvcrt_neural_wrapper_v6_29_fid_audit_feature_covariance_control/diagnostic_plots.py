from io21 import *
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 # The exclusion29 view reuses measured points and recomputes its own pooled FID.
 import statistics
 d='mcljcv_exclude_overlap29';folder=ROOT/'results/ip_-1'/d
 raw=read(folder/'per_sequence_all_qp.csv');pooled=read(folder/'fid_pooled.csv');macro=[]
 for m in METHODS:
  for q in range(10):
   rs=[r for r in raw if r['method']==m and int(r['QP'])==q];assert len(rs)==29
   f=next(r for r in pooled if r['method']==m and int(r['QP'])==q)
   macro.append(dict(dataset=d,method=m,QP=q,**{k:statistics.mean(float(r[k]) for r in rs) for k in ('bpp','LPIPS','DISTS','FloLPIPS')},
       FID=float(f['FID']),samples=1856,sequences=29,rate_axis='dataset mean actual bpp',
       aggregation='equal29 video weights for perceptual metrics; pooled1856 frame FID; no extra encoding'))
 write(folder/'all_qp_summary.csv',macro)
 for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
  fig,ax=plt.subplots(figsize=(9,6))
  for m in METHODS:
   rs=sorted((r for r in macro if r['method']==m),key=lambda r:r['bpp'])
   ax.plot([r['bpp'] for r in rs],[r[metric] for r in rs],'o-',label=m,markersize=3)
  ax.set(xlabel='Bitrate (dataset mean actual bpp)',ylabel=metric+' (lower is better)',title=d+' IP=-1')
  ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(folder/f'Rate_{metric}.png',dpi=180);plt.close(fig)
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  folder=ROOT/'results/covariance_diagnostic'/d;raw=read(folder/'fid_decomposition_raw.csv');grid=read(folder/'fid_decomposition_equal_rate.csv')
  for metric in ('D_mu','D_cov','FID'):
   fig,ax=plt.subplots(figsize=(8,5))
   for m in METHODS:
    rs=sorted((r for r in raw if r['method']==m),key=lambda x:float(x['bpp']));ax.plot([float(r['bpp']) for r in rs],[float(r[metric]) for r in rs],'o-',label=m,markersize=3)
   ax.set(xlabel='Dataset mean actual bpp',ylabel=metric,title=f'{d} measured {metric}');ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();p=ROOT/'plots/FID_components'/d;p.mkdir(parents=True,exist_ok=True);fig.savefig(p/f'{metric}_measured.png',dpi=160);plt.close(fig)
 for field in ('stat_loss','stat_B1_ratio','stat_cosine','global_clip_coefficient'):
  import json
  rows=[json.loads(x) for x in (ROOT/'branches/B_covstats/training_logs/train.jsonl').read_text().splitlines()]
  fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['adaptation_step'] for r in rows],[r[field] for r in rows],linewidth=.6);ax.set(xlabel='Valid optimizer updates',ylabel=field,title=field);ax.grid(alpha=.3);fig.tight_layout();p=ROOT/'plots/training';p.mkdir(parents=True,exist_ok=True);fig.savefig(p/(field+'.png'),dpi=160);plt.close(fig)
 print('DIAGNOSTIC PLOTS PASS',flush=True)
if __name__=='__main__':main()
