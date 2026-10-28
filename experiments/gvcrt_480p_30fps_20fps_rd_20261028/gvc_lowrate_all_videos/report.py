import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;B=R.parent;sys.path.insert(0,str(B))
from report480 import fid
from common480 import load,sha
cfg=load(B/'config.json');videos=load(B/'manifest.json')['videos'];thresholds=[.12,.15,.18,.21,.25,.28,.32,.36,.4,.45,.5,.55,.6,.7,.8,1.,2.,4.,8.];methods=['C_C1_lr2_FD_1000','original'];expected=len(videos)*len(thresholds)*2
source=B/'h266_qp54_63/results/rd_summary.csv'
assert (B/'h266_qp54_63/complete.json').exists() and source.exists()
with source.open(newline='') as f:base=list(csv.DictReader(f))
base=[r for r in base if not (r['method'] in methods and int(r['rate_index'])==0)]
new=[];per=[]
for dataset,n in cfg['dataset_counts'].items():
 vs=[v for v in videos if v['dataset']==dataset];assert len(vs)==n
 for method in methods:
  for idx,th in enumerate(thresholds):
   records=[load(R/'points'/dataset/v['name']/method/f't{th:g}.json') for v in vs]
   for r,v in zip(records,vs):
    assert r['status']=='PASS' and r['encoder_decoder_state_hash_verified'] and r['source_rgb_sha256']==v['rgb_sha256'] and r['threshold']==th and r['frames']==64 and r['rate_accounting_fps']==20 and sha(r['bitstream_path'])==r['bitstream_sha256'] and sha(r['feature_path'])==r['feature_sha256']
    per.append({k:r[k] for k in ('dataset','sequence','method','threshold','kbps','PSNR','LPIPS','DISTS','real_bytes')})
   gt=np.concatenate([np.load(r['reference_feature_path']) for r in records]);pred=np.concatenate([np.load(r['feature_path']) for r in records]);value=fid(gt,pred)
   new.append(dict(dataset=dataset,method=method,rate_index=100+idx,sequences=n,frames=n*64,kbps=float(np.mean([r['kbps'] for r in records])),bpp=float(np.mean([r['real_bytes']*8/(64*480*854) for r in records])),PSNR=float(np.mean([r['PSNR'] for r in records])),LPIPS=float(np.mean([r['LPIPS'] for r in records])),DISTS=float(np.mean([r['DISTS'] for r in records])),FID=value,threshold=th));print('SUMMARY',dataset,method,th,flush=True)
with (R/'per_sequence_extra.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(per[0]));w.writeheader();w.writerows(per)
rows=base+new
with (R/'rd_summary.csv').open('w',newline='') as f:
 keys=list(base[0])+['threshold'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
out=R/'results0-50';out.mkdir(exist_ok=True)
for dataset in cfg['dataset_counts']:
 d=out/dataset;d.mkdir(exist_ok=True)
 for metric in ('LPIPS','PSNR','DISTS','FID'):
  fig,ax=plt.subplots(figsize=(10,7));vis=[]
  for method in cfg['methods']+['GLC']:
   points=sorted((r for r in rows if r['dataset']==dataset and r['method']==method),key=lambda r:float(r['kbps']));xs=[float(r['kbps']) for r in points];ys=[float(r[metric]) for r in points]
   if not xs:continue
   line,=ax.plot(xs,ys,marker='o',markersize=3,label=method)
   for x,y in zip(xs,ys):
    if 0<=x<=50:vis.append(y);ax.annotate(f'{x:.2f}',(x,y),xytext=(3,4),textcoords='offset points',fontsize=5,color=line.get_color(),clip_on=True)
   for x1,x2,y1,y2 in zip(xs,xs[1:],ys,ys[1:]):
    if x1<50<x2:vis.append(y1+(y2-y1)*(50-x1)/(x2-x1))
  ax.set_xlim(0,50)
  if vis:
   lo,hi=min(vis),max(vis);margin=(hi-lo)*.1 or max(abs(lo)*.1,.01);ax.set_ylim(lo-margin,hi+margin)
  ax.set_xlabel('Bitrate (kbps, normalized to 20 fps)');ax.set_ylabel(metric+(' (dB, higher is better)' if metric=='PSNR' else ' (lower is better)'));ax.set_title(f'{dataset} | 854×480 center crop | 64 frames | GVC threshold sweep');ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
  for ext in ['png','pdf','svg']:fig.savefig(d/f'Rate_{metric}.{ext}',dpi=180)
  plt.close(fig)
(R/'complete.json').write_text(json.dumps(dict(status='COMPLETE',points=expected,dataset_counts=cfg['dataset_counts'],thresholds=thresholds,methods=methods,comparison_source=str(source)),indent=2)+'\n')
