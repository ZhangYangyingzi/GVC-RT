from preflight import *
from data import load
from evaluate import csvwrite
def main():
 import numpy as np
 from scipy.linalg import svdvals
 from scipy.interpolate import PchipInterpolator
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 records=[load(p)['result'] for p in (R/'results/points').glob('*.json')];csvwrite(R/'results/per_video_all_q.csv',records);pooled=[];equal=[]
 manifests=load(R/'eval_manifest.json');methods=['A_original','B_rgb_P','C_latent_P'];qs=load(R/'protocol.json')['q_grid']
 for d,vs in manifests.items():
  for m in methods:
   for q in qs:
    points=[r for r in records if r['dataset']==d and r['method']==m and r['q']==q];assert len(points)==len(vs)
    aa=[];bb=[]
    for r in sorted(points,key=lambda x:x['video']):
     with np.load(r['feature']) as z:aa.append(z['real'].astype(np.float64));bb.append(z['reconstruction'].astype(np.float64))
    a=np.concatenate(aa);b=np.concatenate(bb);ac=a-a.mean(0);bc=b-b.mean(0);dm=float(((a.mean(0)-b.mean(0))**2).sum());dc=float((ac**2).sum()/(len(a)-1)+(bc**2).sum()/(len(b)-1)-2*svdvals(ac@bc.T/(len(a)-1)).sum())
    pooled.append(dict(dataset=d,method=m,q=q,bpp=np.mean([x['bpp'] for x in points]),FID=dm+dc,D_mu=dm,D_cov=dc,frames=len(a),ddof=1,GT_hash=hashlib.sha256(a.tobytes()).hexdigest()))
  dest=R/'results'/d;dest.mkdir(parents=True,exist_ok=True)
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   fig,ax=plt.subplots(figsize=(6,4))
   for m in methods:
    curve=[r for r in pooled if r['dataset']==d and r['method']==m] if metric=='FID' else [dict(bpp=np.mean([r['bpp'] for r in records if r['dataset']==d and r['method']==m and r['q']==q]),**{metric:np.mean([r[metric] for r in records if r['dataset']==d and r['method']==m and r['q']==q])}) for q in qs]
    curve=sorted(curve,key=lambda x:x['bpp']);ax.plot([r['bpp'] for r in curve],[r[metric] for r in curve],marker='o',label=m)
   ax.set_xlabel('Bitrate (actual bpp; local GLC rANS)');ax.set_ylabel(metric);ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(dest/f'Rate_{metric}.png',dpi=160);plt.close(fig)
  for video in list(range(len(vs)))+['pooled']:
   for metric in (('FID',) if video=='pooled' else ('LPIPS','DISTS','FloLPIPS')):
    curves={m:[r for r in (pooled if video=='pooled' else records) if r['dataset']==d and r['method']==m and (video=='pooled' or r['video']==video)] for m in methods};lo=max(min(r['bpp'] for r in c) for c in curves.values());hi=min(max(r['bpp'] for r in c) for c in curves.values());values=None
    if hi>lo and all(len({r['bpp'] for r in c})==10 for c in curves.values()):
     grid=np.linspace(np.log(lo),np.log(hi),100);values={}
     for m,c in curves.items():
      c=sorted(c,key=lambda x:x['bpp']);values[m]=PchipInterpolator(np.log([r['bpp'] for r in c]),[r[metric] for r in c],extrapolate=False)(grid)
    for m,ref in [('B_rgb_P','A_original'),('C_latent_P','A_original'),('C_latent_P','B_rgb_P')]:
     row=dict(dataset=d,video=video,metric=metric,method=m,reference=ref,common_bpp_min=lo,common_bpp_max=hi,rate_type='real_local_nonofficial_rans',grid='100 uniform ln(bpp)',delta_definition='method-reference',status='available' if values is not None else 'unavailable')
     if values is not None:row.update(method_mean=float(values[m].mean()),reference_mean=float(values[ref].mean()),delta=float((values[m]-values[ref]).mean()),better_fraction=float((values[m]<values[ref]).mean()))
     equal.append(row)
 csvwrite(R/'results/pooled_fid.csv',pooled);csvwrite(R/'results/equal_rate_per_sequence.csv',equal)
 summary=[]
 for key in {(r['dataset'],r['metric'],r['method'],r['reference']) for r in equal}:
  rs=[r for r in equal if (r['dataset'],r['metric'],r['method'],r['reference'])==key];good=[r for r in rs if r['status']=='available'];row=dict(zip(('dataset','metric','method','reference'),key));row.update(covered=len(good),expected=len(rs),aggregation='equal-video weights; FID fixed pooled',status='available' if len(good)==len(rs) else 'unavailable')
  if len(good)==len(rs):row.update(method_mean=float(np.mean([r['method_mean'] for r in rs])),reference_mean=float(np.mean([r['reference_mean'] for r in rs])),delta=float(np.mean([r['delta'] for r in rs])))
  summary.append(row)
 csvwrite(R/'results/equal_rate_summary.csv',summary)
 for d,vs in manifests.items():
  from PIL import Image
  from data import Frames
  f=Frames(vs[0]);dest=R/'previews'/d;dest.mkdir(parents=True,exist_ok=True)
  for i in (16,32,63):Image.fromarray(f.array(i)).save(dest/f'GT_{i}.png')
  f.close()
 trainrows=[]
 for m in methods[1:]:
  rows=[load(p) for p in sorted((R/f'branches/{m}/receipts').glob('*.json'))]
  assert len(rows)==1000
  (R/f'branches/{m}/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  trainrows.extend([dict(branch=m,step=r['step'],domain=r['domain'],q=r['q'],**r['loss'],gradient_preclip=r['gradient_preclip'],clip_coefficient=r['clip_coefficient'],parameter_delta=r['parameter_delta'],seconds=r['seconds']) for r in rows])
 csvwrite(R/'results/training.csv',trainrows)
 expected=sum(len(v)*30 for v in manifests.values());assert len(records)==expected
 save('final_integrity.json',dict(status='PASS',updates={m:load(R/f'branches/{m}/status.json')['updates'] for m in methods[1:]},evaluation_completed=len(records),evaluation_expected=expected,per_point_frames=64,per_point_transitions=63,rate_type='real_local_nonofficial_rans',source_weights_unchanged=all(sha(r['path'])==r['sha256'] for r in load(R/'audits/checkpoint_audit.json')['models'])))
 (R/'REPORT.md').write_text('# Execution record\n\nB/C: 1000 effective updates each. See final_integrity.json for exact evaluation counts. Real bytes are from the pre-existing local non-official GLC rANS extension, including GOP and frame headers. Original author pretraining data unknown. Mixed U-Long/UVG training; independent data pools and UVG diagnostic results are separate. No checkpoint selection. Numeric records are in results/.\n')
if __name__=='__main__':main()
