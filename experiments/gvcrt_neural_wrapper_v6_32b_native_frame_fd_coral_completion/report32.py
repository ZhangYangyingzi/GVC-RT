"""Verified historical numeric aggregation helpers only; terminal handling is completion_report32b.py."""
from io32 import *
import math,collections,subprocess
def fields(rows,keys):return [{k:r.get(k) for k in keys} for r in rows]
def same_qp_summary(records,pools):
 """Fixed QP5 raw diagnostics, with measured rate differences stated explicitly."""
 rows=[]
 for step in (250,1000):
  methods=['original','B1_initial']+[f'{b}_{step}' for b in BRANCHES]
  for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
   base='mcljcv' if d=='mcljcv29' else d
   vs=load(ROOT/'eval_manifest.json')[base]
   if d=='mcljcv29':vs=[v for v in vs if 'videoSRC17' not in v['name']]
   ids={v['video_index'] for v in vs}
   for m in methods:
    rr=[r for r in records if r['dataset']==base and r['method']==m and r['QP']==5 and r['video'] in ids]
    row=dict(phase='250 diagnostic' if step==250 else '1000 main',dataset=d,method=m,QP=5,expected_videos=len(ids),completed_videos=len(rr),comparison='same QP; measured bpp can differ')
    if len(rr)!=len(ids):row.update(status='unavailable',reason='incomplete fixed video pool')
    else:
     row.update(status='available',**{k:sum(float(r[k]) for r in rr)/len(rr) for k in ('bpp','LPIPS','DISTS','FloLPIPS')})
     pool=next((r for r in pools if r['dataset']==d and r['method']==m and r['QP']==5 and r['status']=='available'),None)
     if pool:row.update(**{k:pool[k] for k in ('FID','D_mu','D_cov')})
    rows.append(row)
 write(ROOT/'same_qp_summary.csv',rows)
 return rows
def pooled(records):
 import numpy as np
 from scipy.linalg import svdvals
 out=[]
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  base='mcljcv' if d=='mcljcv29' else d;vs=load(ROOT/'eval_manifest.json')[base]
  if d=='mcljcv29':vs=[v for v in vs if 'videoSRC17' not in v['name']]
  for m in sorted({r['method'] for r in records}):
   for q in range(10):
    points=[next((r for r in records if r['dataset']==base and r['video']==v['video_index'] and r['method']==m and r['QP']==q),None) for v in vs]
    if any(p is None for p in points):
     out.append(dict(dataset=d,method=m,QP=q,status='unavailable',reason='incomplete fixed video pool',completed_videos=sum(p is not None for p in points),expected_videos=len(vs)));continue
    real=[];rec=[]
    for r in points:
     with np.load(r['feature_path']) as z:
      a=z['real'].astype(np.float64);b=z['reconstruction'].astype(np.float64);assert a.shape==b.shape==(64,2048) and np.isfinite(a).all() and np.isfinite(b).all();real.append(a);rec.append(b)
    a=np.concatenate(real);b=np.concatenate(rec);ma=a.mean(0);mb=b.mean(0);za=a-ma;zb=b-mb;dm=float(np.square(ma-mb).sum());dc=float(np.square(za).sum()/(len(a)-1)+np.square(zb).sum()/(len(b)-1)-2*svdvals(za@zb.T/(len(a)-1),check_finite=True).sum());fid=dm+dc
    assert all(math.isfinite(x) for x in (fid,dm,dc))
    out.append(dict(dataset=d,method=m,QP=q,status='available',FID=fid,D_mu=dm,D_cov=dc,bpp=sum(r['bpp'] for r in points)/len(points),sample_count=len(a),video_count=len(points),ddof=1,aggregation='fixed dataset UID pool; covariance not averaged per video',GT_pool_sha256=hashlib.sha256(a.tobytes()).hexdigest(),reconstruction_pool_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
 return out
def equal_rate(records,pools,early=False):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 methods=['original','B1_initial']+[f'{b}_{250 if early else 1000}' for b in BRANCHES]
 comparisons=[(methods[3],methods[2]),(methods[4],methods[3]),(methods[5],methods[3]),(methods[4],methods[5]),(methods[4],methods[2]),(methods[5],methods[2])]+[(m,r) for m in methods[2:] for r in methods[:2]]
 rows=[];gridrows=[]
 def compare(curves,d,v,metric):
  counts={m:len(curves.get(m,[])) for m in methods};expect=3 if early else 10
  # Baselines diagnostic view uses same QP0/5/9, never compare 3-QP means to 10-QP main.
  if any(n!=expect for n in counts.values()):
   reason='incomplete six-method common-rate matrix';lo=hi=None;values=None
  else:
   xs={m:np.array([p['bpp'] for p in curves[m]],float) for m in methods}
   if any(len(set(x))!=len(x) or np.any(x<=0) for x in xs.values()):reason='duplicate/nonpositive bpp';lo=hi=None;values=None
   else:
    lo=max(float(x.min()) for x in xs.values());hi=min(float(x.max()) for x in xs.values());values=None;reason='no positive common rate interval'
    if hi>lo:
     grid=np.exp(np.linspace(np.log(lo),np.log(hi),100));values={}
     for m,x in xs.items():
      order=np.argsort(x);y=np.array([p[metric] for p in curves[m]],float);values[m]=PchipInterpolator(np.log(x[order]),y[order],extrapolate=False)(np.log(grid));assert np.isfinite(values[m]).all()
     for k,bpp in enumerate(grid):gridrows.append(dict(dataset=d,video=v,metric=metric,index=k,bpp=float(bpp),**{m:float(values[m][k]) for m in methods}))
  for m,r in comparisons:
   row=dict(dataset=d,video=v,metric=metric,method=m,reference=r,rate_axis='actual bpp',grid='100 uniform ln(bpp)',shared_bpp_min=lo,shared_bpp_max=hi,delta_definition='method-reference',covered_points=counts,expected_points=expect,aggregation='video-equal-weight' if v!='pooled' else 'fixed pooled features; mean dataset bpp',phase='250 diagnostic' if early else '1000 main')
   if values is None:row.update(status='unavailable',reason=reason,method_mean=None,reference_mean=None,delta=None,relative_change_percent=None,better_grid_fraction=None)
   else:
    a=float(values[m].mean());b=float(values[r].mean());row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,better_grid_fraction=float((values[m]<values[r]).mean()))
   rows.append(row)
 for d,vs in load(ROOT/'eval_manifest.json').items():
  for v in vs:
   curves={m:[r for r in records if r['dataset']==d and r['video']==v['video_index'] and r['method']==m and (not early or r['QP'] in (0,5,9))] for m in methods}
   for metric in ('LPIPS','DISTS','FloLPIPS'):compare(curves,d,v['name'],metric)
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  curves={m:[r for r in pools if r['dataset']==d and r['method']==m and r['status']=='available' and (not early or r['QP'] in (0,5,9))] for m in methods}
  for metric in ('FID','D_mu','D_cov'):compare(curves,d,'pooled',metric)
 return rows,gridrows
