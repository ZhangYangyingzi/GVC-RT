from io32 import *
import math
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
