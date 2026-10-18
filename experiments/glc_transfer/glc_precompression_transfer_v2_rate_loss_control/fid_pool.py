"""Fixed-pool ddof=1 FID with hash-keyed, CPU-only resumable cache."""
from preflight import *
from data import load
def components(a,b):
 import numpy as np
 from scipy.linalg import svdvals
 assert a.shape==b.shape and len(a)>1 and np.isfinite(a).all() and np.isfinite(b).all()
 ac=a-a.mean(0);bc=b-b.mean(0);dm=float(((a.mean(0)-b.mean(0))**2).sum());dc=float((ac**2).sum()/(len(a)-1)+(bc**2).sum()/(len(b)-1)-2*svdvals(ac@bc.T/(len(a)-1)).sum());assert dm+dc>=-1e-7
 return dict(FID=max(0.,dm+dc),D_mu=dm,D_cov=dc,frames=len(a),ddof=1,GT_hash=hashlib.sha256(a.tobytes()).hexdigest())
def point(job):
 import numpy as np
 d,m,q,scope,points=job;sig=[dict(path=r['feature'],sha256=sha(r['feature'])) for r in points];path=f'cache/fid_statistics/{d}_{m}_{q}_{scope}.json'
 if (R/path).exists():
  old=load(R/path)
  if old.get('signature')==sig and old.get('implementation_sha256')==sha(__file__):return old['result']
 aa=[];bb=[]
 for r in points:
  with np.load(r['feature']) as z:
   a=z['real'].astype(np.float64);b=z['reconstruction'].astype(np.float64);aa.append(a[1:] if scope=='P_only' else a);bb.append(b[1:] if scope=='P_only' else b)
 result=dict(dataset=d,method=m,q=q,frame_scope=scope,bpp=float(np.mean([r['bpp'] for r in points])),rate_type='real_local_nonofficial_rans',rate_axis='equal-video mean total-stream actual bpp',videos=len(points),aggregation='fixed pooled features, not mean per-video FID',**components(np.concatenate(aa),np.concatenate(bb)))
 save(path,dict(signature=sig,implementation_sha256=sha(__file__),result=result));return result
