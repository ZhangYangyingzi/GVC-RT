"""Compute complete pooled rate points on CPU while independent GPU tasks run."""
from io21 import *
import math,statistics,fcntl
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def main():
 import numpy as np
 lock=(ROOT/'pooled_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);vs=sources('mcljcv');fid=module('precompute_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').low_rank_fid;gtpath=ROOT/'features/GT_mcljcv.npy';gtpath.parent.mkdir(exist_ok=True)
 while not gtpath.exists():
  for m in METHODS:
   ps=[point('mcljcv',m,v['video_index'],0) for v in vs]
   if all(p.exists() for p in ps):
    arrays=[];src=[]
    for p in ps:
     r=load(p);assert sha(r['feature_path'])==r['feature_sha256']
     with np.load(r['feature_path']) as z:arrays.append(z['real'].copy())
     src.append(dict(point=str(p),feature_path=r['feature_path'],sha256=r['feature_sha256']))
    reference=np.concatenate(arrays);assert reference.shape==(len(vs)*64,2048);tmp=gtpath.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,reference)
    tmp.replace(gtpath);dump(ROOT/'audits/GT_feature_pool.json',dict(status='PASS',method_used_for_GT_extraction=m,GT_depends_only_on_source_RGB_and_frozen_Inception=True,source_features=src,shape=list(reference.shape),GT_sha256=sha(gtpath)));break
  if not gtpath.exists():time.sleep(20)
 reference=np.load(gtpath);pool=sha(gtpath);first=reference.astype(np.float64);a=first-first.mean(0);completed=set()
 while len(completed)<50:
  for m in METHODS:
   for q in range(10):
    key=f'{m}_qp{q}';cp=ROOT/'results/mcljcv/pooled_points'/f'{key}.json'
    if key in completed:continue
    if cp.exists():assert load(cp)['GT_features_sha256']==pool;completed.add(key);continue
    ps=[point('mcljcv',m,v['video_index'],q) for v in vs]
    if not all(p.exists() for p in ps):continue
    rs=[load(p) for p in ps];recs=[];paths=[]
    for i,r in enumerate(rs):
     assert sha(r['feature_path'])==r['feature_sha256']
     with np.load(r['feature_path']) as z:assert np.allclose(z['real'],reference[i*64:(i+1)*64],atol=1e-6,rtol=1e-5);recs.append(z['reconstruction'].copy())
     paths.append(dict(path=r['feature_path'],sha256=r['feature_sha256']))
    second=np.concatenate(recs).astype(np.float64);b=second-second.mean(0);mu=float(np.square(first.mean(0)-second.mean(0)).sum());cov=float(np.square(a).sum()/(len(a)-1)+np.square(b).sum()/(len(b)-1)-2*np.linalg.svd((a@b.T)/(len(a)-1),compute_uv=False).sum());value=float(fid(first,second));assert abs(value-mu-cov)<1e-7+1e-8*abs(value)
    r=dict(dataset='mcljcv',method=m,QP=q,bpp=statistics.mean(r['bpp'] for r in rs),FID=value,D_mu=mu,D_cov=cov,FID_closure_error=abs(value-mu-cov),samples=len(reference),sequences=len(vs),GT_features_sha256=pool,feature_sources=paths,checkpoint_SHA256=rs[0]['checkpoint_sha256'],aggregation='fixed dataset pooled features; sample covariance ddof1; float64',**{k:statistics.mean(r[k] for r in rs) for k in ['LPIPS','DISTS','FloLPIPS','kbps','kbps_20fps','PSNR','SSIM']});dump(cp,r);completed.add(key);print('POOLED READY',key,flush=True);dump(ROOT/'pooled_status.json',dict(status='RUNNING',completed=len(completed),expected=50))
  time.sleep(15)
 dump(ROOT/'pooled_status.json',dict(status='PASS',completed=50,expected=50));print('POOLED PRECOMPUTE PASS',flush=True)
if __name__=='__main__':main()
