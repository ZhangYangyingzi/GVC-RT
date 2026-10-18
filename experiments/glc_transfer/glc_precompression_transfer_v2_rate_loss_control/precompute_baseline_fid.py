"""CPU-only fixed-pool baseline FID cache; final report revalidates every key."""
from preflight import *
from data import load
from fid_pool import point
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl

def main():
 lock=(R/'baseline_fid.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=load(R/'eval_manifest.json');qs=load(R/'protocol.json')['q_grid'];jobs=[]
 for d,vs in manifest.items():
  for m in ('A_original','v1_C_latent_P'):
   for q in qs:
    rr=[]
    for i in range(len(vs)):
     receipt=load(R/f'results/points/{d}_{i}_{m}_{q}.json')
     assert receipt['status']=='PASS' and receipt['reuse']['manifest_identical']
     assert all(Path(p).exists() and sha(p)==h for p,h in receipt['artifacts'].items())
     rr.append(receipt['result'])
    for scope in ('all_frame','P_only'):jobs.append((d,m,q,scope,rr))
    if d=='mcljcv':
     subset=[r for r in rr if 'videoSRC17' not in r['name']];assert len(subset)==29
     jobs.append(('mcljcv29_sensitivity',m,q,'all_frame',subset))
 completed=0;started=time.time()
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(point,j) for j in jobs]
  for future in as_completed(futures):
   result=future.result();completed+=1
   save('audits/baseline_fid_precompute.json',dict(status='PASS' if completed==len(jobs) else 'RUNNING',completed=completed,expected=len(jobs),CPU_workers=4,GPU_workers=0,implementation_sha256=sha(R/'fid_pool.py'),additional_encodes=0,seconds=time.time()-started,last_result={k:result[k] for k in ('dataset','method','q','frame_scope')}))
   print('BASELINE FID',completed,'/',len(jobs),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save('audits/baseline_fid_precompute_error.json',dict(status='FAIL',error=traceback.format_exc(),final_report_must_revalidate=True));raise
