"""Locked, occupancy-aware five-mode preflight followed by a fixed 40-task queue."""
from util import *
import concurrent.futures,fcntl,queue,subprocess,time
def available(gpu):
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 row=next(s.split(',') for s in lines if int(s.split(',')[0])==gpu)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return row[1].strip() not in apps and int(row[2])>=12000
def run_queue(specs):
 q=queue.Queue()
 for spec in specs:q.put(spec)
 def worker(gpu):
  from task import valid
  while True:
   try:mode,source,qp=q.get_nowait()
   except queue.Empty:return
   k=key(mode,source,qp);path=P/'server_artifacts/evaluation'/f'{k}.json'
   if valid(path):print('RESUME',k,flush=True);continue
   while not available(gpu):print('WAIT_GPU',gpu,k,flush=True);time.sleep(15)
   env=dict(os.environ,PYTHONPATH=str(P)+os.pathsep+str(ORIGINAL),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
   with (P/'server_artifacts/logs'/f'{k}.log').open('a') as log:
    result=subprocess.run([PYTHON,'-B','-u',str(P/'task.py'),'--mode',mode,'--source',source,'--qp',str(qp),'--gpu',str(gpu)],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
   assert result.returncode==0,(k,result.returncode)
   assert valid(path),k
   print('PASS',k,'GPU',gpu,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(worker,g) for g in (4,5,6,7)]
  for f in fs:f.result()
def cross_checks(source,qp):
 rs={m:load(P/'server_artifacts/evaluation'/f'{key(m,source,qp)}.json') for m in MODES}
 assert len({r['I_payload_sha256'] for r in rs.values()})==1
 assert rs['A']['stream_sha256']==rs['IDENTITY']['stream_sha256']
 assert rs['FRAME']['stream_sha256']==rs['CHANNEL']['stream_sha256']
 for a,b in [('A','IDENTITY'),('FRAME','CHANNEL')]:
  x=load(P/'server_artifacts/states'/f'{key(a,source,qp)}.json');y=load(P/'server_artifacts/states'/f'{key(b,source,qp)}.json')
  assert x['encoder']==y['encoder'] and x['decoder']==y['decoder']
  if a=='FRAME':
   for k in ('PSNR','LPIPS','DISTS','FloLPIPS'):assert rs[a][k]==rs[b][k]
 return dict(sequence=source,QP=qp,I_payload_same=True,A_IDENTITY_stream_states_equal=True,FRAME_CHANNEL_stream_states_metrics_equal=True)
def main():
 lock=(P/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from task import valid
 for p,h in load(P/'dependency_manifest.json').items():assert sha(p)==h
 for p,r in load(P/'weight_manifest.json').items():assert sha(p)==r['sha256']
 dump(P/'run_status.json',dict(status='PREFLIGHT',training_updates=0,signature=signature()))
 run_queue([(m,'UVal0',2) for m in MODES])
 check=cross_checks('UVal0',2)
 dump(P/'preflight.json',dict(status='PASS',signature=signature(),full64_modes=list(MODES),cross_mode_checks=check,all_encoder_decoder_maps_hashes_exact=True,native_A_IDENTITY_parity=True,training_updates=0))
 print('PREFLIGHT_PASS',flush=True)
 dump(P/'run_status.json',dict(status='EVALUATION',signature=signature(),training_updates=0))
 run_queue([(m,s,q) for s in SOURCES for q in (2,7) for m in MODES if (s,q)!=('UVal0',2)])
 checks=[cross_checks(s,q) for s in SOURCES for q in (2,7)]
 dump(P/'cross_mode_audit.json',dict(status='PASS',records=checks,raw_H_output_channels=1,FRAME_CHANNEL_degenerate_by_architecture=True))
 dump(P/'run_status.json',dict(status='COMPUTE_COMPLETE',signature=signature(),tasks=40,training_updates=0))
 print('COMPUTE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BlockingIOError:print('RUNNER_ALREADY_ACTIVE',flush=True)
 except BaseException as e:
  dump(P/'run_status.json',dict(status='FAILED',error=repr(e)));raise
