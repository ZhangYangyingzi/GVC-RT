"""Shared B1 prefill, strictly excluding unavailable full-resolution tasks."""
from io32 import *
import argparse,resource,traceback,fcntl
def main(gpu,shard,shards):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from training32 import setup,seed,official_model,full_pass,check_frozen,rng_state,restore_rng,buffer_state,restore_buffers
 from frames32 import PlanFrames,full_eligible
 ctx=setup();official=official_model();oh=tensor_hash(official.state_dict());seed(load(ROOT/'config.json')['seed_global']);rng=rng_state();buf=buffer_state(ctx[3]);sig=signature('prefill')
 torch.cuda.reset_peak_memory_stats();start=time.time();n=0
 for i,row in enumerate(load(ROOT/'plans/shared.json')['updates']):
  if i%shards!=shard:continue
  p=ROOT/'reference_stats/receipts'/f'{i:04d}.json'
  if p.exists():
   r=load(p)
   if r['signature']==sig and (r['status']=='BLOCKED' or all(sha(k)==v for k,v in r['artifacts'].items())):continue
  if not full_eligible(row):
   dump(p,dict(status='BLOCKED',signature=sig,plan_index=i,reason='FULL_RESOLUTION_MISMATCH',video=row['video'],source_sha256=row['source_sha256'],artifacts={}));continue
  tic=time.time();restore_rng(rng);restore_buffers(ctx[3],buf);frames=PlanFrames(row,'cuda:0',full=True)
  z,gt,_,_=full_pass(ctx,frames,row['external_qp'],official);frames.close();restore_buffers(ctx[3],buf);restore_rng(rng)
  f=ROOT/'features/prefill'/f'{i:04d}.npz';save_npz(f,real=gt.numpy(),reconstruction=z.numpy())
  dump(p,dict(status='PASS',signature=sig,plan_index=i,domain=row['domain'],QP=row['external_qp'],video=row['video'],source_sha256=row['source_sha256'],full_RGB_sha256=row['full_RGB_sha256'],optimizer_updates=0,frames=64,features=15,shape=list(z.shape),official_hash=oh,artifacts={str(f):sha(f)},feature_path=str(f),seconds=time.time()-tic,physical_GPU=gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
  n+=1
  dump(ROOT/'reference_stats'/f'worker_{shard}.json',dict(status='RUNNING',PID=os.getpid(),gpu=gpu,last_index=i,completed_this_launch=n,seconds=time.time()-start,updated_unix=time.time(),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
  if n==1 or n%25==0:print('PREFILL',shard,i,flush=True)
 check_frozen(ctx);assert oh==tensor_hash(official.state_dict())
 dump(ROOT/'reference_stats'/f'worker_{shard}.json',dict(status='PASS',PID=os.getpid(),gpu=gpu,completed_this_launch=n,updated_unix=time.time(),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
def aggregate():
 import torch,numpy as np
 from distribution32 import moments,reference
 rows=load(ROOT/'plans/shared.json')['updates'];allr=[];refs={};seeds={};coverage=[];sig=signature('prefill')
 for i,row in enumerate(rows):
  p=ROOT/'reference_stats/receipts'/f'{i:04d}.json';r=load(p);assert r['signature']==sig;allr.append(r)
 for d in ('ulong','uvg'):
  for q in range(10):
   ids=[i for i,r in enumerate(rows) if r['domain']==d and r['external_qp']==q];key=f'{d}:{q}';missing=[i for i in ids if allr[i]['status']!='PASS'];pool=[];rp=[]
   if not missing:
    for i in ids:
     r=allr[i];assert sha(r['feature_path'])==r['artifacts'][r['feature_path']]
     with np.load(r['feature_path']) as a:pool.append(a['real']);rp.append(a['reconstruction'])
    z=torch.from_numpy(np.concatenate(pool));g=torch.from_numpy(np.concatenate(rp));refs[key]=reference(z);mu,M,S=moments(g);seeds[key]=dict(mu=mu,M=M,counter=0)
   coverage.append(dict(stratum=key,expected_plan_entries=len(ids),completed=sum(allr[i]['status']=='PASS' for i in ids),expected_features=15*len(ids),unique_target_UIDs=len({(rows[i]['source_sha256'],k) for i in ids for k in rows[i]['source_frame_indices'][49:]}),unique_windows=len({(rows[i]['source_sha256'],tuple(rows[i]['source_frame_indices'])) for i in ids}),unique_videos=len({rows[i]['source_sha256'] for i in ids}),status='BLOCKED' if missing else 'PASS',missing_plan_indices=missing,covariance_rank_upper_bound=min(2048,15*len(ids)-1),correlated_frames=True))
 p=ROOT/'reference_stats/partial_seeds.pt';atomic_torch(p,dict(references=refs,seeds=seeds,signature=sig))
 ok=sum(r['status']=='PASS' for r in allr)
 dump(ROOT/'reference_stats/index.json',dict(status='PASS' if ok==1000 else 'BLOCKED',path=str(p),sha256=sha(p),coverage=coverage,completed=ok,expected=1000,optimizer_updates=0,complete_reference_strata=len(refs),total_strata=20,missing=[r['plan_index'] for r in allr if r['status']!='PASS']))
 dump(ROOT/'prefill_manifest.json',dict(plan_sha256=sha(ROOT/'plans/shared.json'),entries=allr))
 dump(ROOT/'prefill_integrity.json',load(ROOT/'reference_stats/index.json'))
 if ok<1000:dump(ROOT/'calibration/final.json',dict(status='BLOCKED',completed_windows=0,expected_windows=40,reason='Incomplete required full-frame GT/seed pool; no partial-stratum substitution',missing_plan_indices=[r['plan_index'] for r in allr if r['status']!='PASS']))
 print('PREFILL AGGREGATE',ok,'/1000',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7));ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=2);ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 try:aggregate() if a.aggregate else main(a.gpu,a.shard,a.shards)
 except Exception:error(dict(stage='prefill',shard=a.shard,error=traceback.format_exc()));raise
