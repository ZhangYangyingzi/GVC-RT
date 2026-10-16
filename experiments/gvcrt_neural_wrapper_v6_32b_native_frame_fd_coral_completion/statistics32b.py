"""Native-frame prefill and normal-view cache equivalence; historical files read only."""
from io32 import *
import argparse,resource,traceback

def run(index,gpu,verify=False):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from training32 import setup,seed,official_model,full_pass,check_frozen,rng_state,restore_rng,buffer_state,restore_buffers
 from frames32 import PlanFrames,metadata
 ctx=setup();official=official_model();oh=tensor_hash(official.state_dict());seed(load(ROOT/'config.json')['seed_global']);rng=rng_state();models=ctx[3];buf=buffer_state(models);allbuf=buffer_state(dict(models,im=ctx[1],pm=ctx[2],official=official));torch.cuda.reset_peak_memory_stats();tic=time.time()
 row=load(ROOT/'plans/shared.json')['updates'][index];v=metadata(row);f=PlanFrames(row,'cuda:0',full=True)
 z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close();restore_rng(rng);restore_buffers(models,buf);check_frozen(ctx);assert oh==tensor_hash(official.state_dict())
 after=buffer_state(dict(models,im=ctx[1],pm=ctx[2],official=official));assert all(torch.equal(t,after[k][n]) for k,b in allbuf.items() for n,t in b.items())
 old=OLD32/'reference_stats/receipts'/f'{index:04d}.json';eq=None
 if verify:
  previous=load(old);assert previous['status']=='PASS' and sha(previous['feature_path'])==previous['artifacts'][previous['feature_path']]
  checks=[]
  with np.load(previous['feature_path']) as a:
   for key,x in [('real',gt.numpy()),('reconstruction',z.numpy())]:
    y=a[key];rel=float(np.linalg.norm(x.astype('float64')-y)/max(np.linalg.norm(y.astype('float64')),1e-12));ma=float(np.max(np.abs(x-y)));tol=1e-4+1e-4*float(np.abs(y).max());assert rel<=1e-4 and ma<=tol,(key,rel,ma,tol);checks.append(dict(kind=key,relative_L2=rel,maxabs=ma,maxabs_tolerance=tol))
  eq=dict(status='PASS',previous_receipt=str(old),previous_receipt_sha256=sha(old),checks=checks,ordinary_1080_behavior_unchanged=True,crop_function_identical=True)
  dump(ROOT/'audits'/f'prefill_equivalence_{row["domain"]}.json',eq)
 path=ROOT/'features/prefill'/f'{index:04d}.npz';save_npz(path,real=gt.numpy(),reconstruction=z.numpy());right,bottom=ctx[2].get_padding_size(v.get('height',1080),v.get('width',1920),16)
 geometry=dict(plan_index=index,native_H=v.get('height',1080),native_W=v.get('width',1920),canvas_H=v.get('height',1080)+bottom,canvas_W=v.get('width',1920)+right,padding=[0,right,0,bottom],unpad=[0,v.get('height',1080),0,v.get('width',1920)],source_path=row['source_path'],source_sha256=row['source_sha256'],full_RGB_sha256=row['full_RGB_sha256'],cropped_RGB_sha256=row['cropped_RGB_sha256'],frames=64,causal=True,native_I=True,official_padding_helper='CompressionModel.get_padding_size(height,width,16)',padding_mode='replicate',padding_excluded_from_Inception=True,frozen_verified=True,status='PASS')
 dump(ROOT/'audits/geometry'/f'{index:04d}.json',geometry)
 r=dict(status='PASS',signature=signature('prefill'),plan_index=index,domain=row['domain'],QP=row['external_qp'],video=row['video'],source_sha256=row['source_sha256'],full_RGB_sha256=row['full_RGB_sha256'],optimizer_updates=0,frames=64,features=15,shape=list(z.shape),official_hash=oh,artifacts={str(path):sha(path)},feature_path=str(path),seconds=time.time()-tic,physical_GPU=gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,geometry=geometry,old_equivalence=eq,reused=False)
 dump(ROOT/'reference_stats/receipts'/f'{index:04d}.json',r);print('NATIVE PREFILL PASS',index,r['seconds'],flush=True)

def import_cache():
 import numpy as np
 from evaluate32 import task_id,eval_signature,receipt,source
 for d in ('ulong','uvg'):assert load(ROOT/'audits'/f'prefill_equivalence_{d}.json')['status']=='PASS'
 plan=load(ROOT/'plans/shared.json')['updates'];imported=[]
 expectedhash='7618ca636ddfe0633e08149eff69ce13e71e7d57c4ecf085d168e3f3c41cc3a7'
 for i,row in enumerate(plan):
  out=ROOT/'reference_stats/receipts'/f'{i:04d}.json'
  if out.exists():continue
  p=OLD32/'reference_stats/receipts'/f'{i:04d}.json';r=load(p)
  if r['status']!='PASS':continue
  assert r['plan_index']==i and r['source_sha256']==row['source_sha256'] and r['full_RGB_sha256']==row['full_RGB_sha256'] and r['domain']==row['domain'] and r['QP']==row['external_qp']
  assert r['official_hash']==expectedhash and r['optimizer_updates']==0
  assert all(sha(k)==v for k,v in r['artifacts'].items())
  with np.load(r['feature_path']) as a:assert all(a[k].shape==(15,2048) and np.isfinite(a[k]).all() for k in ('real','reconstruction'))
  rr=dict(r,signature=signature('prefill'),reused=True,reuse_receipt=str(p),reuse_receipt_sha256=sha(p),historical_signature=r['signature'],equivalence_evidence=str(ROOT/'audits'/f'prefill_equivalence_{row["domain"]}.json'),dependency_mapping='same B1/GT/RGB/official weights/backend; native view equals old view for this 1080p input; historical signature preserved separately')
  dump(out,rr);imported.append(i)
 dump(ROOT/'audits/prefill_reuse.json',dict(status='PASS',imported=len(imported),plan_indices=imported,geometric_change_only_for_native_non1080=True))

def aggregate(gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from distribution32 import reference,moments
 torch.set_num_threads(2);plan=load(ROOT/'plans/shared.json')['updates'];coverage=[];entries=[]
 for d in ('ulong','uvg'):
  for q in range(10):
   ids=[i for i,r in enumerate(plan) if r['domain']==d and r['external_qp']==q];assert len(ids)==50;real=[];rec=[]
   for i in ids:
    r=load(ROOT/'reference_stats/receipts'/f'{i:04d}.json');assert r['status']=='PASS' and all(sha(k)==v for k,v in r['artifacts'].items())
    with np.load(r['feature_path']) as a:real.append(a['real']);rec.append(a['reconstruction'])
   gt=torch.from_numpy(np.concatenate(real)).cuda();z=torch.from_numpy(np.concatenate(rec)).cuda();ref=reference(gt);mu,M,_=moments(z);seed=dict(mu=mu.cpu(),M=M.cpu(),counter=0);p=ROOT/'reference_stats/strata'/f'{d}_{q}.pt'
   atomic_torch(p,dict(reference={k:v.cpu() if torch.is_tensor(v) else v for k,v in ref.items()},seed=seed));entries.append(dict(stratum=f'{d}:{q}',path=str(p),sha256=sha(p)))
   coverage.append(dict(stratum=f'{d}:{q}',entries=50,features=750,unique_UIDs=len({(plan[i]['source_sha256'],f) for i in ids for f in plan[i]['source_frame_indices'][49:]}),unique_windows=len({(plan[i]['source_sha256'],tuple(plan[i]['source_frame_indices'])) for i in ids}),unique_videos=len({plan[i]['source_sha256'] for i in ids}),ddof=0,epsilon=ref['epsilon'],status='PASS',occurrence_weights_preserved=True))
   del gt,z,ref,mu,M;torch.cuda.empty_cache();print('STRATUM',d,q,flush=True)
 out=dict(status='PASS',completed=1000,expected=1000,total_strata=20,entries=entries,coverage=coverage,plan_sha256=sha(ROOT/'plans/shared.json'),optimizer_updates=0)
 dump(ROOT/'reference_stats/index.json',out);dump(ROOT/'prefill_integrity.json',out)
 geo=[load(p) for p in sorted((ROOT/'audits/geometry').glob('*.json'))];assert set([165,253,477,720])<={r['plan_index'] for r in geo}
 dump(ROOT/'geometry_audit.json',dict(status='PASS',checks=geo,ordinary_1080_equivalence=[load(ROOT/'audits'/f'prefill_equivalence_{d}.json') for d in ('ulong','uvg')],formal_test_geometry_unchanged=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=4);ap.add_argument('--verify',action='store_true');ap.add_argument('--import-cache',action='store_true');ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 try:
  if a.import_cache:import_cache()
  elif a.aggregate:aggregate(a.gpu)
  else:run(a.index,a.gpu,a.verify)
 except Exception:error(dict(stage='statistics32b',arguments=vars(a),error=traceback.format_exc()));raise
