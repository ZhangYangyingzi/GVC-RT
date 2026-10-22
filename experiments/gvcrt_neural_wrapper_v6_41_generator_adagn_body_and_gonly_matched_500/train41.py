"""Matched full-loss G-only continuation from the immutable original step zero.

This module owns every output. Historical modules are copied locally for the
mathematical paths; reference data are opened read-only through hashed indices.
"""
from io32 import *
import argparse, fcntl, math, resource, traceback

SOURCE = ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
SOURCE_SHA = '18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074'
BRANCH = BRANCHES[0]

def main(gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
 import torch
 from training32 import setup, official_model, rng_state, restore_rng, buffer_state, restore_buffers, crop_c1, full_pass, groups, norm, dot, check_frozen
 from calibration32b import get_stratum, feature_derivatives
 from distribution32 import to, commit
 from frames32 import PlanFrames
 b=ROOT/'branches'/BRANCH;b.mkdir(parents=True,exist_ok=True)
 lock=(b/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(SOURCE)==SOURCE_SHA
 assert sha(ROOT/'plans/shared.json')=='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42'
 cfg=load(ROOT/'effective_config.json');ctx=setup(BRANCH)
 v4,im,pm,models,quality,common,params,frozen=ctx
 assert not list(models['wrapper'].parameters()) and all(not p.requires_grad for p in models['bridge'].parameters())
 assert len(params)==len(list(models['generator'].parameters())) and all(p.requires_grad for p in params)
 official=official_model();allmods=dict(models,im=im,pm=pm,official=official,lp=quality[0],dists=quality[1])
 start_source=torch.load(SOURCE,map_location='cpu',weights_only=True)
 assert start_source['adaptation_step']==0 and start_source['plan_cursor']==0
 assert start_source['plan_sha256']==sha(ROOT/'plans/shared.json')
 assert start_source['seed_index_sha256']==sha(ROOT/'reference_stats/index.json')
 for k,m in models.items():m.load_state_dict(start_source[k],strict=True)
 restore_buffers(allmods,start_source['buffers']);restore_rng(start_source['crop_rng'])
 assert v4.module_hash(models['bridge'])=='9c4ca3f7a3e2f502688509ca82555f0ce7ee6fe8c31a89fa9f8dff1104953cd3'
 assert v4.module_hash(models['generator'])=='318ea46772ebd1ae756b71edb430552e2e704c7fc248f8b140b5f28e77d11aab'
 bridge_initial=tensor_hash(models['bridge'].state_dict())
 assert frozen['compression']==start_source['frozen_hashes']['compression'] and frozen['I']==start_source['frozen_hashes']['I']
 opt=torch.optim.AdamW([dict(name='generator',params=params,lr=1e-6)],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 ema=start_source['EMA'];global_rng=start_source['global_rng'];step0_crop=start_source['crop_rng']
 signature_value=hashlib.sha256(json.dumps(dict(source=SOURCE_SHA,plan=sha(ROOT/'plans/shared.json'),
  config=sha(ROOT/'effective_config.json'),code={n:sha(ROOT/n) for n in ('train41.py','training32.py','distribution32.py','frames32.py','calibration32b.py','io32.py')}),sort_keys=True).encode()).hexdigest()
 latest=b/'checkpoints/latest.pt';start=0
 if latest.exists():
  state=torch.load(latest,map_location='cpu',weights_only=True)
  assert state['signature']==signature_value and state['source_checkpoint_sha256']==SOURCE_SHA and state['branch']==BRANCH
  start=state['adaptation_step'];assert state['plan_cursor']==start
  for k,m in models.items():m.load_state_dict(state[k],strict=True)
  opt.load_state_dict(state['optimizer']);ema=state['EMA'];global_rng=state['global_rng']
  restore_buffers(allmods,state['buffers']);restore_rng(state['crop_rng'])
  assert sum(x['counter'] for x in ema.values())==start
  if start:assert {int(x['step']) for x in opt.state.values()}=={start}
  dump(b/'restore_receipt.json',dict(status='PASS',from_path=str(latest),sha256=sha(latest),step=start,signature=signature_value))
  del state
 assert tensor_hash(models['bridge'].state_dict())==bridge_initial
 plan=load(ROOT/'plans/shared.json')['updates'];assert len(plan)>=500
 def save(step,receipt=None):
  assert tensor_hash(models['bridge'].state_dict())==bridge_initial
  check_frozen(ctx)
  assert sum(x['counter'] for x in ema.values())==step
  state=dict(schema='v641_gonly',branch=BRANCH,adaptation_step=step,plan_cursor=step,
   committed_optimizer_updates=step,source_checkpoint_sha256=SOURCE_SHA,signature=signature_value,
   plan_sha256=sha(ROOT/'plans/shared.json'),seed_index_sha256=sha(ROOT/'reference_stats/index.json'),
   initial_module_hashes=start_source['initial_module_hashes'],frozen_hashes=frozen,
   wrapper=models['wrapper'].state_dict(),bridge=models['bridge'].state_dict(),generator=models['generator'].state_dict(),
   optimizer=opt.state_dict(),crop_rng=rng_state(),global_rng=global_rng,buffers=buffer_state(allmods),EMA=ema,last_receipt=receipt)
  atomic_torch(latest,state)
  if step in (0,100,250,500):
   cp=b/'checkpoints'/f'adaptation_step_{step}.pt';atomic_torch(cp,state)
   inf=ROOT/'evaluation/checkpoints'/f'{BRANCH}_{step}.pt';atomic_torch(inf,dict(bridge=models['bridge'].state_dict(),generator=models['generator'].state_dict()))
   idx=load(b/'checkpoint_index.json') if (b/'checkpoint_index.json').exists() else {}
   idx[str(step)]=dict(path=str(cp),sha256=sha(cp),inference_path=str(inf),inference_sha256=sha(inf),
    bridge_hash=v4.module_hash(models['bridge']),generator_hash=v4.module_hash(models['generator']),
    optimizer_step=step,EMA_commits=sum(x['counter'] for x in ema.values()))
   dump(b/'checkpoint_index.json',idx)
 if start==0 and not (b/'checkpoint_index.json').exists():save(0)
 torch.cuda.reset_peak_memory_stats()
 for step in range(start+1,501):
  t=time.time();item=plan[step-1];row=source_row(item)
  assert item['step']==step and hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['source_row_canonical_sha256']
  opt.zero_grad(set_to_none=True);before_scope=buffer_state(allmods)
  f=PlanFrames(row,'cuda:0');g0,values,trace=crop_c1(ctx,f,row['external_qp']);f.close()
  crng=rng_state();cbuf=buffer_state(allmods)
  restore_rng(global_rng);restore_buffers(allmods,before_scope)
  f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close()
  gnext=rng_state();key=f'{row["domain"]}:{row["external_qp"]}'
  old=to(ema[key],'cuda:0');_,ref=get_stratum(key,'cuda:0')
  fg,ls,candidate=feature_derivatives(z,old,ref,('FD',))
  restore_rng(global_rng);restore_buffers(allmods,before_scope)
  f=PlanFrames(row,'cuda:0',full=True);_,_,unit,replay=full_pass(ctx,f,row['external_qp'],official,fg['FD'],expected=z);f.close()
  restore_rng(crng);restore_buffers(allmods,cbuf);global_rng=gnext
  extra=[g*cfg['lambda_FD'] for g in unit];total=[a+e for a,e in zip(g0,extra)]
  assert groups(models,g0)['generator']['norm']>0 and groups(models,unit)['generator']['norm']>0
  assert all(torch.isfinite(g).all() for g in total) and math.isfinite(ls['FD'])
  assert [x['actual_qp'] for x in trace]==row['actual_qps']
  pre=norm(total);formula=norm([g-a-e for g,a,e in zip(total,g0,extra)])/max(pre,1e-30);assert formula<=1e-6
  before=[p.detach().clone() for p in params]
  for p,g in zip(params,total):p.grad=g
  n=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(n) and n>0
  clip=min(1.,1./(n+1e-6));post=norm([p.grad for p in params]);assert post>0
  append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,time=time.time(),pid=os.getpid()))
  opt.step();append(b/'optimizer_calls.jsonl',dict(event='APPLIED',step=step,time=time.time(),pid=os.getpid()))
  delta=norm([p.detach()-a for p,a in zip(params,before)]);assert delta>0 and math.isfinite(delta)
  ema[key]=to(commit(old,candidate),'cpu');assert sum(x['counter'] for x in ema.values())==step
  assert tensor_hash(models['bridge'].state_dict())==bridge_initial
  r=dict(step=step,source_row_sha256=item['source_row_canonical_sha256'],domain=row['domain'],QP=row['external_qp'],
   losses=values,FD=ls['FD'],weighted_FD=cfg['lambda_FD']*ls['FD'],C1_norm=norm(g0),FD_unit_norm=norm(unit),
   FD_weighted_norm=norm(extra),gradient_formula_error=formula,preclip_norm=pre,postclip_norm=post,
   clip_coefficient=clip,generator_delta_norm=delta,bridge_delta_norm=0,EMA_stratum=key,
   EMA_stratum_counter=ema[key]['counter'],EMA_total_commits=step,replay=replay,
   optimizer_internal_step=step,seconds=time.time()-t,physical_gpu=gpu,
   peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  save(step,r);dump(b/'receipts'/f'{step}.json',r)
  append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,time=time.time(),pid=os.getpid()))
  dump(b/'training_status.json',dict(status='PASS' if step==500 else 'RUNNING',committed_updates=step,
   EMA_commits=step,peak_memory_MiB=r['peak_memory_MiB'],seconds_per_update=r['seconds'],updated_unix=time.time()))
  if step==1 or step%25==0:print('TRAIN_GONLY',step,'/500',round(r['seconds'],2),flush=True)
 assert sum(x['counter'] for x in ema.values())==500
 assert tensor_hash(models['bridge'].state_dict())==bridge_initial
 dump(b/'training_integrity.json',dict(status='PASS',committed_updates=500,EMA_commits=500,
  bridge_initial_hash=bridge_initial,bridge_final_hash=tensor_hash(models['bridge'].state_dict()),
  generator_final_hash=v4.module_hash(models['generator']),optimizer_internal_step=500,
  source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
 try:main(a.gpu)
 except Exception:
  error(dict(stage='G_only_training',traceback=traceback.format_exc(),time=time.time()))
  raise
