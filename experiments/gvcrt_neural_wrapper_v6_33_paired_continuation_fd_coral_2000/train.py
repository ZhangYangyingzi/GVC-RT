"""Exactly1000 committed continuation updates; preserved AdamW/RNG/EMA lineage."""
from io32 import *
import argparse,fcntl,resource,math,traceback

def main(branch,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from training32 import setup,crop_c1,full_pass,official_model,rng_state,restore_rng,buffer_state,restore_buffers,norm,groups,check_frozen,dot
 from frames32 import PlanFrames
 from calibration32b import get_stratum,feature_derivatives
 from distribution32 import to,commit
 from migration33 import load_state,state_digest
 cfg=load(ROOT/'config.json');b=ROOT/'branches'/branch;b.mkdir(exist_ok=True);lf=(b/'train.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(ROOT/'audits/data_isolation.json')['status']=='PASS'
 ctx=setup();v4,im,pm,models,quality,common,params,frozen=ctx;official=official_model();oh=tensor_hash(official.state_dict());cal=load(ROOT/'calibration/final.json');calsha=sha(ROOT/'calibration/final.json');seedsha=sha(ROOT/'reference_stats/index.json')
 # Construct the exact optimizer class/groups, then restore all saved moments/steps.
 opt=torch.optim.AdamW([dict(name=k,params=list(models[k].parameters()),lr=lr) for k,lr in zip(('bridge','generator'),(6e-6,1e-6))],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 s,lineage=load_state(ctx,opt,branch);start=s['adaptation_step'];global_rng=s['global_rng'];visits=s['visits'];ema=s.get('EMA');observed=s.get('observed',{});parent=cfg['branch_sources'][branch];sig=signature('train');plan=load(ROOT/'plans/continuation.json')['updates'];idx=load(b/'checkpoint_index.json') if (b/'checkpoint_index.json').exists() else {}
 def save(step,row=None):
  check_frozen(ctx);assert oh==tensor_hash(official.state_dict());assert {int(x['step']) for x in opt.state.values()}=={step}
  if ema is not None:assert sum(v['counter'] for v in ema.values())==step and all(ema[k]['counter']==visits[k] for k in ema)
  state=dict(schema='v633',branch=branch,adaptation_step=step,additional_step=step-1000,committed_optimizer_updates=step,committed_additional_updates=step-1000,replay_cursor=step-1000,signature=sig,source_signature_mapping=lineage,config_sha256=sha(ROOT/'config.json'),protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=sha(ROOT/'plans/shared.json'),continuation_sha256=sha(ROOT/'plans/continuation.json'),parent_training_checkpoint_sha256=parent['sha256'],calibration_sha256=calsha,seed_index_sha256=seedsha,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),crop_rng=rng_state(),global_rng=global_rng,buffers=buffer_state(models),visits=visits,observed=observed,EMA=ema,EMA_status='N/A' if ema is None else 'RESTORED_AND_CONTINUED',frozen_hashes=frozen,last_receipt=row)
  atomic_torch(b/'checkpoints/latest.pt',state)
  if step in (1000,1001,1250,1500,1750,2000):
   p=b/'checkpoints'/f'adaptation_step_{step}.pt';atomic_torch(p,state);e=ROOT/'evaluation/checkpoints'/f'{branch}_{step}.pt';atomic_torch(e,{k:m.state_dict() for k,m in models.items()});hashes={k:v4.module_hash(m) for k,m in models.items()};deploy={k:tensor_hash({n:t.half() for n,t in models[k].state_dict().items()}) for k in ('bridge','generator')}
   idx[str(step)]=dict(path=str(p),sha256=sha(p),adaptation_step=step,additional_step=step-1000,parent_training_checkpoint=parent,model_hashes=hashes,module_hashes=hashes,compression_hash=frozen['compression'],deployment_receiver_hashes=deploy,inference=dict(path=str(e),sha256=sha(e),module_hashes=hashes),optimizer_internal_step=step,EMA_counters=None if ema is None else {k:v['counter'] for k,v in ema.items()});dump(b/'checkpoint_index.json',idx)
 if start==1000:
  if '1000' not in idx:save(1000)
  init=load(b/'checkpoint_index.json')['1000'];assert init['module_hashes']==parent['model_hashes'];assert state_digest(opt.state_dict())==lineage['optimizer_state_sha256']
  dump(b/'migration_equivalence.json',dict(status='PASS',optimizer_state_sha256=state_digest(opt.state_dict()),model_hashes=init['module_hashes'],RNG_preserved=True,buffers_preserved=True,EMA_preserved=ema is not None,EMA_status='N/A' if ema is None else '50 per stratum',starting_snapshot_sha256=init['sha256'],parent_sha256=parent['sha256'],zero_optimizer_updates=True))
 elif s['last_receipt']:dump(b/'receipts'/f'{start}.json',s['last_receipt'])
 del s
 torch.cuda.reset_peak_memory_stats();tic=time.time()
 for step in range(start+1,2001):
  item=plan[step-1001];row=item['source_row'];assert item['adaptation_step']==step and row['adaptation_step']==step-1000
  raw=json.dumps(row,sort_keys=True,separators=(',',':')).encode();assert hashlib.sha256(raw).hexdigest()==item['source_row_canonical_sha256'];t=time.time();buf=buffer_state(models);opt.zero_grad(set_to_none=True)
  f=PlanFrames(row,'cuda:0');base,sums,trace=crop_c1(ctx,f,row['external_qp']);f.close();after_rng=rng_state();after_buf=buffer_state(models);key=f'{row["domain"]}:{row["external_qp"]}'
  restore_rng(global_rng);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close();global_after=rng_state();extra=[torch.zeros_like(g) for g in base];replay=[];values={};next_state=None
  if ema is not None:
   _,ref=get_stratum(key,'cuda:0');old=to(ema[key],'cuda:0');objectives=('FD',) if branch==BRANCHES[1] else ('mean','CORAL');weights=dict(FD=cal['lambda_FD'],mean=cal['alpha_mu'],CORAL=cal['alpha_cov']);fg,values,next_state=feature_derivatives(z,old,ref,objectives);combined=sum(weights[k]*fg[k] for k in objectives)
   restore_rng(global_rng);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);_,_,extra,replay=full_pass(ctx,f,row['external_qp'],official,fg if step==1001 else combined,expected=z);f.close()
   unit_contributions={k:groups(models,g) for k,g in extra.items()} if isinstance(extra,dict) else None
   if isinstance(extra,dict):extra=[sum(weights[k]*extra[k][j] for k in objectives) for j in range(len(params))]
  else:weights={};objectives=();unit_contributions=None
  global_rng=global_after;restore_rng(after_rng);restore_buffers(models,after_buf);gr=[a+e for a,e in zip(base,extra)];grad=groups(models,gr);eg=groups(models,extra)
  assert all(v['finite'] and v['norm']>0 for v in grad.values());assert ema is None or all(v['finite'] and v['norm']>0 for v in eg.values());assert [x['actual_qp'] for x in trace]==row['actual_qps']
  pre=norm(gr);merge_error=norm([g-a-e for g,a,e in zip(gr,base,extra)])/max(pre,1e-30);assert merge_error<=1e-6
  # Record composition before the p.grad aliases are modified by clipping.
  for p,g in zip(params,gr):p.grad=g
  before=[p.detach().clone() for p in params];rep=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(rep) and rep>0;clip=min(1.,1./(rep+1e-6));post=norm([p.grad for p in params]);clip_error=norm([p.grad-(a+e)*clip for p,a,e in zip(params,base,extra)])/max(post,1e-30);assert clip_error<=1e-6
  append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,additional_step=step-1000,time=time.time(),PID=os.getpid(),committed_before=step-1));opt.step();assert all(p.isfinite().all() for p in params);delta=groups(models,[p.detach()-v for p,v in zip(params,before)]);assert all(v['norm']>0 for v in delta.values())
  visits[key]+=1
  if ema is not None:ema[key]=to(commit(old,next_state),'cpu');assert ema[key]['counter']==visits[key]
  else:observed[key]=observed.get(key,0)+1
  fp=ROOT/'features/training'/branch/f'{step}.npz';save_npz(fp,reconstruction=z.numpy(),real=gt.numpy());additional_loss=sum(weights[k]*values[k] for k in objectives)
  receipt=dict(row,source_plan_index=item['source_plan_index'],source_plan_step=item['source_plan_step'],source_row_canonical_sha256=item['source_row_canonical_sha256'],branch=branch,adaptation_step=step,additional_step=step-1000,committed_optimizer_updates=step,committed_additional_updates=step-1000,**sums,distribution_losses=values,distribution_weights={k:weights[k] for k in objectives},additional_loss=additional_loss,complete_objective_loss=sums['total_loss']+additional_loss,actual_qps=[x['actual_qp'] for x in trace],frame_trace=trace,gradient_modules=grad,additional_gradient_modules=eg,first_step_unit_objective_gradients=unit_contributions,additional_gradient_norm=norm(extra),C1_full_gradient_norm=norm(base),additional_to_C1_ratio=norm(extra)/norm(base),additional_C1_cosine=None if ema is None else dot(base,extra)/(norm(base)*norm(extra)),parameter_delta=delta,global_preclip_norm=pre,global_postclip_norm=post,clip_coefficient=clip,gradient_formula_error=merge_error,gradient_formula_error_scope='preclip',postclip_scaling_error=clip_error,learning_rates={g['name']:g['lr'] for g in opt.param_groups},full_frame_status='PASS_NATIVE_FULL_CHAIN',feature_path=str(fp),feature_sha256=sha(fp),stratum=key,stratum_visits=visits[key],EMA_commit_count=None if ema is None else ema[key]['counter'],total_EMA_commits=None if ema is None else sum(v['counter'] for v in ema.values()),additional_EMA_commits=0 if ema is None else step-1000,replay=replay,auxiliary_forward_did_not_change_crop_rng_or_buffers=True,supervised_P_frames=15,gpu=gpu,seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=True,calibration_sha256=calsha,seed_index_sha256=seedsha,discriminator_optimizer_updates=0,optimizer_internal_step=step)
  save(step,receipt);dump(b/'receipts'/f'{step}.json',receipt);append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,additional_step=step-1000,time=time.time(),PID=os.getpid()))
  dump(b/'training_status.json',dict(status='PASS' if step==2000 else 'RUNNING',PID=os.getpid(),gpu=gpu,adaptation_step=step,committed_updates=step,additional_updates=step-1000,updated_unix=time.time(),seconds_per_update=(time.time()-tic)/(step-start),peak_memory_MiB=receipt['peak_memory_MiB'],peak_RSS_MiB=receipt['peak_RSS_MiB'],EMA_commits=receipt['total_EMA_commits']))
  if step==1001:dump(ROOT/'audits'/f'first_committed_step_{branch}.json',receipt)
  if step==1001 or step%25==0:print('TRAIN',branch,step,'/2000;additional',step-1000,receipt['seconds'],flush=True)
 rows=[load(b/'receipts'/f'{i}.json') for i in range(1001,2001)]
 write(b/'gradient_summary.csv',[dict(step=r['adaptation_step'],additional_step=r['additional_step'],domain=r['domain'],QP=r['external_qp'],C1_norm=r['C1_full_gradient_norm'],additional_norm=r['additional_gradient_norm'],additional_to_C1=r['additional_to_C1_ratio'],cosine=r['additional_C1_cosine'],preclip=r['global_preclip_norm'],clip=r['clip_coefficient'],merge_error=r['gradient_formula_error'],postclip_error=r['postclip_scaling_error'],EMA_commits=r['total_EMA_commits'],optimizer_internal_step=r['optimizer_internal_step'],seconds=r['seconds'],peak_MiB=r['peak_memory_MiB']) for r in rows]);textfile(b/'train.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));check_frozen(ctx)
 assert set(visits.values())=={100}
 if ema is not None:assert set(v['counter'] for v in ema.values())=={100}
 dump(b/'training_integrity.json',dict(status='PASS',additional_updates=1000,cumulative_adaptation_updates=2000,optimizer_restored=True,optimizer_internal_step=2000,additional_supervised_P_frames=15000,exact_second_exposure=True,frozen=frozen,EMA_status='N/A' if ema is None else 'PASS',EMA_additional_commits=0 if ema is None else 1000,EMA_total_commits=None if ema is None else 2000,EMA_counters=None if ema is None else {k:v['counter'] for k,v in ema.items()},discriminator_optimizer_updates=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES,required=True);ap.add_argument('--gpu',choices=(4,5,6,7),type=int,required=True);a=ap.parse_args()
 try:main(a.branch,a.gpu)
 except Exception:error(dict(stage='train',branch=a.branch,error=traceback.format_exc()));raise
