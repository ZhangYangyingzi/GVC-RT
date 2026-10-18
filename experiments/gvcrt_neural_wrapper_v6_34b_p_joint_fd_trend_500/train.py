"""Two original-initialized matched FD objectives, exactly5000 committed updates each."""
from io32 import *
import argparse,fcntl,resource,math,traceback
def main(branch,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from training32 import setup,official_model,seed,rng_state,restore_rng,buffer_state,restore_buffers,crop_c1,full_pass,groups,norm,dot,check_frozen
 from calibration32b import get_stratum,feature_derivatives
 from distribution32 import to,commit
 from frames32 import PlanFrames
 from migration33 import state_digest
 b=ROOT/'branches'/branch;b.mkdir(exist_ok=True);lf=(b/'train.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(ROOT/'audits/sanity.json')['status']=='PASS';assert load(ROOT/'reference_stats/index.json')['status']=='PASS'
 cfg=load(ROOT/'config.json');ctx=setup(branch);v4,im,pm,models,quality,common,params,frozen=ctx;official=official_model();oh=tensor_hash(official.state_dict())
 seed(20261034);crop_rng=rng_state();global_rng=rng_state();initial={k:v4.module_hash(m) for k,m in models.items()}
 opt=torch.optim.AdamW([dict(name=k,params=[p for p in m.parameters() if p.requires_grad],lr=cfg['optimizer'][k+'_lr']) for k,m in models.items() if any(p.requires_grad for p in m.parameters())],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 plan_sha=sha(ROOT/'plans/shared.json');cfg_sha=sha(ROOT/'config.json');protocol_sha=sha(ROOT/'protocol.json');sig=signature('train');plan=load(ROOT/'plans/shared.json')['updates'];seedsha=sha(ROOT/'reference_stats/index.json');ema={i['stratum']:to(get_stratum(i['stratum'],'cpu')[0],'cpu') for i in load(ROOT/'reference_stats/index.json')['entries']};start=0;latest=b/'checkpoints/latest.pt';index=load(b/'checkpoint_index.json') if (b/'checkpoint_index.json').exists() else {};allmods=dict(models,im=im,pm=pm,official=official,lp=quality[0],dists=quality[1])
 if latest.exists():
  s=torch.load(latest,map_location='cpu',weights_only=True)
  accepted_signatures={sig,'fcfccc56cfb3b9ae45016df5d82e706973f99ac20c594cd834acabeea1b93e03'}
  assert s['signature'] in accepted_signatures and s['seed_index_sha256']==seedsha and s['branch']==branch
  for k,m in models.items():m.load_state_dict(s[k],strict=True)
  opt.load_state_dict(s['optimizer']);start=s['adaptation_step'];ema=s['EMA'];global_rng=s['global_rng'];restore_rng(s['crop_rng']);restore_buffers(allmods,s['buffers']);initial=s['initial_module_hashes'];assert sum(x['counter'] for x in ema.values())==start
  if s.get('last_receipt'):dump(b/'receipts'/f'{start}.json',s['last_receipt'])
  dump(b/'restore_receipts'/f'{time.time_ns()}.json',dict(status='PASS',source=str(latest),source_sha256=sha(latest),step=start,optimizer_restored=True,RNG_buffers_EMA_restored=True,signature=sig))
  del s
 else:restore_rng(crop_rng)
 def save(step,receipt=None):
  check_frozen(ctx);assert oh==tensor_hash(official.state_dict());assert sum(v['counter'] for v in ema.values())==step
  if step:assert {int(x['step']) for x in opt.state.values()}=={step}
  state=dict(schema='v634',branch=branch,adaptation_step=step,committed_optimizer_updates=step,plan_cursor=step,signature=sig,plan_sha256=plan_sha,seed_index_sha256=seedsha,config_sha256=cfg_sha,protocol_sha256=protocol_sha,initial_module_hashes=initial,frozen_hashes=frozen,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),crop_rng=rng_state(),global_rng=global_rng,buffers=buffer_state(allmods),EMA=ema,last_receipt=receipt)
  atomic_torch(latest,state)
  if step in cfg['checkpoint_steps'] or step==500:
   p=b/'checkpoints'/f'adaptation_step_{step}.pt';atomic_torch(p,state);e=ROOT/'evaluation/checkpoints'/f'{branch}_{step}.pt';atomic_torch(e,{k:m.state_dict() for k,m in models.items()});mh={k:v4.module_hash(m) for k,m in models.items()}
   index[str(step)]=dict(path=str(p),sha256=sha(p),adaptation_step=step,module_hashes=mh,model_hashes=mh,compression_hash=frozen['compression'],I_hash=frozen['I'],deployment_receiver_hashes={k:tensor_hash({n:t.half() for n,t in models[k].state_dict().items()}) for k in ('bridge','generator')},inference=dict(path=str(e),sha256=sha(e),module_hashes=mh),optimizer_internal_step=step,EMA_counters={k:v['counter'] for k,v in ema.items()},initialization='original public GVC-RT only',P_mode='Identity' if branch==BRANCHES[0] else 'NeuralWrapper');dump(b/'checkpoint_index.json',index)
 if start==0 and '0' not in index:save(0)
 elif start in cfg['checkpoint_steps'] and str(start) not in index:save(start)
 torch.cuda.reset_peak_memory_stats();tic=time.time()
 for step in range(start+1,501):
  t=time.time();item=plan[step-1];row=source_row(item);assert item['step']==step and hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['source_row_canonical_sha256']
  opt.zero_grad(set_to_none=True);before_scope=buffer_state(allmods);f=PlanFrames(row,'cuda:0');g0,values,trace=crop_c1(ctx,f,row['external_qp']);f.close();crng=rng_state();cbuf=buffer_state(allmods)
  restore_rng(global_rng);restore_buffers(allmods,before_scope);f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close();gnext=rng_state();key=f'{row["domain"]}:{row["external_qp"]}';ema_before_hash=state_digest(ema[key]);old=to(ema[key],'cuda:0');_,ref=get_stratum(key,'cuda:0');fg,ls,candidate=feature_derivatives(z,old,ref,('FD',))
  restore_rng(global_rng);restore_buffers(allmods,before_scope);f=PlanFrames(row,'cuda:0',full=True);_,_,unit,replay=full_pass(ctx,f,row['external_qp'],official,fg['FD'],expected=z);f.close();restore_rng(crng);restore_buffers(allmods,cbuf);global_rng=gnext
  gu=groups(models,unit);gbase=groups(models,g0);extra=[g*cfg['lambda_FD'] for g in unit];total=[a+e for a,e in zip(g0,extra)];gmods=groups(models,total)
  for name,m in models.items():
   if any(p.requires_grad for p in m.parameters()):assert gu[name]['finite'] and gu[name]['norm']>0 and gmods[name]['finite'] and gmods[name]['norm']>0,(step,name,gu[name])
  assert [x['actual_qp'] for x in trace]==row['actual_qps'];assert math.isfinite(ls['FD'])
  pre=norm(total);formula=norm([g-a-e for g,a,e in zip(total,g0,extra)])/max(pre,1e-30);assert formula<=1e-6
  # Compute before assigning grad aliases and in-place clipping.
  for p,g in zip(params,total):p.grad=g
  before=[p.detach().clone() for p in params];n=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(n) and n>0;clip=min(1.,1./(n+1e-6));post=norm([p.grad for p in params]);cliperr=norm([p.grad-(a+e)*clip for p,a,e in zip(params,g0,extra)])/max(post,1e-30);assert cliperr<=1e-6
  append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,committed_before=step-1,time=time.time(),PID=os.getpid()));opt.step();append(b/'optimizer_calls.jsonl',dict(event='APPLIED',step=step,time=time.time(),PID=os.getpid()));assert all(p.isfinite().all() for p in params);deltas=groups(models,[p.detach()-a for p,a in zip(params,before)]);assert all(v['norm']>0 for k,v in deltas.items() if v['trainable_parameters'])
  ema[key]=to(commit(old,candidate),'cpu');assert sum(v['counter'] for v in ema.values())==step
  r=dict(row,branch=branch,adaptation_step=step,plan_row_sha256=item['source_row_canonical_sha256'],losses=values,FD=ls['FD'],weighted_FD=cfg['lambda_FD']*ls['FD'],total_loss=values['total_loss']+cfg['lambda_FD']*ls['FD'],gradient_C1=gbase,gradient_FD_unit=gu,gradient_total=gmods,gradient_formula_error=formula,postclip_scaling_error=cliperr,C1_norm=norm(g0),FD_unit_norm=norm(unit),FD_weighted_norm=norm(extra),FD_C1_ratio=norm(extra)/max(norm(g0),1e-30),FD_C1_cosine=dot(g0,extra)/max(norm(g0)*norm(extra),1e-30),global_preclip_norm=pre,global_postclip_norm=post,clip_coefficient=clip,parameter_delta=deltas,learning_rates={g['name']:g['lr'] for g in opt.param_groups},EMA_before_sha256=ema_before_hash,EMA_after_sha256=state_digest(ema[key]),EMA_stratum=key,EMA_stratum_counter=ema[key]['counter'],EMA_total_commits=step,frame_trace=trace,replay=replay,supervised_P_frames=15,warmup_P_frames=48,native_I_bypass=True,auxiliary_RNG_buffers_restored=True,optimizer_internal_step=step,physical_GPU=gpu,seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=True,discriminator_optimizer_updates=0)
  save(step,r);dump(b/'receipts'/f'{step}.json',r);append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,time=time.time(),PID=os.getpid()));dump(b/'training_status.json',dict(status='PASS' if step==500 else 'RUNNING',PID=os.getpid(),committed_updates=step,adaptation_step=step,EMA_commits=step,updated_unix=time.time(),peak_memory_MiB=r['peak_memory_MiB'],peak_RSS_MiB=r['peak_RSS_MiB'],seconds_per_update=(time.time()-tic)/(step-start)))
  if step==1:dump(ROOT/'audits'/f'first_committed_step_{branch}.json',r)
  if step==1 or step%25==0 or step==500:print('TRAIN',branch,step,'/500',r['seconds'],flush=True)
 rows=[load(b/'receipts'/f'{s}.json') for s in range(start+1,501)];assert sum(v['counter'] for v in ema.values())==500;check_frozen(ctx)
 write(b/'gradient_summary.csv',[dict(step=r['adaptation_step'],domain=r['domain'],QP=r['external_qp'],C1_norm=r['C1_norm'],FD_unit_norm=r['FD_unit_norm'],FD_weighted_norm=r['FD_weighted_norm'],FD_C1_ratio=r['FD_C1_ratio'],cosine=r['FD_C1_cosine'],preclip=r['global_preclip_norm'],clip=r['clip_coefficient'],merge_error=r['gradient_formula_error'],postclip_error=r['postclip_scaling_error'],wrapper_C1=r['gradient_C1']['wrapper']['norm'],wrapper_FD=r['gradient_FD_unit']['wrapper']['norm'],bridge_C1=r['gradient_C1']['bridge']['norm'],bridge_FD=r['gradient_FD_unit']['bridge']['norm'],generator_C1=r['gradient_C1']['generator']['norm'],generator_FD=r['gradient_FD_unit']['generator']['norm'],EMA_commits=r['EMA_total_commits'],seconds=r['seconds'],peak_MiB=r['peak_memory_MiB']) for r in rows]);dump(b/'training_integrity.json',dict(status='PASS',updates_added=500-start,updates_total=500,optimizer_internal_step=500,EMA_total_commits=500,EMA_counters={k:v['counter'] for k,v in ema.items()},frozen=frozen,clip_frequency=sum(r['clip_coefficient']<1 for r in rows)/len(rows),initial=initial,final={k:v4.module_hash(m) for k,m in models.items()},discriminator_optimizer_updates=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES,required=True);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
 try:main(a.branch,a.gpu)
 except Exception:error(dict(stage='train',branch=a.branch,error=traceback.format_exc()));raise
