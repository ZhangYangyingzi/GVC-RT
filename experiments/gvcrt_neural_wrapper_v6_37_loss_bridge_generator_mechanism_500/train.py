"""Fixed 500-step no-P nested basic-objective branch."""
from io32 import *
import argparse, fcntl, math, resource, traceback

BRANCH='D_basicDISTS_noP_500'
PARENT=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000'
PARENT_N=PARENT/'branches/N_noP_BG_FD'
PLAN_SHA='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42'
PARENT_STEP0_SHA='18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074'
F_TRAIN_SHA='b29a1236a5158aa07f2b685f945723064c376404f5564d1f79cb5b35beb839d9'
F_INFER_SHA='ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc'

def basic_crop(ctx, frames, q, audit=False):
 import torch
 import torch.nn.functional as F
 v4,im,pm,models,quality,common,params,_=ctx
 base=[torch.zeros_like(p) for p in params]; recg=[torch.zeros_like(p) for p in params]
 captured=[]; trace=[]; diagnostics=[]
 component={k:[torch.zeros_like(p) for p in params] for k in ("LPIPS","weighted_reconstruction","weighted_DISTS","rate_proxy")} if audit else {}
 hook=pm.dec.register_forward_hook(lambda _m,_a,o:captured.append(o))
 def advance(rec):
  assert len(captured)==1
  with torch.no_grad():
   pm.add_ref_frame(captured.pop().detach(),(rec.detach()*2-1).half())
  assert not pm.dpb[0].frame.requires_grad and not pm.dpb[0].feature.requires_grad
 try:
  pm.clear_dpb();pm.set_curr_poc(0)
  with torch.no_grad(): pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),q)['x_hat'])
  beta=common.beta_q('schedule_s1p0',q)
  for pos in range(1,64):
   gt=frames[pos]; qa=common.frame_qp(pm,q,pos)
   with torch.enable_grad() if pos>=49 else torch.no_grad():
    proxy=models['wrapper'](gt)
    rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
    if pos>=49:
     lp=quality[0](rec,gt,normalize=True).mean()
     deep=[];original_once=quality[1].forward_once
     if audit:
      def capture(x):
       fs=original_once(x)
       if x.requires_grad:deep.append(fs[4])
       return fs
      quality[1].forward_once=capture
     try:ds=quality[1](rec,gt,require_grad=True).mean()
     finally:quality[1].forward_once=original_once
     rl=F.l1_loss(rec,gt)
     pl=F.l1_loss(proxy,gt)
     rb=rate/65536
     fused=lp+.5*ds+beta*rb+.01*pl
     if audit:
      for name,val,scale,weight in [("LPIPS",lp/15,1,1),("weighted_reconstruction",rl*(4096/15),4096,.1383589248954108),("weighted_DISTS",.5*ds/15,1,1),("rate_proxy",(beta*rb+.01*pl)/15,1,1)]:
       gg=torch.autograd.grad(val,params,retain_graph=True,allow_unused=True) if val.requires_grad else [None]*len(params)
       for dst,g in zip(component[name],gg):
        if g is not None:dst.add_(g.detach().float()/scale*weight)
      assert len(deep)==1
      dg=torch.autograd.grad(.5*ds/15,[rec,deep[0]]+params,retain_graph=True,allow_unused=True)
      assert all(g is not None and torch.isfinite(g).all() and g.float().norm()>0 for g in dg[:2])
      diagnostics.append(dict(position=pos,RGB_gradient_norm=float(dg[0].float().norm()),DISTS_feature4_gradient_norm=float(dg[1].float().norm()),feature_shape=list(deep[0].shape),finite=True))
     gs=torch.autograd.grad(fused/15,params,retain_graph=True,allow_unused=True)
     for dst,g in zip(base,gs):
      if g is not None: dst.add_(g.detach().float())
     rg=torch.autograd.grad(rl*(4096/15),params,allow_unused=True)
     for dst,g in zip(recg,rg):
      if g is not None: dst.add_(g.detach().float()/4096*.1383589248954108)
     vals=dict(LPIPS=float(lp.detach()),weighted_LPIPS=float(lp.detach()),rate_bpp=float(rb.detach()),beta_q=float(beta),beta_rate=float((beta*rb).detach()),proxy_L1=float(pl.detach()),weighted_proxy_L1=float((.01*pl).detach()),reconstruction_L1=float(rl.detach()),weighted_reconstruction_L1=float((.1383589248954108*rl).detach()),DISTS=float(ds.detach()),weighted_DISTS=.5*float(ds.detach()),legacy_stat_status='inactive_not_computed',FD_status='inactive_not_computed')
     vals['total_loss']=vals['LPIPS']+vals['weighted_DISTS']+vals['beta_rate']+vals['weighted_proxy_L1']+vals['weighted_reconstruction_L1']
     assert all(math.isfinite(v) for v in (vals['LPIPS'],vals['rate_bpp'],vals['proxy_L1'],vals['reconstruction_L1'],vals['total_loss']))
     trace.append(dict(position=pos,actual_qp=qa,normalization_denominator=15,**vals))
   advance(rec)
 finally: hook.remove()
 assert len(trace)==15 and all(t['proxy_L1']==0 for t in trace)
 total=[a+b for a,b in zip(base,recg)]
 def n(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs))
 return total,dict(losses={k:sum(float(t[k]) for t in trace)/15 for k in ('LPIPS','DISTS','weighted_DISTS','rate_bpp','beta_rate','proxy_L1','weighted_proxy_L1','reconstruction_L1','weighted_reconstruction_L1','total_loss')},inactive={'legacy_stat':'not computed','FD':'not computed','Inception_train_forward':'not computed','EMA':'not instantiated'},component_gradient_modules={k:__import__('training32').groups(models,v) for k,v in component.items()},deep_gradient_checks=diagnostics,base_grad_norm=n(base),reconstruction_grad_norm=n(recg),total_grad_norm=n(total),trace=trace)

def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from training32 import setup,seed,rng_state,restore_rng,buffer_state,restore_buffers,norm,groups,check_frozen
 from frames32 import PlanFrames
 b=ROOT/'branches'/BRANCH;b.mkdir(parents=True,exist_ok=True)
 lock=(b/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(ROOT/'plans/shared.json')==PLAN_SHA
 assert sha(PARENT_N/'checkpoints/adaptation_step_0.pt')==PARENT_STEP0_SHA
 assert load(ROOT/'audits/step0_gradient.json')['status']=='PASS'
 ctx=setup('N_noP_BG_FD');v4,im,pm,models,quality,common,params,frozen=ctx
 assert len(params)>0 and all(p.dtype==torch.float32 for p in params)
 allmods=dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1])
 source=torch.load(PARENT_N/'checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)
 assert source['adaptation_step']==0 and source['committed_optimizer_updates']==0 and source['plan_cursor']==0
 assert source['initial_module_hashes']['bridge']==v4.module_hash(models['bridge'])
 assert source['initial_module_hashes']['generator']==v4.module_hash(models['generator'])
 for k in ('wrapper','bridge','generator'):models[k].load_state_dict(source[k],strict=True)
 restore_buffers(allmods,{k:v for k,v in source['buffers'].items() if k in allmods})
 restore_rng(source['crop_rng'])
 initial={k:v4.module_hash(m) for k,m in models.items()}
 assert initial==source['initial_module_hashes']
 assert initial['wrapper']=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
 # Fresh, matched optimizer (no full-objective moments are restored).
 opt=torch.optim.AdamW([dict(name=k,params=[p for p in models[k].parameters() if p.requires_grad],lr={'bridge':6e-6,'generator':1e-6}[k]) for k in ('bridge','generator')],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 seed_rng=rng_state(); buffers=buffer_state(allmods)
 plan=load(ROOT/'plans/shared.json')['updates'][:500]
 assert len(plan)==500
 branchindex={}
 step0={'status':'PASS','source_checkpoint':str(PARENT_N/'checkpoints/adaptation_step_0.pt'),'source_sha256':sha(PARENT_N/'checkpoints/adaptation_step_0.pt'),'source_step':0,'optimizer_policy':'fresh AdamW at step0','initial_module_hashes':initial,'frozen_hashes':frozen,'crop_rng_restored':True,'plan_sha256':PLAN_SHA,'signature':signature('D_basicDISTS')}
 dump(b/'audits_step0.json',step0)
 latest=b/'checkpoints/latest.pt';start=0
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=True)
  assert st['signature']==signature('D_basicDISTS') and st['plan_sha256']==PLAN_SHA and st['branch']==BRANCH
  for k in ('wrapper','bridge','generator'):models[k].load_state_dict(st[k],strict=True)
  opt.load_state_dict(st['optimizer']);start=int(st['committed_updates']);assert st['plan_cursor']==start
  restore_rng(st['crop_rng']);restore_buffers(allmods,st['buffers']);assert start<=500
  if start:
   rr=st['last_receipt'];dump(b/'receipts'/f'{start}.json',rr)
   append(b/'recovery.jsonl',dict(PID=os.getpid(),authoritative_committed_checkpoint=start,signature=st['signature'],time=time.time()))
  del st
 else:
  for k,m in models.items():m.load_state_dict(source[k],strict=True)
  restore_rng(source['crop_rng']);restore_buffers(allmods,{k:v for k,v in source['buffers'].items() if k in allmods})
 def save(step,receipt):
  check_frozen(ctx)
  state=dict(schema='v636',branch=BRANCH,adaptation_step=step,committed_updates=step,plan_cursor=step,signature=signature('D_basicDISTS'),plan_sha256=PLAN_SHA,seed=20261034,initial_module_hashes=initial,frozen_hashes=frozen,wrapper=models['wrapper'].state_dict(),bridge=models['bridge'].state_dict(),generator=models['generator'].state_dict(),optimizer=opt.state_dict(),crop_rng=rng_state(),buffers=buffer_state(allmods),last_receipt=receipt)
  atomic_torch(latest,state)
  if step in (0,100,250,500):
   p=b/'checkpoints'/f'adaptation_step_{step}.pt';atomic_torch(p,state)
   if step:
    inf=ROOT/'evaluation/checkpoints'/f'{BRANCH}_{step}.pt';atomic_torch(inf,{k:models[k].state_dict() for k in ('wrapper','bridge','generator')})
   else: inf=None
   ix={**load(b/'checkpoint_index.json')} if (b/'checkpoint_index.json').exists() else {}
   ix[str(step)]=dict(path=str(p),sha256=sha(p),adaptation_step=step,committed_updates=step,optimizer_internal_step=step,initial_module_hashes=initial,module_hashes={k:v4.module_hash(m) for k,m in models.items()},inference=None if inf is None else dict(path=str(inf),sha256=sha(inf)))
   dump(b/'checkpoint_index.json',ix)
 if start==0 and not latest.exists():save(0,step0)
 losses_path=b/'training_log.jsonl';calls=b/'optimizer_calls.jsonl'
 tic0=time.time();torch.cuda.reset_peak_memory_stats();attempt=str(os.getpid())+'_'+str(time.time_ns())
 for step in range(start+1,501):
  item=plan[step-1];row=item['source_row']
  assert item['step']==step
  assert hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['source_row_canonical_sha256']
  opt.zero_grad(set_to_none=True);t=time.time();before_buffers=buffer_state(allmods)
  frames=PlanFrames(row,'cuda:0');grads,summary=basic_crop(ctx,frames,row['external_qp']);frames.close()
  assert [x['actual_qp'] for x in summary['trace']]==row['actual_qps']
  assert all(g is None or bool(g.isfinite().all()) for g in grads)
  before=[p.detach().clone() for p in params];pre=norm(grads);module_pre=groups(models,grads)
  for p,g in zip(params,grads):p.grad=g
  preclip=float(torch.nn.utils.clip_grad_norm_(params,1.0));clip=min(1.,1./(preclip+1e-6));post=norm([p.grad for p in params])
  assert math.isfinite(preclip) and preclip>0 and math.isfinite(post)
  append(calls,dict(event='STARTED',step=step,attempt=attempt,time=time.time(),PID=os.getpid()))
  opt.step();append(calls,dict(event='APPLIED',step=step,attempt=attempt,time=time.time(),PID=os.getpid()))
  assert all(torch.isfinite(p).all() for p in params)
  delta=[p.detach()-p0 for p,p0 in zip(params,before)];deltas=groups(models,[torch.zeros(0) if d is None else d for d in delta])
  assert norm(delta)>0
  after_buffers=buffer_state(allmods)
  for name in ('im','pm','lp','dists'):
   for k,v in before_buffers[name].items():assert torch.equal(v,after_buffers[name][k]),(name,k)
  record=dict(step=step,source_plan_step=item['step'],plan_row_sha256=item['source_row_canonical_sha256'],domain=row['domain'],video=row['video'],source_sha256=row['source_sha256'],canonical_indices=row['canonical_indices'],crop=[row['crop_x'],row['crop_y'],256,256],external_qp=row['external_qp'],losses=summary['losses'],inactive=summary['inactive'],gradient_norms=dict(total_preclip=preclip,global_norm_raw=pre,global_postclip=post,clip_coefficient=clip,bridge_preclip=module_pre['bridge']['norm'],generator_preclip=module_pre['generator']['norm'],bridge_postclip=groups(models,grads)['bridge']['norm'],generator_postclip=groups(models,grads)['generator']['norm']),parameter_delta_norms={k:v['norm'] for k,v in deltas.items()},learning_rates={g['name']:g['lr'] for g in opt.param_groups},optimizer_internal_step=step,elapsed_seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,freeze_check='PASS',finite=True)
  record.update(PID=os.getpid(),attempt=attempt,frame_trace=summary['trace'],frozen_hashes=frozen)
  save(step,record)
  recpath=b/'receipts'/f'{step}.json'
  if recpath.exists():dump(b/'uncommitted_attempts'/f'{step}_{time.time_ns()}.json',load(recpath))
  dump(recpath,record);append(losses_path,record);append(calls,dict(event='COMMITTED',step=step,attempt=attempt,time=time.time(),PID=os.getpid(),checkpoint_sha256=sha(latest)))
  if step in (1,2,5,10,25,50,100,250,500):
   dump(b/'training_status.json',dict(status='RUNNING' if step<500 else 'PASS',committed_updates=step,elapsed_seconds=time.time()-tic0,seconds_per_update=(time.time()-tic0)/(step-start),gpu=gpu,peak_memory_MiB=record['peak_memory_MiB']))
   print('TRAIN',step,'/500',round(record['elapsed_seconds'],2),'s',flush=True)
 dump(b/'training_status.json',dict(status='PASS',committed_updates=500,optimizer_internal_step=500,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,completed_unix=time.time()))
 print('TRAIN_COMPLETE',BRANCH,500,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
 try:main(a.gpu)
 except Exception:
  error(dict(stage='D_train',error=traceback.format_exc()));raise
