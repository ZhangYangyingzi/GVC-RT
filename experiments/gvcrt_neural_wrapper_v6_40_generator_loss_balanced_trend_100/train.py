"""v6.40 Generator-only basic versus +0.5 DISTS; historical VJP ordering preserved."""
from io32 import *
import argparse, fcntl, math, resource, traceback

BRANCH='D_basicDISTS_noP_500'
PARENT=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000'
PARENT_N=PARENT/'branches/N_noP_BG_FD'
PLAN_SHA='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42'
PARENT_STEP0_SHA='18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074'
F_TRAIN_SHA='b29a1236a5158aa07f2b685f945723064c376404f5564d1f79cb5b35beb839d9'
F_INFER_SHA='ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc'

def basic_crop(ctx, frames, q, audit=False, dists_weight=.5):
 import torch
 import torch.nn.functional as F
 v4,im,pm,models,quality,common,params,_=ctx
 base=[torch.zeros_like(p) for p in params]; recg=[torch.zeros_like(p) for p in params]
 captured=[]; trace=[]; diagnostics=[]; actual_frames=[q]
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
   gt=frames[pos]; qa=common.frame_qp(pm,q,pos);actual_frames.append(qa)
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
     fused=(lp+dists_weight*ds if dists_weight else lp)+beta*rb+.01*pl
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
     vals=dict(LPIPS=float(lp.detach()),weighted_LPIPS=float(lp.detach()),rate_bpp=float(rb.detach()),beta_q=float(beta),beta_rate=float((beta*rb).detach()),proxy_L1=float(pl.detach()),weighted_proxy_L1=float((.01*pl).detach()),reconstruction_L1=float(rl.detach()),weighted_reconstruction_L1=float((.1383589248954108*rl).detach()),DISTS=float(ds.detach()),weighted_DISTS=dists_weight*float(ds.detach()),legacy_stat_status='inactive_not_computed',FD_status='inactive_not_computed')
     vals['total_loss']=vals['LPIPS']+vals['weighted_DISTS']+vals['beta_rate']+vals['weighted_proxy_L1']+vals['weighted_reconstruction_L1']
     assert all(math.isfinite(v) for v in (vals['LPIPS'],vals['rate_bpp'],vals['proxy_L1'],vals['reconstruction_L1'],vals['total_loss']))
     trace.append(dict(position=pos,actual_qp=qa,normalization_denominator=15,**vals))
   advance(rec)
 finally: hook.remove()
 assert len(trace)==15 and all(t['proxy_L1']==0 for t in trace)
 total=[a+b for a,b in zip(base,recg)]
 def n(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs))
 return total,dict(losses={k:sum(float(t[k]) for t in trace)/15 for k in ('LPIPS','DISTS','weighted_DISTS','rate_bpp','beta_rate','proxy_L1','weighted_proxy_L1','reconstruction_L1','weighted_reconstruction_L1','total_loss')},inactive={'legacy_stat':'not computed','FD':'not computed','Inception_train_forward':'not computed','EMA':'not instantiated'},component_gradient_modules={k:__import__('training32').groups(models,v) for k,v in component.items()},deep_gradient_checks=diagnostics,base_grad_norm=n(base),reconstruction_grad_norm=n(recg),total_grad_norm=n(total),trace=trace,actual_frame_qps=actual_frames,**({'audit_component_tensors':component} if audit else {}))

def main(branch,gpu,stop):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 import training32 as tr
 from frames32 import PlanFrames
 b=ROOT/'branches'/branch;b.mkdir(parents=True,exist_ok=True)
 lock=(b/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(PARENT_N/'checkpoints/adaptation_step_0.pt')==PARENT_STEP0_SHA
 ctx=tr.setup();v4,im,pm,models,quality,common,_,_=ctx
 source=torch.load(PARENT_N/'checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)
 allmods=dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1])
 for k,m in models.items():m.load_state_dict(source[k],strict=True)
 tr.restore_buffers(allmods,{k:v for k,v in source['buffers'].items() if k in allmods})
 active=load(ROOT/'protocol.json')['methods'][branch];assert active==['generator']
 for k in ('bridge','generator'):
  models[k].requires_grad_(k in active);models[k].train(k in active)
 params=[p for k in ('wrapper','bridge','generator') for p in models[k].parameters() if p.requires_grad]
 frozen=dict(compression=v4.compression_hash(im,pm),I=v4.module_hash(im),all_frozen_codec_parameters_and_buffers=tr.codec_frozen_state(im,pm),quality=[v4.module_hash(m) for m in quality])
 ctx=(v4,im,pm,models,quality,common,params,frozen)
 initial={k:v4.module_hash(m) for k,m in models.items()};assert initial==source['initial_module_hashes']
 frozenBG={k:initial[k] for k in ('bridge','generator') if k not in active}
 opt=torch.optim.AdamW([dict(name=k,params=list(models[k].parameters()),lr={'bridge':6e-6,'generator':1e-6}[k]) for k in active],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 assert not opt.state
 tr.restore_rng(source['crop_rng'])
 def frozen_check():
  tr.check_frozen(ctx)
  for k,h in frozenBG.items():
   assert v4.module_hash(models[k])==h
   assert all(not p.requires_grad and p.grad is None for p in models[k].parameters())
  assert all(p.grad is None for m in (im,*quality) for p in m.parameters())
  assert all(p.grad is None for n,p in pm.named_parameters() if not n.startswith('recon_generation_net.decoder.'))
 def state(step,receipt):
  frozen_check()
  return dict(branch=branch,committed_updates=step,adaptation_step=step,plan_cursor=step,signature=signature(branch),plan_sha256=sha(ROOT/'plans/first100.json'),initial_module_hashes=initial,wrapper=models['wrapper'].state_dict(),bridge=models['bridge'].state_dict(),generator=models['generator'].state_dict(),optimizer=opt.state_dict(),crop_rng=tr.rng_state(),buffers=tr.buffer_state(allmods),last_receipt=receipt)
 def receipt_commit(step,rec):
  dest=b/'receipts'/f'{step}.json'
  if not dest.exists():dump(dest,rec)
  events=[json.loads(l) for l in (b/'optimizer_calls.jsonl').read_text().splitlines()] if (b/'optimizer_calls.jsonl').exists() else []
  if not any(x['event']=='COMMITTED' and x['step']==step for x in events):append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,PID=os.getpid(),time=time.time(),receipt_sha256=sha(dest)))
 latest=b/'checkpoints/latest.pt';start=0
 step0=dict(status='PASS',source_RNG_hash=hashlib.sha256(repr(source['crop_rng']['python']).encode()+source['crop_rng']['torch'].numpy().tobytes()+source['crop_rng']['cuda'].numpy().tobytes()).hexdigest(),initial_module_hashes=initial,source_sha256=PARENT_STEP0_SHA,optimizer_state_count=0,active_modules=active,frozen_BG=frozenBG,RNG_restored=True,buffers_restored=True,plan_sha256=sha(ROOT/'plans/first100.json'))
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=True);assert st['signature']==signature(branch) and st['branch']==branch
  for k,m in models.items():m.load_state_dict(st[k],strict=True)
  opt.load_state_dict(st['optimizer']);tr.restore_rng(st['crop_rng']);tr.restore_buffers(allmods,st['buffers']);start=st['committed_updates'];assert st['plan_cursor']==start
  if start:receipt_commit(start,st['last_receipt'])
  if start in (50,100) and not (b/f'checkpoints/adaptation_step_{start}.pt').exists():atomic_torch(b/f'checkpoints/adaptation_step_{start}.pt',st)
  events=[json.loads(l) for l in (b/'optimizer_calls.jsonl').read_text().splitlines()] if (b/'optimizer_calls.jsonl').exists() else []
  for e in events:
   if e['event']=='APPLIED' and e['step']>start and not any(x['event']=='DISCARDED_REPLAY' and x.get('old_PID')==e['PID'] and x['step']==e['step'] for x in events):append(b/'optimizer_calls.jsonl',dict(event='DISCARDED_REPLAY',step=e['step'],old_PID=e['PID'],committed_cursor=start,time=time.time()))
  del st
 else:
  if branch=='G_dists':
   from diagnostic import verify
   verify(ctx,source,allmods,basic_crop)
  dump(b/'step0_audit.json',step0);st=state(0,step0);atomic_torch(latest,st);atomic_torch(b/'checkpoints/adaptation_step_0.pt',st);del st
 plan=load(ROOT/'plans/first100.json')['updates'];torch.cuda.reset_peak_memory_stats();beg=time.time()
 for step in range(start+1,stop+1):
  item=plan[step-1];row=item['source_row'];assert item['step']==step
  assert hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['source_row_canonical_sha256']
  opt.zero_grad(set_to_none=True);t=time.time();frames=PlanFrames(row,'cuda:0');gs,summary=basic_crop(ctx,frames,row['external_qp'],dists_weight=load(ROOT/'protocol.json')['dists_weights'][branch]);frames.close()
  assert [x['actual_qp'] for x in summary['trace']]==row['actual_qps']
  assert summary['actual_frame_qps']==row['actual_frame_qps']
  assert all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in gs)
  before=[p.detach().clone() for p in params];norms=tr.groups(models,gs)
  for p,g in zip(params,gs):p.grad=g
  raw=float(torch.nn.utils.clip_grad_norm_(params,1.0));assert math.isfinite(raw) and raw>0
  post=tr.norm([p.grad for p in params]);append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,PID=os.getpid(),time=time.time(),from_committed=start if step==start+1 else step-1))
  opt.step();append(b/'optimizer_calls.jsonl',dict(event='APPLIED',step=step,PID=os.getpid(),time=time.time()))
  delta=tr.groups(models,[p.detach()-a for p,a in zip(params,before)])
  assert all(delta[k]['norm']>0 and norms[k]['norm']>0 for k in active)
  frozen_check();assert {int(s['step']) for s in opt.state.values()}=={step}
  rec=dict(actual_frame_qps=summary['actual_frame_qps'],reference_detach=True,supervised_frames=15,step=step,source_plan_step=item['source_plan_step'],source_row_sha256=item['source_row_canonical_sha256'],domain=row['domain'],video=row['video'],external_qp=row['external_qp'],crop=[row['crop_x'],row['crop_y'],256,256],frame_indices=row['source_frame_indices'],losses=summary['losses'],frame_trace=summary['trace'],gradient_modules=norms,preclip_norm=raw,postclip_norm=post,clip_coefficient=min(1,1/(raw+1e-6)),parameter_delta_norms=delta,frozen_BG=frozenBG,frozen_check='PASS',optimizer_internal_step=step,physical_GPU=gpu,PID=os.getpid(),seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,time=time.time())
  st=state(step,rec);atomic_torch(latest,st)
  if step in (50,100):atomic_torch(b/f'checkpoints/adaptation_step_{step}.pt',st)
  del st;receipt_commit(step,rec)
  dump(b/'training_status.json',dict(status='PASS' if step==100 else 'SMOKE_PASS' if step==2 else 'RUNNING',committed_updates=step,optimizer_internal_step=step,physical_GPU=gpu,peak_memory_MiB=rec['peak_memory_MiB'],time=time.time()))
  if step in (1,2,10,25,50,75,100):print(branch,step,'/100',round(rec['seconds'],2),'s',flush=True)
 if stop==100:
  current={k:v4.module_hash(m) for k,m in models.items()}
  assert all((current[k]!=initial[k])==(k in active) for k in ('bridge','generator'))
  atomic_torch(ROOT/f'evaluation/checkpoints/{branch}_100.pt',{k:m.state_dict() for k,m in models.items()})
  dump(b/'final_integrity.json',dict(status='PASS',committed_updates=100,optimizer_internal_step=100,initial_module_hashes=initial,final_module_hashes=current,frozen_BG=frozenBG,active_modules=active,frozen_hashes=frozen,plan_sha256=sha(ROOT/'plans/first100.json'),inference_checkpoint_sha256=sha(ROOT/f'evaluation/checkpoints/{branch}_100.pt')))
 print('EXIT',branch,stop,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['G_basic','G_dists'],required=True);ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);ap.add_argument('--stop',type=int,choices=[0,2,100],required=True);a=ap.parse_args()
 try:main(a.branch,a.gpu,a.stop)
 except Exception:error(dict(stage='training',branch=a.branch,time=time.time(),traceback=traceback.format_exc()));raise
