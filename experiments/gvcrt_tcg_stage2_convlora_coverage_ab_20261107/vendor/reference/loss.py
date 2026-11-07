"""v6.40 basic_crop, extended only with detached adjacent-frame RAFT motion loss.

Historical reconstruction VJP scaling/order and normalization are retained.
"""
import math
def basic_crop(ctx, frames, q, audit=False, dists_weight=.5, raft=None, raft_transform=None):
 import torch
 import torch.nn.functional as F
 v4,im,pm,models,quality,common,params,_=ctx
 base=[torch.zeros_like(p) for p in params]; recg=[torch.zeros_like(p) for p in params]
 captured=[]; trace=[]; diagnostics=[]; actual_frames=[q]
 previous_gt=None; previous_rec=None
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
     assert previous_gt is not None and previous_rec is not None
     ga,gb=raft_transform(previous_gt,gt)
     ra,rb_motion=raft_transform(previous_rec.detach(),rec)
     with torch.no_grad(): target_flow=raft(ga,gb)[-1]
     predicted_flow=raft(ra,rb_motion)[-1]
     motion=F.l1_loss(predicted_flow,target_flow)
     assert torch.isfinite(motion)
     rl=F.l1_loss(rec,gt)
     pl=F.l1_loss(proxy,gt)
     rb=rate/65536
     fused=(lp+dists_weight*ds if dists_weight else lp)+beta*rb+.01*pl+.01*motion
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
     vals=dict(motion_loss=float(motion.detach()),weighted_motion_loss=.01*float(motion.detach()),LPIPS=float(lp.detach()),weighted_LPIPS=float(lp.detach()),rate_bpp=float(rb.detach()),beta_q=float(beta),beta_rate=float((beta*rb).detach()),proxy_L1=float(pl.detach()),weighted_proxy_L1=float((.01*pl).detach()),reconstruction_L1=float(rl.detach()),weighted_reconstruction_L1=float((.1383589248954108*rl).detach()),DISTS=float(ds.detach()),weighted_DISTS=dists_weight*float(ds.detach()),legacy_stat_status='inactive_not_computed',FD_status='inactive_not_computed')
     vals['total_loss']=vals['LPIPS']+vals['weighted_DISTS']+vals['beta_rate']+vals['weighted_proxy_L1']+vals['weighted_reconstruction_L1']+vals['weighted_motion_loss']
     assert all(math.isfinite(v) for v in (vals['LPIPS'],vals['rate_bpp'],vals['proxy_L1'],vals['reconstruction_L1'],vals['total_loss']))
     trace.append(dict(position=pos,actual_qp=qa,normalization_denominator=15,**vals))
   advance(rec)
   previous_gt=gt.detach(); previous_rec=rec.detach()
 finally: hook.remove()
 assert len(trace)==15 and all(t['proxy_L1']==0 for t in trace)
 total=[a+b for a,b in zip(base,recg)]
 def n(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs))
 return total,dict(losses={k:sum(float(t[k]) for t in trace)/15 for k in ('motion_loss','weighted_motion_loss','LPIPS','DISTS','weighted_DISTS','rate_bpp','beta_rate','proxy_L1','weighted_proxy_L1','reconstruction_L1','weighted_reconstruction_L1','total_loss')},inactive={'legacy_stat':'not computed','FD':'not computed','Inception_train_forward':'not computed','EMA':'not instantiated'},component_gradient_modules={k:__import__('training32').groups(models,v) for k,v in component.items()},deep_gradient_checks=diagnostics,base_grad_norm=n(base),reconstruction_grad_norm=n(recg),total_grad_norm=n(total),trace=trace,actual_frame_qps=actual_frames,**({'audit_component_tensors':component} if audit else {}))
