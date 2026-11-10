"""Corrected confidence/RMS temporal loss and stateless paired integer shifts."""
from common_s2 import *
from experiment_model import reconstruct,auxiliary,detach_history,cache_snapshot,assert_cache
from model_context import assert_reset
from temporal import backwarp,temporal_errors
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
SEGMENTS=[(49,52),(53,56),(57,60),(61,63)]
def norm(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values if v is not None))

def latent_field(data,i,device):
 h,w=data['info']['native_shape'];ph,pw=data['info']['padded_shape'];lh,lw=ph//16,pw//16;flow=data['GT_backward'][i].to(device).float().detach();valid=data['GT_valid'][i].to(device).bool().detach()
 def gt(j):return data['GT'][j].permute(2,0,1)[None].to(device).float()/255
 with torch.no_grad():
  previous,bounds=backwarp(gt(i-1),flow);valid=valid&bounds;residual=(gt(i)-previous).abs().mean(1,keepdim=True)
  valid=F.pad(valid,(0,pw-w,0,ph-h),value=False);residual=F.pad(residual,(0,pw-w,0,ph-h));flow=F.pad(flow,(0,pw-w,0,ph-h))
  small=F.interpolate(flow,size=(lh,lw),mode='bilinear',align_corners=False)*flow.new_tensor([lw/pw,lh/ph]).reshape(1,2,1,1)
  O=-F.adaptive_max_pool2d(-valid.float(),(lh,lw))>.5;num=F.adaptive_avg_pool2d(residual*valid,(lh,lw));den=F.adaptive_avg_pool2d(valid.float(),(lh,lw));r=num/den.clamp_min(1e-12);confidence=torch.exp(-r/.02)
  yy,xx=torch.meshgrid(torch.arange(lh,device=device),torch.arange(lw,device=device),indexing='ij');px=xx[None]+small[:,0];py=yy[None]+small[:,1];O=O&((px>=0)&(px<=lw-1)&(py>=0)&(py<=lh-1)).unsqueeze(1)
 return small.detach(),O.detach(),confidence.detach()

def latent_loss(a,previous,flow,O,c):
 b,bounds=backwarp(previous,flow.detach());O=(O&bounds).detach();count=int(O.sum())
 if not count:return None,dict(missing=True,valid_pixels=0)
 sigma=(((O*(a.square()+b.square())).sum((2,3),keepdim=True)/(2*O.sum((2,3),keepdim=True)))+1e-8).sqrt().clamp_min(1e-4)
 d=((a-b).abs()/sigma).mean(1,keepdim=True);loss=(O*c*d).sum()/O.sum();cv=c[O];q=torch.quantile(cv.float(),cv.new_tensor([0,.25,.5,.75,1]));raw=((a-b).abs()*O).sum()/(O.sum()*a.shape[1]);rms=(a.square()*O).sum().div(O.sum()*a.shape[1]).sqrt()
 return loss,dict(missing=False,valid_pixels=count,valid_fraction=float(O.float().mean()),confidence_mean=float(cv.mean()),confidence_q0=float(q[0]),confidence_q25=float(q[1]),confidence_q50=float(q[2]),confidence_q75=float(q[3]),confidence_q100=float(q[4]),student_RMS=float(rms.detach()),raw_MC_difference=float(raw.detach()),normalized_MC_difference=float((O*d).sum().detach()/O.sum()),L_lat=float(loss.detach()),RMS_gradient=True,denominator=count)

def window_masks(data,device):
 fields={i:latent_field(data,i,device) for i in range(49,64)}
 return dict(output_frames=sum(bool(data['GT_valid'][i].any()) for i in range(49,64)),latent_frames=sum(bool(O.any()) for f,O,c in fields.values()),fields=fields)

def equivariance(c,feature,qp,delta,crop=64,check_cache=True):
 snap=cache_snapshot(c) if check_cache else None
 with torch.no_grad():target=auxiliary(c,feature,qp);target=torch.roll(target,(8*delta[0],8*delta[1]),(-2,-1))
 shifted_feature=torch.roll(feature,tuple(delta),(-2,-1));prediction=auxiliary(c,shifted_feature,qp);value=(prediction-target).abs()[...,crop:-crop,crop:-crop].mean()
 if snap is not None:assert_cache(c,snap)
 return value

def backward_window(c,data,counts,coeff,mode='total',reference=False,masks=None):
 device=c['device'];assert data['info']['training'];assert_reset(c);masks=window_masks(data,device) if masks is None else masks;nv=masks['output_frames'];nl=masks['latent_frames'];trace=[];obj=0.;previous=c['core'].unit(data['original'][0].to(device));zprev=None
 def gt(i):return data['GT'][i].permute(2,0,1)[None].to(device).float()/255
 shift=load(ROOT/'plans/auxiliary_shifts.json')[data['info']['source_row']['sample_key']]
 with torch.no_grad():
  for i in range(1,49):raw,zprev=reconstruct(c,data['features'][i],data['info']['actual_qps'][i],latent=True);previous=c['core'].unit(raw)
 for first,last in SEGMENTS:
  detach_history(c);previous=previous.detach();zprev=zprev.detach();segment=[]
  for i in range(first,last+1):
   raw,z=reconstruct(c,data['features'][i],data['info']['actual_qps'][i],True,True);out=c['core'].unit(raw);g=gt(i);spatial=mode in ['total','spatial','LPIPS_Alex','DISTS'];motion=mode in ['total','base','RAFT_motion']
   lp=checkpoint(lambda x,y:c['quality'][0](x,y,normalize=True).mean(),out,g,use_reentrant=False) if spatial or mode=='base' else out.new_zeros(())
   ds=checkpoint(lambda x,y:c['quality'][1](x,y,require_grad=True).mean(),out,g,use_reentrant=False) if spatial or mode=='base' else out.new_zeros(())
   rgb=F.l1_loss(out,g);Lsp=lp+.5*ds+.1383589248954108*rgb
   tr,te,vf=temporal_errors(out,previous,g,gt(i-1),data['GT_backward'][i].to(device),data['GT_valid'][i].to(device));f,O,conf=masks['fields'][i];lt,diag=latent_loss(z,zprev,f,O,conf)
   raft=out.new_zeros(())
   if motion:
    flow_target=data['GT_forward'][i].to(device)
    def motion_fn(prev,cur,flow_target=flow_target):
     a,b=c['transform'](prev,cur);return F.l1_loss(c['raft'](a,b)[-1],flow_target)
    raft=checkpoint(motion_fn,previous,out,use_reentrant=False)
   eq=None
   if i==last and (mode=='equiv' or mode=='total' and c['experiment_group']==GROUPS[2]):eq=equivariance(c,data['features'][i],data['info']['actual_qps'][i],shift[str(i)])
   if mode in ['total','base']:
    value=(Lsp+.05*raft)/(15*counts['batch'])
    if tr is not None:value=value+LAMBDA_TEMP*(tr+.5*te)/(nv*counts['output'])
    if mode=='total' and c['experiment_group'] in GROUPS[1:] and lt is not None:value=value+coeff['lambda_lat']*lt/(nl*counts['latent'])
    if eq is not None:value=value+coeff['lambda_equiv']*eq/(4*counts['batch'])
   elif mode=='spatial':value=Lsp/(15*counts['batch'])
   elif mode=='latent':value=None if lt is None else lt/(nl*counts['latent'])
   elif mode=='equiv':value=None if eq is None else eq/(4*counts['batch'])
   elif mode=='LPIPS_Alex':value=lp/(15*counts['batch'])
   elif mode=='DISTS':value=ds/(15*counts['batch'])
   elif mode=='RAFT_motion':value=raft/(15*counts['batch'])
   else:raise ValueError(mode)
   if value is not None:assert torch.isfinite(value),('NONFINITE_LOSS',mode,i);segment.append(value)
   trace.append(dict(frame=i,LPIPS_Alex=float(lp.detach()) if spatial or mode=='base' else None,DISTS=float(ds.detach()) if spatial or mode=='base' else None,RGB_L1=float(rgb.detach()),RAFT_motion=float(raft.detach()) if motion else None,RGB_MCResidual=None if tr is None else float(tr.detach()),Sobel_MCResidual=None if te is None else float(te.detach()),L_equiv=None if eq is None else float(eq.detach()),**diag));previous,zprev=out,z
  if segment:
   total=sum(segment);obj+=float(total.detach())
   if reference:
    gradients=torch.autograd.grad(total,c['params'],allow_unused=True)
    for p,g in zip(c['params'],gradients):
     if g is not None:
      if p.grad is None:p.grad=g.detach()
      else:p.grad.add_(g.detach())
   else:total.backward()
  detach_history(c);previous=previous.detach();zprev=zprev.detach()
  if motion:c['raft'].corr_block.corr_pyramid=[x.detach() for x in c['raft'].corr_block.corr_pyramid]
  del segment,out,raw,z,lp,ds,Lsp,rgb,raft,tr,te,lt,eq,value
 assert_reset(c)
 return dict(objective_contribution=obj,output_valid_frames=nv,latent_valid_frames=nl,trace=trace,normalization='valid pixels/channels, valid framepairs/window, global4 or global nonempty windows exactly once; equiv four calls/window')
