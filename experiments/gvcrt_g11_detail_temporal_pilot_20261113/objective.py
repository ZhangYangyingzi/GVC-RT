from util import *
import torch
import torch.nn.functional as F
from parent_objective import quality_models
from temporal import backwarp,sobel
from model import G11
METRICS=['LPIPS','DISTS','highpass_signed_error_MSE']
def highpass(x):
 k=x.new_tensor([1,4,6,4,1])/16;k=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5);return x-F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),k,groups=3)
def crop(x,clip):return x[...,clip['y']:clip['y']+256,clip['x']:clip['x']+256]
def rollout(branch,data,clip,gradient=False):
 seq,start=clip['video'],clip['start'];state=None;ys=[];stats=[];states=[];resets=[]
 for i in range(start-2,start+4):
  enabled=gradient and i>=start
  with torch.set_grad_enabled(enabled):
   d=data.tensors(seq,i);reset=bool(d['meta']['scene_cut'])
   if reset:state=None
   hh=d['h_s'];aux=None
   if branch is not None:hh,aux=branch(hh,d['current_c'],d['previous_c'],d['meta'])
   out,state=data.receiver.suffix(hh,d['x'],d['q'],state,checkpointing=enabled)
   if i<start:state=state.detach()
   else:ys.append(out);states.append(state);stats.append(aux);resets.append(reset)
 return ys,stats,states,resets

def errors(quality,ys,data,clip):
 vals={k:[] for k in METRICS};l1=[];native=[];gt=[]
 for i,y in enumerate(ys,clip['start']):
  target=data.gt(clip['video'],i);y=y[...,:target.shape[-2],:target.shape[-1]];native.append(y);gt.append(target);a,b=crop(y,clip),crop(target,clip);vals['LPIPS'].append(quality[0](a,b,normalize=True).mean());vals['DISTS'].append(quality[1](a,b,require_grad=True).mean());vals['highpass_signed_error_MSE'].append(crop(highpass(y)-highpass(target),clip).square().mean());l1.append((a-b).abs().mean())
 es={k:torch.stack(v).mean() for k,v in vals.items()};sp=es['LPIPS']+.5*es['DISTS']+.1383589248954108*torch.stack(l1).mean();return es,sp,native,gt

def temporal_loss(ys,targets,data,clip,resets):
 vals=[];counts=[]
 for j in range(1,4):
  if resets[j]:continue
  flow,valid=data.supervision(clip['video'],clip['start']+j);mask=crop(valid,clip).float();count=mask.sum();counts.append(int(count))
  if not count.item():continue
  terms=[]
  for op in [lambda x:x,sobel]:
   a,b=op(ys[j]),op(ys[j-1]);g,p=op(targets[j]),op(targets[j-1]);bw,_=backwarp(b,flow);pw,_=backwarp(p,flow);terms.append((crop((a-bw)-(g-pw),clip).abs()*mask).sum()/(count*a.shape[1]))
  vals.append(terms[0]+.5*terms[1])
 return (torch.stack(vals).mean() if vals else sum(y.sum()*0 for y in ys)),counts

def reference_errors(quality,data,clip):
 with torch.no_grad():ys,_,_,_=rollout(None,data,clip);es,_,_,_=errors(quality,ys,data,clip)
 return {k:float(v) for k,v in es.items()}
def clip_loss(branch,candidate,reference,quality,clip,mu,means,lam):
 ref=reference_errors(quality,reference,clip);ys,stats,states,resets=rollout(branch,candidate,clip,True);es,sp,yn,gt=errors(quality,ys,candidate,clip);temp,counts=temporal_loss(yn,gt,candidate,clip,resets);g=torch.stack([(es[k]-1.01*ref[k])/max(means[k],1e-8) for k in METRICS]);budget=(mu*g.relu()).sum();loss=sp+lam*temp+budget
 assert torch.isfinite(loss) and torch.isfinite(g).all(),'NONFINITE_OBJECTIVE';return dict(loss=loss,L_sp=sp,L_temp=temp,L_budget=budget,g=g,errors=es,reference=ref,stats=stats,states=states,outputs=ys,valid_counts=counts)
def grad_norm(gs):return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5
def grad_groups(branch,natives):
 out={}
 for n,p in list(branch.named_parameters())+natives:
  key='receiver_native' if n.startswith('stage3.') else n.split('.')[0];out[key]=out.get(key,0.)+(0. if p.grad is None else float(p.grad.detach().double().square().sum()))
 return {k:v**.5 for k,v in out.items()}
