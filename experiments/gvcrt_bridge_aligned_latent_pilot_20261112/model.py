"""Post-Bridge feature matching, original-latent warp and one gated residual."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def grid(h,w,device):
 y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij');return torch.stack((x,y),0)[None].float()
def sample(x,coords):
 h,w=x.shape[-2:];g=torch.stack((2*(coords[:,0]+.5)/w-1,2*(coords[:,1]+.5)/h-1),-1)
 return F.grid_sample(x,g,mode='bilinear',padding_mode='zeros',align_corners=False)
def bounds(c,h,w):return ((c[:,0:1]>=0)&(c[:,0:1]<=w-1)&(c[:,1:2]>=0)&(c[:,1:2]<=h-1))
def resize_flow(u,size):
 h,w=u.shape[-2:];oh,ow=size;return F.interpolate(u,size=size,mode='bilinear',align_corners=False)*u.new_tensor([ow/w,oh/h]).view(1,2,1,1)
def cost_volume(cur,past,u):
 b,c,h,w=cur.shape;ctr=grid(h,w,u.device)+u;cn=F.normalize(cur,dim=1,eps=1e-6);vol=[];masks=[]
 # Nine offsets per call bounds temporary storage on full-resolution evaluation.
 for dy in range(-4,5):
  offs=u.new_tensor([[dx,dy] for dx in range(-4,5)]).view(1,9,2,1,1)
  coords=ctr[:,None]+offs;cs=coords.reshape(b*9,2,h,w)
  p=sample(past[:,None].expand(-1,9,-1,-1,-1).reshape(b*9,c,h,w),cs).reshape(b,9,c,h,w)
  corr=(cn[:,None]*F.normalize(p,dim=2,eps=1e-6)).sum(2);vol.append(corr);masks.append(bounds(cs,h,w).reshape(b,9,h,w))
 corr=torch.cat(vol,1);valid=torch.cat(masks,1);anyvalid=valid.any(1,keepdim=True)
 logits=(corr/.1).masked_fill(~valid,-1e9);prob=logits.softmax(1)*valid;prob=prob/prob.sum(1,keepdim=True).clamp_min(1e-12)
 maximum=torch.where(anyvalid,corr.masked_fill(~valid,-1e9).amax(1,keepdim=True),torch.zeros_like(u[:,:1]))
 entropy=-(prob*prob.clamp_min(1e-12).log()).sum(1,keepdim=True)/math.log(81)
 entropy=torch.where(anyvalid,entropy,torch.ones_like(entropy))
 return corr.masked_fill(~valid,0),torch.cat((maximum,entropy),1),anyvalid

class MotionAligner(nn.Module):
 def __init__(self,f_rms):
  super().__init__();self.register_buffer('f_rms',f_rms.float().view(1,256,1,1).clamp_min(.01));self.project=nn.Conv2d(256,32,1)
  self.refiners=nn.ModuleList([nn.Sequential(nn.Conv2d(147,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,2,3,padding=1)) for _ in range(3)])
 def forward(self,current,past):
  cur=self.project(current/self.f_rms);old=self.project(past/self.f_rms)
  cs=[cur,F.avg_pool2d(cur,2),F.avg_pool2d(cur,4)];ps=[old,F.avg_pool2d(old,2),F.avg_pool2d(old,4)];flows={};u=None
  for level in [2,1,0]:
   c,p=cs[level],ps[level];b,_,h,w=c.shape
   u=c.new_zeros(b,2,h,w) if u is None else resize_flow(u,(h,w))
   corr,_,ok=cost_volume(c,p,u);warped=sample(p,grid(h,w,c.device)+u)
   du=self.refiners[level](torch.cat((c,warped,corr,u),1));u=torch.where(ok,u+du,torch.zeros_like(u));flows[8*(2**level)]=u
  _,e,ok=cost_volume(cur,old,u);u=torch.where(ok,u,torch.zeros_like(u));flows[8]=u;flows['valid8']=ok
  return u,e,flows

class Block(nn.Module):
 def __init__(self):super().__init__();self.a=nn.Conv2d(64,64,3,padding=1);self.b=nn.Conv2d(64,64,3,padding=1)
 def forward(self,x):return x+self.b(F.silu(self.a(x)))
class Fusion(nn.Module):
 def __init__(self,l_rms,qp_mean,qp_std):
  super().__init__();self.register_buffer('l_rms',l_rms.float().view(1,18,1,1).clamp_min(.01));self.register_buffer('qp_mean',torch.as_tensor(qp_mean).float());self.register_buffer('qp_std',torch.as_tensor(qp_std).float().clamp_min(.01))
  self.reliability=nn.Sequential(nn.Conv2d(59,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1))
  self.adapter=nn.Sequential(nn.Conv2d(54,64,1),nn.SiLU(),Block(),Block(),nn.Conv2d(64,18,1))
  for m in [self.reliability[-1],self.adapter[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
 def forward(self,l,a,e,v,qp,previous_qp):
  ln=l/self.l_rms;an=a/self.l_rms;z=torch.cat((ln,an,(ln-an).abs()),1);b,_,h,w=l.shape
  qs=l.new_tensor([qp,previous_qp]).view(1,2,1,1).expand(b,-1,h,w);qs=(qs-self.qp_mean)/self.qp_std
  q=v*torch.sigmoid(self.reliability(torch.cat((z,e,qs,v),1)));delta=self.adapter(z)*self.l_rms
  return l+q*delta,dict(q=q,delta=delta,v=v,a=a)

class AlignedLatent(nn.Module):
 def __init__(self,stats,group):
  super().__init__();assert group in ['S_current','U_unaligned','T_aligned'];self.group=group
  self.fusion=Fusion(stats['l_rms'],stats['qp_mean'],stats['qp_std'])
  self.aligner=None if group=='S_current' else MotionAligner(stats['f_rms']);self.reset()
 def reset(self):self.previous=None
 def forward(self,l,f,qp,video,frame,reset=False):
  p=self.previous;b,_,h,w=l.shape
  compatible=(not reset and p is not None and p['video']==video and p['frame']==frame-1 and p['shape']==list(l.shape))
  u=f.new_zeros(b,2,*f.shape[-2:]);e=l.new_zeros(b,2,h,w);e[:,1]=1;v=l.new_zeros(b,1,h,w);a=l;flows=None
  if compatible:
   if self.group=='S_current':v=torch.ones_like(v);pq=qp
   else:
    u,e8,flows=self.aligner(f,p['f']);e=F.interpolate(e8,size=(h,w),mode='bilinear',align_corners=False);u16=F.interpolate(u,size=(h,w),mode='bilinear',align_corners=False)/2
    if self.group=='U_unaligned':u16=torch.zeros_like(u16)
    coords=grid(h,w,l.device)+u16;v=(bounds(coords,h,w)&(F.interpolate(flows['valid8'].float(),size=(h,w),mode='bilinear',align_corners=False)>.999)).float();a=torch.where(v.bool(),sample(p['l'],coords),l);pq=p['qp']
   out,aux=self.fusion(l,a,e,v,qp,pq)
  else:
   # Keep computation graph to all fusion parameters even on missing history, with exact zero gate.
   out,aux=self.fusion(l,l,e,v,qp,qp)
  self.previous=dict(video=video,frame=frame,shape=list(l.shape),qp=qp)
  if self.group!='S_current':self.previous.update(f=f.detach(),l=l.detach())
  aux.update(u=u,flows=flows,e=e,original_l=l,corrected_l=out,history_ready=compatible)
  return out,aux

def bridge(c,f):
 m=c['models']['bridge'];x=F.pixel_unshuffle(f.float(),2)
 for i in [0,1,2]:x=m[i](x)
 return m[3](x*torch.sigmoid(x))
def decode(c,l,qp,grad=False):
 g=c['models']['generator'];g.set_temporal_cascade_grad(grad)
 return (g(l,c['pm'].q_scale_recon[qp:qp+1].float(),use_tcg=True,update_buffer=True).clamp(-1,1)+1)/2
