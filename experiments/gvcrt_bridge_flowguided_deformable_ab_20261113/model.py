"""One raw-latent history, flow-guided modulated9-point deformable sampling."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

def grid(h,w,device):
 y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij');return torch.stack((x,y),0)[None].float()
def sample(x,coords):
 h,w=x.shape[-2:];g=torch.stack((2*(coords[:,0]+.5)/w-1,2*(coords[:,1]+.5)/h-1),-1);return F.grid_sample(x,g,mode='bilinear',padding_mode='zeros',align_corners=False)
def bounds(c,h,w):return ((c[:,0:1]>=0)&(c[:,0:1]<=w-1)&(c[:,1:2]>=0)&(c[:,1:2]<=h-1))
def resize_flow(u,size):
 h,w=u.shape[-2:];oh,ow=size;return F.interpolate(u,size=size,mode='bilinear',align_corners=False)*u.new_tensor([ow/w,oh/h]).view(1,2,1,1)
class Block(nn.Module):
 def __init__(self):super().__init__();self.a=nn.Conv2d(64,64,3,padding=1);self.b=nn.Conv2d(64,64,3,padding=1)
 def forward(self,x):return x+self.b(F.silu(self.a(x)))
class Fusion(nn.Module):
 def __init__(self):
  super().__init__();self.reliability=nn.Sequential(nn.Conv2d(59,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1));self.adapter=nn.Sequential(nn.Conv2d(54,64,1),nn.SiLU(),Block(),Block(),nn.Conv2d(64,18,1))
  for m in [self.reliability[-1],self.adapter[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
class DeformableHistory(nn.Module):
 def __init__(self,stats,group):
  super().__init__();assert group in ['A_zero_base','B_flow_guided'];self.group=group
  for name,value in [('l_rms',stats['l_rms'].view(1,18,1,1)),('evidence_rms',stats['evidence_rms'].view(1,2,1,1)),('qp_mean',torch.as_tensor(stats['qp_mean'])),('qp_std',torch.as_tensor(stats['qp_std']))]:self.register_buffer(name,value.float().clamp_min(.01) if name.endswith('rms') or name=='qp_std' else value.float())
  self.offset=nn.Sequential(nn.Conv2d(58,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,27,3,padding=1));nn.init.zeros_(self.offset[-1].weight);nn.init.zeros_(self.offset[-1].bias)
  self.dcn_weight=nn.Parameter(torch.zeros(18,18,3,3));self.dcn_bias=nn.Parameter(torch.zeros(18));self.dcn_weight.data[:,:,1,1]=2*torch.eye(18)
  self.fusion=Fusion();self.reset()
 def reset(self):self.previous=None
 def align(self,current,raw_previous,base_flow,qp):
  b,_,h,w=current.shape;coarse_coords=grid(h,w,current.device)+base_flow;rough=torch.where(bounds(coarse_coords,h,w),sample(raw_previous,coarse_coords),current)
  pred=self.offset(torch.cat((current,rough,(current-rough).abs(),base_flow,qp),1));residual=pred[:,:18].reshape(b,9,2,h,w).tanh();modulation=pred[:,18:].sigmoid()
  # torchvision offsets are [dy0,dx0,dy1,dx1,...]; base flow is always [dx,dy].
  offsets=residual+base_flow[:,[1,0],None].transpose(1,2);pk=current.new_tensor([[x,y] for y in [-1,0,1] for x in [-1,0,1]]).view(1,9,2,1,1)
  coords=grid(h,w,current.device)[:,None]+pk+base_flow[:,None]+residual[:,:,[1,0]];valid=bounds(coords.reshape(b*9,2,h,w),h,w).reshape(b,9,h,w);mask=modulation*valid
  aligned=deform_conv2d(raw_previous,offsets.reshape(b,18,h,w),self.dcn_weight,self.dcn_bias,padding=(1,1),mask=mask)
  v=valid[:,4:5].float();aligned=torch.where(v.bool(),aligned,current)
  return aligned,v,dict(offset=residual,modulation=modulation,point_validity=valid,base_flow=base_flow)
 def forward(self,l,decoded_ref_backward,decoded_ref_evidence,qp,video,frame,beta=1.,reset=False):
  p=self.previous;b,_,h,w=l.shape;ready=not reset and p is not None and p['video']==video and p['frame']==frame-1 and p['shape']==list(l.shape);ln=l/self.l_rms
  pq=p['qp'] if ready else qp;qs=(l.new_tensor([qp,pq]).view(1,2,1,1).expand(b,-1,h,w)-self.qp_mean)/self.qp_std
  base=resize_flow(decoded_ref_backward,(h,w)) if self.group=='B_flow_guided' else l.new_zeros(b,2,h,w);e=F.interpolate(decoded_ref_evidence,size=(h,w),mode='bilinear',align_corners=False)/self.evidence_rms
  if ready:an,v,aux=self.align(ln,p['l']/self.l_rms,base,qs)
  else:
   an=ln;v=l.new_zeros(b,1,h,w);aux=dict(offset=l.new_zeros(b,9,2,h,w),modulation=l.new_full((b,9,h,w),.5),point_validity=torch.zeros(b,9,h,w,device=l.device,dtype=torch.bool),base_flow=base)
  z=torch.cat((ln,an,(ln-an).abs()),1);q=v*self.fusion.reliability(torch.cat((z,e,qs,v),1)).sigmoid();delta=self.fusion.adapter(z)*self.l_rms;out=l+float(beta)*q*delta
  self.previous=dict(video=video,frame=frame,qp=qp,shape=list(l.shape),l=l.detach());aux.update(q=q,delta=delta,v=v,original_l=l,corrected_l=out,aligned_l=an*self.l_rms,history_ready=ready);return out,aux

def bridge(c,f):
 m=c['models']['bridge'];x=F.pixel_unshuffle(f.float(),2)
 for i in [0,1,2]:x=m[i](x)
 return m[3](x*torch.sigmoid(x))
def decode(c,l,qp,grad=False):
 g=c['models']['generator'];g.set_temporal_cascade_grad(grad);return (g(l,c['pm'].q_scale_recon[qp:qp+1].float(),use_tcg=True,update_buffer=True).clamp(-1,1)+1)/2
