"""Strict independent Original reference; frozen RAFT receives decoded RGB only."""
from util import *
import torch,types
import torch.nn.functional as F
from model import sample,grid,bounds,resize_flow

def load_raft():
 from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
 from common_s2 import RAFT,RAFT_SHA
 assert sha(RAFT)==RAFT_SHA;m=raft_large(weights=None,progress=False);m.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True);return m.cuda().eval().requires_grad_(False),Raft_Large_Weights.C_T_SKHT_V2.transforms()

def fp32_original(pm):
 g=pm.recon_generation_net;g.float()
 for m in g.modules():
  if m.__class__.__name__ in ['DepthConvBlock','StageBlock']:
   m.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),m)
   if hasattr(m,'proxy'):m.proxy=None
 if hasattr(g.decoder,'forward_torch'):g.decoder.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),g.decoder)
 original=g.forward;g.forward=types.MethodType(lambda self,f,q:original(f.float(),q.float()),g)

def independent_original():
 from original_context import runtime
 from original_runtime import load_models
 rt=runtime(0);im,pm=load_models(rt);fp32_original(pm);return rt,im,pm

def coexist_original():
 # B300 objects retain their original class/function globals. Original is loaded
 # under a temporary clean src namespace and never receives any adapted weights.
 saved={n:m for n,m in sys.modules.items() if n=='src' or n.startswith('src.')}
 for n in saved:del sys.modules[n]
 try:
  rt,im,pm=independent_original();original_modules={n:m for n,m in sys.modules.items() if n=='src' or n.startswith('src.')}
 finally:
  for n in list(sys.modules):
   if n=='src' or n.startswith('src.'):del sys.modules[n]
  sys.modules.update(saved)
 assert not any('tcg' in n.lower() or 'lora' in n.lower() for n,p in pm.named_parameters())
 rt['independent_src_modules']=original_modules;return rt,im,pm

def decoded_flow(raft,transform,current,previous):
 a,b=transform(current,previous);back=raft(a,b,num_flow_updates=12)[-1];forward=raft(b,a,num_flow_updates=12)[-1];h,w=back.shape[-2:];coords=grid(h,w,back.device)+back;valid=bounds(coords,h,w)
 photo=(current-sample(previous,coords)).abs().mean(1,keepdim=True);fb=(back+sample(forward,coords)).square().sum(1,keepdim=True).sqrt();e=torch.cat((photo,fb),1)*valid
 return back,forward,e
