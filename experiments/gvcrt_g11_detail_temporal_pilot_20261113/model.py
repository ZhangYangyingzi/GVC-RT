"""G11 only: frozen full S, flow-guided DCN128 and one native receiver block."""
from util import *
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from alignment import grid,sample,bounds
from parent_model import Fusion as SpatialS,Residual,Receiver,restore_b300

def load_cref():
 c=restore_b300();path=Path(load(ROOT/'weight_manifest.json')['D12_S_900']['path']);assert sha(path)==load(ROOT/'weight_manifest.json')['D12_S_900']['sha256'];cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['step']==900 and cp['condition']=='S' and cp['scale']=='D12' and cp['seed']==20261112
 assert thash(c['pm'].state_dict().items())==cp['frozen_hashes']['pm'] and thash(c['im'].state_dict().items())==cp['frozen_hashes']['im'],'PARENT_BASE_HASH_MISMATCH'
 s=SpatialS(512).cuda().float().eval().requires_grad_(False);s.load_state_dict(cp['state'],strict=True);c['S']=s
 assert all(m.inference_gain==.5 for m in c['models']['generator'].modules() if hasattr(m,'inference_gain'))
 return c

def receiver_named(c):
 block=c['models']['generator'].stage3.blocks[0];names=[]
 for name,module in block.named_modules():
  if isinstance(module,nn.Conv2d) and 'lora' not in name.split('.'):
   for pn,p in module.named_parameters(recurse=False):names.append(('stage3.blocks.0.'+(name+'.' if name else '')+pn,p))
 assert len({id(p) for _,p in names})==len(names);assert sum(p.numel() for _,p in names)==988800,[(n,p.numel()) for n,p in names]
 return names

def allow_receiver(c):
 names=receiver_named(c)
 for _,p in names:p.requires_grad_(True)
 actual={id(p) for p in c['pm'].parameters() if p.requires_grad};assert actual=={id(p) for _,p in names}
 return names

def frozen_hashes(c):
 allowed={id(p) for _,p in receiver_named(c)};frozen=[]
 for n,p in c['pm'].state_dict(keep_vars=True).items():
  if id(p) not in allowed:frozen.append((n,p))
 return dict(P_excluding_native_receiver=thash(frozen),I=thash(c['im'].state_dict().items()),Bridge=thash(c['models']['bridge'].state_dict().items()),S=thash(c['S'].state_dict().items()),codec=thash((n,t) for n,t in c['pm'].state_dict().items() if not n.startswith('recon_generation_net.')))

def native_state(c):return {n:p.detach().cpu().clone() for n,p in receiver_named(c)}
def load_native(c,state):
 names=receiver_named(c);assert set(state)=={n for n,p in names}
 with torch.no_grad():
  for n,p in names:p.copy_(state[n].to(p.device))

class G11(nn.Module):
 def __init__(self):
  super().__init__();self.predictor=nn.Sequential(nn.Conv2d(388,128,3,padding=1,bias=False),nn.SiLU(),nn.Conv2d(128,27,3,padding=1,bias=True));nn.init.zeros_(self.predictor[-1].weight);nn.init.zeros_(self.predictor[-1].bias)
  self.dcn_weight=nn.Parameter(torch.zeros(128,128,3,3));self.dcn_weight.data[:,:,1,1]=2*torch.eye(128)
  self.P_h=nn.Sequential(nn.Conv2d(512,128,1,bias=False),nn.SiLU());self.fusion=nn.Sequential(nn.Conv2d(514,128,1,bias=False),nn.SiLU(),Residual(),Residual());self.output=nn.Conv2d(128,512,1,bias=False);nn.init.zeros_(self.output.weight)
 def align(self,current,previous,meta):
  b,_,h,w=current.shape;base=meta['coords']-grid(h,w,current.device);external=meta['external'].bool();coarse=sample(previous,meta['coords']);coarse=torch.where(external & bounds(meta['coords'],h,w),coarse,current);err=meta['fb_error'];pred=self.predictor(torch.cat((current,coarse,current-coarse,base,external.float(),err),1));offset=pred[:,:18].reshape(b,9,2,h,w).tanh();mod=pred[:,18:].sigmoid();pk=current.new_tensor([[x,y] for y in [-1,0,1] for x in [-1,0,1]]).view(1,9,2,1,1)
  coords=grid(h,w,current.device)[:,None]+pk+base[:,None]+offset[:,:,[1,0]];valid=bounds(coords.reshape(b*9,2,h,w),h,w).reshape(b,9,h,w)&external
  off=offset+base[:,[1,0],None].transpose(1,2);a=deform_conv2d(previous,off.reshape(b,18,h,w),self.dcn_weight,padding=(1,1),mask=mod*valid);v=valid[:,4:5].float();a=torch.where(v.bool(),a,current)
  return a,v,dict(offset=offset,modulation=mod,point_validity=valid,base_flow=base)
 def forward(self,h_s,current,raw_previous,meta):
  a,v,aux=self.align(current,raw_previous,meta);delta=self.output(self.fusion(torch.cat((self.P_h(h_s),current,a,current-a,v,meta['fb_error']),1)));out=h_s+v*delta;aux.update(v=v,delta=delta,correction=v*delta);return out,aux
