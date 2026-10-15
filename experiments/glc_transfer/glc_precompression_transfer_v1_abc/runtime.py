"""GLC-only import namespace and isolated sender adapters."""
from preflight import R,GLC,GVC,sha,save,state_hash
import sys,torch,time
from torch import nn
import torch.nn.functional as F
from src.models.image_model import GLC_Image
from src.models.video_model import GLC_Video
from src.utils.test_utils import get_state_dict,set_torch_env
assert str(GLC) in __import__('inspect').getfile(GLC_Video)
FA=[0,1,0,2,0,2,0,2]
class Block(nn.Module):
 def __init__(self,h):
  super().__init__();self.conv1=nn.Conv2d(h,h,3,padding=1);self.conv2=nn.Conv2d(h,h,3,padding=1)
 def forward(self,x):return x+self.conv2(F.relu(self.conv1(x)))
class Sender(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;c,h=(3,64) if kind=='B_rgb_P' else (256,44)
  self.input=nn.Conv2d(c,h,3,padding=1);self.blocks=nn.Sequential(*[Block(h) for _ in range(5)]);self.output=nn.Conv2d(h,c,3,padding=1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,x):
  y=x+self.output(self.blocks(self.input(x)))
  return y.clamp(0,1) if self.kind=='B_rgb_P' else y
def models(device):
 set_torch_env();torch.set_num_threads(2)
 out=[]
 for cls,p in ((GLC_Image,GLC/'ckpts/GLC_image.pth.tar'),(GLC_Video,GLC/'ckpts/GLC_Video.pth.tar')):
  m=cls();s=get_state_dict(str(p));keys=m.state_dict();assert not set(keys)-set(s)
  m.load_state_dict({k:s[k] for k in keys},strict=True)
  if cls is GLC_Image:m.interpolate_q()
  m.to(device).float().eval().requires_grad_(False);out.append(m)
 return out
def quality(device):
 import lpips
 from DISTS_pytorch import DISTS
 return (lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False),DISTS().to(device).eval().requires_grad_(False))
def losses(out,gt,q):
 return dict(LPIPS=q[0](out,gt,normalize=True).mean(),DISTS=q[1](out,gt,require_grad=True).mean(),L1=(out-gt).abs().mean())
def detach(dpb):return {k:v.detach() if torch.is_tensor(v) else v for k,v in dpb.items()}
def codec_input(x):
 h,w=x.shape[-2:];pad=GLC_Image.get_padding_size(h,w,64)
 return F.pad(x,pad,mode='replicate')*2-1,pad
def step(im,pm,x,dpb,q,pos,sender=None,mode='test',coder=None):
 h,w=x.shape[-2:]
 if pos and sender is not None and sender.kind=='B_rgb_P':x=sender(x)
 inp,pad=codec_input(x);hook=None
 if pos and sender is not None and sender.kind=='C_latent_P':
  hook=pm.encoder.register_forward_pre_hook(lambda m,args:(sender(args[0]),*args[1:]))
 try:
  if pos==0:
   e=im.test(inp,q) if mode=='test' else im.compress(inp,q,coder)
   d=dict(ref_frame=e['x_hat'],ref_latent=e['ref_latent'],ref_y=None)
  else:
   dpb=dict(dpb)
   if pos%32==1:dpb['ref_y']=None
   e=pm.test(inp,dpb,q,fa_idx=FA[pos%8]) if mode=='test' else pm.compress(inp,dpb,q,coder,fa_idx=FA[pos%8],reset_ref_y=pos%32==1)
   d=e['dpb']
  out=(d['ref_frame'].clamp(-1,1)+1)/2;out=out[:,:,:h,:w]
  return e,out,detach(d),pad
 finally:
  if hook is not None:hook.remove()
