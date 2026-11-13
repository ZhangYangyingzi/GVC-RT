"""Independent Original namespace and matched FP32 reconstruction, codec weights unchanged."""
from util import *
import torch,types

def build_original():
 saved={n:m for n,m in sys.modules.items() if n=='src' or n.startswith('src.')}
 for n in saved:del sys.modules[n]
 try:
  from original_context import runtime
  from original_runtime import load_models
  rt=runtime(0);im,pm=load_models(rt)
 finally:
  for n in list(sys.modules):
   if n=='src' or n.startswith('src.'):del sys.modules[n]
  sys.modules.update(saved)
 g=pm.recon_generation_net;g.float()
 for m in g.modules():
  if m.__class__.__name__ in ['DepthConvBlock','StageBlock']:
   m.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),m)
   if hasattr(m,'proxy'):m.proxy=None
 if hasattr(g.decoder,'forward_torch'):g.decoder.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),g.decoder)
 assert not any('tcg' in n.lower() or 'lora' in n.lower() for n,p in pm.named_parameters())
 pm.eval().requires_grad_(False);im.eval().requires_grad_(False);setup_torch();return rt,im,pm
