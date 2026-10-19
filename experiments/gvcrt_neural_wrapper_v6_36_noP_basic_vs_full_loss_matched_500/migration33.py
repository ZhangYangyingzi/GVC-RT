"""Recursive dtype-aware state hashes; no historical optimizer loading."""
from io32 import *

def state_digest(x):
 import torch
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,torch.Tensor):
   h.update(('tensor:'+str(v.dtype)+str(tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k in v:visit(k);visit(v[k])
  elif isinstance(v,(tuple,list)):
   h.update(type(v).__name__.encode())
   for e in v:visit(e)
  else:h.update((type(v).__name__+':'+repr(v)).encode())
 visit(x);return h.hexdigest()

