"""Read-only stratum load and FD feature derivative helpers; no coefficient calibration entrypoint."""
from io32 import *
import argparse,traceback,resource,math

def get_stratum(key,device):
 import torch
 from distribution32 import to
 item=next(r for r in load(ROOT/'reference_stats/index.json')['entries'] if r['stratum']==key);assert sha(item['path'])==item['sha256'];s=torch.load(item['path'],map_location='cpu',weights_only=True)
 return to(s['seed'],device),to(s['reference'],device)
def feature_derivatives(z,old,ref,objectives=('FD',)):
 import torch
 from distribution32 import losses
 leaf=z.to('cuda:0').double().requires_grad_(True);ls,candidate=losses(leaf,old,ref);gr={}
 for i,name in enumerate(objectives):
  g=torch.autograd.grad(ls[name],leaf,retain_graph=i<len(objectives)-1)[0];assert g.isfinite().all() and g.norm()>0;gr[name]=g.detach()
 return gr,{k:float(v.detach()) for k,v in ls.items()},candidate
