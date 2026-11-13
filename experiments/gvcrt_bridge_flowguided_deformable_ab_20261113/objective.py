from util import *
import torch
import torch.nn.functional as F
from temporal import temporal_errors,sobel
LAMBDA_TEMP=4.63349060748

def quality_models():
 import lpips
 sys.path.append(str(ROOT/'vendor/runtime_dependencies'))
 from DISTS_pytorch import DISTS
 return lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False),DISTS().cuda().eval().requires_grad_(False)
def spatial(q,y,x):return q[0](y,x,normalize=True).mean()+.5*q[1](y,x,require_grad=True).mean()+.1383589248954108*F.l1_loss(y,x)
def temporal(y,x,flow,valid):
 vals=[]
 for i in range(1,len(y)):
  a,b,_=temporal_errors(y[i],y[i-1],x[i],x[i-1],flow[i].cuda().float(),valid[i].cuda().bool())
  if a is not None:vals.append(a+.5*b)
 return torch.stack(vals).mean() if vals else sum(t.sum()*0 for t in y)
