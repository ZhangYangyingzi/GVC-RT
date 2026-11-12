from util import *
import torch
import torch.nn.functional as F
from model import sample,grid,bounds,resize_flow
from temporal import temporal_errors,sobel

def quality_models():
 import lpips
 sys.path.append(str(ROOT/'vendor/runtime_dependencies'))
 from DISTS_pytorch import DISTS
 return lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False),DISTS().cuda().eval().requires_grad_(False)
def spatial(q,y,x):return q[0](y,x,normalize=True).mean()+.5*q[1](y,x,require_grad=True).mean()+.1383589248954108*F.l1_loss(y,x)
def temporal(y,x,flow,valid):
 ts=[]
 for i in range(1,len(y)):
  a,b,_=temporal_errors(y[i],y[i-1],x[i],x[i-1],flow[i].cuda().float(),valid[i].cuda().bool())
  if a is not None:ts.append(a+.5*b)
 assert ts,'NO_VALID_TEMPORAL_PAIR'
 return torch.stack(ts).mean()
def align_loss(flows,flow,valid):
 loss=flow.new_zeros(());epe={}
 for scale,weight in [(8,1),(16,.5),(32,.25)]:
  u=flows[scale];target=resize_flow(flow,u.shape[-2:]);mask=F.interpolate(valid.float(),size=u.shape[-2:],mode='bilinear',align_corners=False)>.999
  count=mask.sum()
  if count==0:
   epe[scale]=None;loss=loss+u.sum()*0;continue
  d=u-target;loss=loss+weight*((d.square()+1e-6).sqrt()*mask).sum()/(2*count)
  epe[scale]=float((d.detach().square().sum(1,keepdim=True).sqrt()*mask).sum()/count)
 return loss,epe
