from util import *
import torch
from model import *

def run():
 torch.set_num_threads(2);seed();stats=dict(f_rms=torch.ones(256),l_rms=torch.ones(18),qp_mean=5.,qp_std=3.);results={}
 for group in ['S_current','U_unaligned','T_aligned']:
  m=AlignedLatent(stats,group);l=torch.randn(1,18,16,16);f=torch.randn(1,256,32,32);m(l,f,5,'v',1);x=torch.randn(1,18,16,20);ff=torch.randn(1,256,32,40);out,a=m(x,ff,7,'v',2);assert torch.equal(out,x) and not a['q'].any(),'RESOLUTION_RESET'
  m(l,f,5,'v',3);out,a=m(l,f,6,'v',4,reset=True);assert torch.equal(out,l) and not a['q'].any(),'GOP_RESET'
  m.reset();captured=[];hook=m.fusion.reliability[0].register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach()))
  m(l,f,5,'v',1);m(l,f,7,'v',2);hook.remove();z=captured[-1];assert z[:,56].eq((7-5)/3).all();assert z[:,57].eq(((7 if group=='S_current' else 5)-5)/3).all();results[group]=dict(resolution_reset=True,explicit_I_GOP_reset=True,actual_current_previous_QP_broadcast=True)
 feat=torch.randn(1,32,8,12,requires_grad=True);u=torch.ones(1,2,8,12)*10000;c,e,v=cost_volume(feat,feat,u);(c.sum()+e.sum()).backward();assert torch.isfinite(feat.grad).all()
 dump(ROOT/'audits/boundary_checks.json',dict(status='PASS',groups=results,all_invalid_backward_finite=True,coordinate_protocol='feature grid units, current->past, centers0, align_corners=False; RGB origins3.5/7.5/15.5 for8/16/32 grids'))
 print('boundary/QP/empty-candidate backward checks PASS')
if __name__=='__main__':run()
