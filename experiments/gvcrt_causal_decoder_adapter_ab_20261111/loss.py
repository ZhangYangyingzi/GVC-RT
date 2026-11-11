"""Spatial and GT-estimated-flow MC residual objectives; no learned-flow loss."""
from common_s2 import *
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from adapter import decode_head
from temporal import temporal_errors,sobel

SEGMENTS=((49,52),(53,56),(57,60),(61,63))


def norm(values):
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values if v is not None))


def load_training_context(device):
    from adapter import CausalAdapter
    from convlora import ParallelConvLoRA
    import lpips
    from DISTS_pytorch import DISTS
    torch.set_num_threads(2)
    # CUDA grid_sample backward is necessarily nondeterministic. Fixed seeds and
    # identical data/order are retained; use the inherited warn-only policy.
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed(TRAIN_SEED)
    a=torch.load(ROOT/'checkpoints/common_adapter_step0.pt',map_location='cpu',weights_only=True)
    branch=CausalAdapter().to(device);branch.load_state_dict(a['adapter'],strict=True)
    head=ParallelConvLoRA(torch.nn.Conv2d(320,192,3,padding=1)).to(device).float()
    head.load_state_dict(torch.load(ROOT/'checkpoints/frozen_FP32_head.pt',map_location='cpu',weights_only=True)['state'],strict=True)
    head.inference_gain=.5;head.eval().requires_grad_(False)
    quality=(lpips.LPIPS(net='alex',version='0.1',verbose=False).to(device).eval().requires_grad_(False),
             DISTS().to(device).eval().requires_grad_(False))
    return dict(adapter=branch,head=head,quality=quality,device=device,params=list(branch.parameters()))


def frame(c,data,i):
    h=data['h'][i-48:i-45]  # i-2..i, with h tensor starting at frame46.
    assert h.shape[0]==3,(i,h.shape)
    three=[x[None].to(c['device']) for x in h]
    return (decode_head(c['head'],c['adapter'].from_three(three,torch.float32)).float()+1)/2


def window(c,data,batch=1,lambda_temp=0.,mode='total',backward=True,gradient_targets=None):
    device=c['device'];valid_count=sum(bool(x.any()) for x in data['GT_valid'])
    assert valid_count>0
    gradients=None;trace=[];objective=0.
    def gt(i):return data['GT'][i-48].permute(2,0,1)[None].to(device).float()/255
    for first,last in SEGMENTS:
        # The boundary frame is always rendered by the current adapter, then detached.
        with torch.no_grad():previous=frame(c,data,first-1)
        terms=[]
        for i in range(first,last+1):
            out=frame(c,data,i);target=gt(i)
            lp=ds=rgb=out.new_zeros(())
            if mode!='temporal':
                lp=checkpoint(lambda x,y:c['quality'][0](x,y,normalize=True).mean(),out,target,use_reentrant=False)
                ds=checkpoint(lambda x,y:c['quality'][1](x,y,require_grad=True).mean(),out,target,use_reentrant=False)
                rgb=F.l1_loss(out,target)
            spatial=lp+.5*ds+.1383589248954108*rgb
            rr,ss,vf=temporal_errors(out,previous,target,gt(i-1),
                data['GT_backward'][i-49].to(device),data['GT_valid'][i-49].to(device))
            temporal=out.new_zeros(()) if rr is None else rr+.5*ss
            if mode=='spatial':value=spatial/(15*batch)
            elif mode=='temporal':value=temporal/(valid_count*batch)
            else:value=spatial/(15*batch)+lambda_temp*temporal/(valid_count*batch)
            assert torch.isfinite(value),('NONFINITE_OBJECTIVE',i)
            terms.append(value)
            edge=sobel(out.detach());edge_gt=sobel(target)
            trace.append(dict(frame=i,LPIPS_Alex=float(lp.detach()),DISTS=float(ds.detach()),RGB_L1=float(rgb.detach()),
                RGB_MCResidual=None if rr is None else float(rr.detach()),Sobel_MCResidual=None if ss is None else float(ss.detach()),
                edge_strength=float(edge.abs().mean()),GT_edge_strength=float(edge_gt.abs().mean()),
                local16_energy_abs_bias=float((F.avg_pool2d(edge.abs(),16,16)-F.avg_pool2d(edge_gt.abs(),16,16)).abs().mean()),
                valid_fraction=vf,segment=[first,last]))
            previous=out
        value=sum(terms);objective+=float(value.detach())
        if gradient_targets is not None:
            gs=torch.autograd.grad(value,gradient_targets,allow_unused=True)
            gs=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(gradient_targets,gs)]
            gradients=gs if gradients is None else [a+b for a,b in zip(gradients,gs)]
        elif backward:value.backward()
        del value,terms,previous,out
    return dict(objective=objective,trace=trace,valid_pairs=valid_count),gradients
