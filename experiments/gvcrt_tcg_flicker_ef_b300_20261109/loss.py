"""Four-frame summed-objective backward; one parameter-gradient set per rank."""
from common_s2 import *
from experiment_model import reconstruct,detach_history
from model_context import assert_reset
from temporal import temporal_errors,sobel
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def detail_error(x,y):
    return torch.stack([(sobel(x if k==1 else F.avg_pool2d(x,k,k))-sobel(y if k==1 else F.avg_pool2d(y,k,k))).abs().mean() for k in [1,2,4]]).mean()

def valid_frames(data):return sum(bool(data['GT_valid'][i].any()) for i in range(49,64))

def norm(params):return math.sqrt(sum(float(p.detach().double().square().sum()) for p in params if p is not None))

def backward_window(c,data,nonempty_windows,lambda_detail=0.,mode='total',reference=False):
    assert data['info']['training'] and data['info']['frames']==64
    device=c['device'];assert_reset(c);nv=valid_frames(data);records=[];objective_value=0.
    def gt(i):return data['GT'][i].permute(2,0,1)[None].to(device).float()/255
    previous=c['core'].unit(data['original'][0].to(device))
    with torch.no_grad():
        for i in range(1,49):previous=c['core'].unit(reconstruct(c,data['features'][i].to(device),data['info']['actual_qps'][i]))
    previous=previous.detach()
    for first,last in [(49,52),(53,56),(57,60),(61,63)]:
        detach_history(c);previous=previous.detach();segment=[]
        for i in range(first,last+1):
            target=gt(i);before=gt(i-1)
            out=c['core'].unit(reconstruct(c,data['features'][i].to(device),data['info']['actual_qps'][i],retain=True))
            lp=checkpoint(lambda a,b:c['quality'][0](a,b,normalize=True).mean(),out,target,use_reentrant=False)
            ds=checkpoint(lambda a,b:c['quality'][1](a,b,require_grad=True).mean(),out,target,use_reentrant=False)
            rgb=F.l1_loss(out,target);spatial=lp+.5*ds+.1383589248954108*rgb
            detail=detail_error(out,target)
            tr,te,v=temporal_errors(out,previous,target,before,data['GT_backward'][i].to(device),data['GT_valid'][i].to(device))
            if mode=='total':
                def motion_fn(prev,cur):
                    a,b=c['transform'](prev,cur)
                    return F.l1_loss(c['raft'](a,b)[-1],data['GT_forward'][i].to(device))
                # Bind frame data; backward recomputation must use this frame.
                flow_target=data['GT_forward'][i].to(device)
                def motion_fn(prev,cur,flow_target=flow_target):
                    a,b=c['transform'](prev,cur)
                    return F.l1_loss(c['raft'](a,b)[-1],flow_target)
                motion=checkpoint(motion_fn,previous,out,use_reentrant=False)
                value=(spatial+.05*motion+lambda_detail*detail)/(15*4)
                if tr is not None:value=value+LAMBDA_TEMP*(tr+.5*te)/(nv*nonempty_windows)
            elif mode=='spatial':value=spatial/15;motion=out.new_zeros(())
            elif mode=='detail':value=detail/15;motion=out.new_zeros(())
            else:raise ValueError(mode)
            assert torch.isfinite(value),('NONFINITE_LOSS',i,mode)
            segment.append(value)
            records.append(dict(frame=i,LPIPS=float(lp.detach()),DISTS=float(ds.detach()),RGB_L1=float(rgb.detach()),RAFT_motion=float(motion.detach()),detail=float(detail.detach()),RGB_temporal=None if tr is None else float(tr.detach()),Sobel_temporal=None if te is None else float(te.detach()),valid_fraction=v))
            previous=out
        total=sum(segment);objective_value+=float(total.detach())
        if reference:
            gradients=torch.autograd.grad(total,c['params'],allow_unused=True)
            for p,g in zip(c['params'],gradients):
                if g is not None:
                    if p.grad is None:p.grad=g.detach()
                    else:p.grad.add_(g.detach())
        else:total.backward()
        detach_history(c);previous=previous.detach()
        if mode=='total':c['raft'].corr_block.corr_pyramid=[t.detach() for t in c['raft'].corr_block.corr_pyramid]
        del total,segment,out,lp,ds,spatial,detail,rgb,tr,te,motion,value
    assert_reset(c)
    return dict(objective_contribution=objective_value,frames_temporal_valid=nv,trace=records)
