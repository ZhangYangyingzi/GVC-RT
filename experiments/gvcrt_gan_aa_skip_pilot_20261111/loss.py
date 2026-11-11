"""Audited H0 objective, real per-domain frame counts and four-frame TBPTT."""
from common_s2 import *
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from experiment_model import reconstruct,detach_history
from model_context import assert_reset
from temporal import temporal_errors


def norm(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values if v is not None))


def targets(data):return list(range(49,64)) if data['info']['frames']==64 else list(range(1,7))


def dot(a,b):
    return sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)


def backward_window(c,data,batch=4,d=None,adv_weight=0.,mode='total',diagnostics=False):
    from gan import adv,g_mode
    assert data['info']['training'] and data['info']['frames'] in [7,64]
    assert_reset(c);device=c['device'];indices=targets(data);n=len(indices)
    valid_count=sum(bool(data['GT_valid'][i].any()) for i in indices)
    def gt(i):return data['GT'][i].permute(2,0,1)[None].to(device).float()/255
    previous=c['core'].unit(data['original'][0].to(device))
    with torch.no_grad():
        for i in range(1,indices[0]):previous=c['core'].unit(reconstruct(c,data['features'][i],data['info']['actual_qps'][i]))
    trace=[];objective=0.;fake_frames=[];diag={'spatial':None,'temporal':None,'adversarial':None}
    if d is not None:g_mode(d)
    for start in range(0,n,4):
        detach_history(c);previous=previous.detach();segment=[];components={'spatial':[],'temporal':[],'adversarial':[]}
        for i in indices[start:start+4]:
            out=c['core'].unit(reconstruct(c,data['features'][i],data['info']['actual_qps'][i],retain=True))
            g=gt(i);zero=out.new_zeros(())
            lp=ds=rgb=motion=rr=ss=zero
            if mode!='adv':
                lp=checkpoint(lambda x,y:c['quality'][0](x,y,normalize=True).mean(),out,g,use_reentrant=False)
                ds=checkpoint(lambda x,y:c['quality'][1](x,y,require_grad=True).mean(),out,g,use_reentrant=False)
                rgb=F.l1_loss(out,g)
                target=data['GT_forward'][i].to(device)
                def raft_motion(prev,cur,target=target):
                    a,b=c['transform'](prev,cur)
                    return F.l1_loss(c['raft'](a,b)[-1],target)
                motion=checkpoint(raft_motion,previous,out,use_reentrant=False)
                rr,ss,vf=temporal_errors(out,previous,g,gt(i-1),data['GT_backward'][i].to(device),data['GT_valid'][i].to(device))
            spatial=(lp+.5*ds+.1383589248954108*rgb)/(n*batch)
            temporal=.05*motion/(n*batch)
            if rr is not None and mode!='adv':temporal=temporal+LAMBDA_TEMP*(rr+.5*ss)/(valid_count*batch)
            adversarial=zero if d is None else adv(d,out)/(n*batch)
            if mode=='adv':value=adversarial
            elif mode=='base':value=spatial+temporal
            else:value=spatial+temporal+adv_weight*adversarial
            assert torch.isfinite(value),('NONFINITE_LOSS',i,mode)
            segment.append(value);components['spatial'].append(spatial);components['temporal'].append(temporal);components['adversarial'].append(adv_weight*adversarial)
            trace.append(dict(frame=i,LPIPS_Alex=float(lp.detach()),DISTS=float(ds.detach()),RGB_L1=float(rgb.detach()),RAFT_motion=float(motion.detach()),RGB_MCResidual=None if rr is None else float(rr.detach()),Sobel_MCResidual=None if ss is None else float(ss.detach()),G_adv=float(adversarial.detach()*n*batch)))
            if d is not None:fake_frames.append((out.detach().cpu(),g.detach().cpu(),1/(n*batch)))
            previous=out
        total=sum(segment);objective+=float(total.detach())
        if diagnostics:
            for name,values in components.items():
                val=sum(values)
                if val.requires_grad:
                    grads=torch.autograd.grad(val,c['params'],retain_graph=True,allow_unused=True)
                    if diag[name] is None:diag[name]=[None if v is None else v.detach().cpu().clone() for v in grads]
                    else:
                        for k,v in enumerate(grads):
                            if v is not None:
                                if diag[name][k] is None:diag[name][k]=v.detach().cpu().clone()
                                else:diag[name][k].add_(v.detach().cpu())
        total.backward();detach_history(c);previous=previous.detach()
        if c['raft'] is not None:c['raft'].corr_block.corr_pyramid=[x.detach() for x in c['raft'].corr_block.corr_pyramid]
        del segment,components,total,out,lp,ds,rgb,motion,rr,ss,value,adversarial
    assert_reset(c)
    return dict(objective_contribution=objective,valid_transitions=valid_count,supervised_frames=n,trace=trace),fake_frames,diag
