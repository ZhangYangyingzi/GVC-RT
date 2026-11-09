"""Segment VJPs retain four-frame graphs; no across-segment BPTT.

Spatial/RAFT terms use15 supervised frames; temporal terms use nonempty-mask
frames. The caller averages four spatial/RAFT windows and available temporal
windows. 4096 is not a coefficient of this objective.
"""
from common_s2 import *
from experiment_model import reconstruct,detach_history
from temporal import temporal_errors
import torch
import torch.nn.functional as F

TERMS=('LPIPS','DISTS_weighted','RGB_L1_weighted','RAFT_weighted','temp_RGB_weighted','temp_Sobel_weighted')

def gradient_norm(values):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in values))

def grouped_norms(c,gradients):
    groups={k:[] for k in ['TCG','LoRA','Bridge_mlp3']}
    for (name,p),g in zip(c['named'],gradients):
        k='LoRA' if '.lora.' in name else 'TCG' if '.decoder.tcg_' in name else 'Bridge_mlp3'
        groups[k].append(g)
    return {k:gradient_norm(v) for k,v in groups.items()}

def window_gradients(c,data,lambda_temp,calibration=False,return_components=False):
    from model_context import assert_reset
    assert data['info']['training'] and data['info']['frames']==64
    device=c['device'];params=c['params'];gen=c['models']['generator'];assert_reset(c)
    component={k:[torch.zeros_like(p) for p in params] for k in TERMS};trace=[];temporal_valid_frames=0;temporal_empty_frames=0;temporal_valid_pixels=0
    def gt(i):return data['GT'][i].permute(2,0,1)[None].to(device).float()/255
    def alignment(i):return __import__('temporal').alignment_field(data,i,device) if c['alignment_enabled'] else None
    temporal_window_frame_count=sum(int(data['GT_valid'][i].sum())>0 for i in range(49,64))
    previous=c['core'].unit(data['original'][0].to(device))
    with torch.no_grad():
        for i in range(1,49):previous=c['core'].unit(reconstruct(c,data['features'][i].to(device),data['info']['actual_qps'][i],alignment(i),retain=False))
    previous=previous.detach()
    for first,last in [(49,52),(53,56),(57,60),(61,63)]:
        detach_history(c);previous=previous.detach();terms={k:[] for k in TERMS}
        for i in range(first,last+1):
            target=gt(i);before=gt(i-1)
            out=c['core'].unit(reconstruct(c,data['features'][i].to(device),data['info']['actual_qps'][i],alignment(i),retain=True))
            lp=c['quality'][0](out,target,normalize=True).mean()
            ds=c['quality'][1](out,target,require_grad=True).mean()
            rgb=F.l1_loss(out,target)
            back=data['GT_backward'][i].to(device);valid=data['GT_valid'][i].to(device)
            tr,te,valid_fraction=temporal_errors(out,previous,target,before,back,valid)
            valid_pixels=int(valid.sum().item())
            temporal_valid=tr is not None
            temporal_valid_frames+=int(temporal_valid);temporal_empty_frames+=int(not temporal_valid);temporal_valid_pixels+=valid_pixels
            if calibration:motion=out.new_zeros(())
            else:
                a,b=c['transform'](previous,out)
                predicted=c['raft'](a,b)[-1]
                motion=F.l1_loss(predicted,data['GT_forward'][i].to(device))
            values=dict(LPIPS=lp,DISTS_weighted=.5*ds,RGB_L1_weighted=.1383589248954108*rgb,RAFT_weighted=.05*motion,
                temp_RGB_weighted=None if tr is None else lambda_temp*tr,temp_Sobel_weighted=None if te is None else lambda_temp*.5*te)
            for k,v in values.items():
                if v is None:continue
                assert torch.isfinite(v),('NONFINITE_LOSS',k,i)
                terms[k].append(v)
            trace.append(dict(frame=i,actual_QP=data['info']['actual_qps'][i],segment_start=first,segment_end=last,valid_fraction=valid_fraction,
                temporal_valid=temporal_valid,temporal_valid_pixels=valid_pixels,LPIPS=float(lp.detach()),DISTS=float(ds.detach()),RGB_L1=float(rgb.detach()),RAFT_motion=float(motion.detach()),RGB_temporal=None if tr is None else float(tr.detach()),Sobel_temporal=None if te is None else float(te.detach()),
                **{k:float(v.detach()) for k,v in values.items() if k!='LPIPS' and v is not None}))
            previous=out
        active=[k for k in TERMS if not (calibration and k=='RAFT_weighted') and terms[k]]
        for j,k in enumerate(active):
            denominator=temporal_window_frame_count if k.startswith('temp_') else 15
            val=sum(terms[k])/denominator
            gradients=torch.autograd.grad(val,params,allow_unused=True,retain_graph=j<len(active)-1)
            for dst,g in zip(component[k],gradients):
                if g is not None:
                    assert torch.isfinite(g).all(),('NONFINITE_COMPONENT_GRADIENT',k)
                    dst.add_(g.detach().float())
        detach_history(c);previous=previous.detach()
        # RAFT keeps a runtime correlation pyramid. It is not persistent model
        # state and must not hold a past segment's autograd graph.
        if not calibration:
            c['raft'].corr_block.corr_pyramid=[x.detach() for x in c['raft'].corr_block.corr_pyramid]
            del predicted,a,b
        del terms,values,val,gradients,out,lp,ds,rgb,tr,te,motion
    total=[sum(component[k][i] for k in TERMS) for i in range(len(params))]
    spatial=[sum(component[k][i] for k in TERMS[:3]) for i in range(len(params))]
    temporal=[component['temp_RGB_weighted'][i]+component['temp_Sobel_weighted'][i] for i in range(len(params))]
    nsp=gradient_norm(spatial);nt=gradient_norm(temporal)
    cosine=sum(float(a.double().mul(b.double()).sum()) for a,b in zip(spatial,temporal))/(nsp*nt) if nsp*nt>0 else None
    def available_mean(key):
        values=[t[key] for t in trace if t.get(key) is not None]
        return sum(values)/len(values) if values else None
    mean_components={k:available_mean(k) for k in TERMS}
    summary=dict(losses={k:available_mean(k) for k in ['LPIPS','DISTS','RGB_L1','RAFT_motion','RGB_temporal','Sobel_temporal']},
        total_loss=None if temporal_window_frame_count==0 else sum(v for v in mean_components.values() if v is not None),gradient_by_component={k:grouped_norms(c,g) for k,g in component.items()},
        spatial_gradient_L2=nsp,temporal_gradient_L2=nt,spatial_temporal_cosine=cosine,gradient_L2=gradient_norm(total),frames_supervised=15,
        frames_temporal_valid=temporal_valid_frames,frames_temporal_empty=temporal_empty_frames,valid_temporal_pixels=temporal_valid_pixels,temporal_objective_available=temporal_window_frame_count>0,
        segment_lengths=[4,4,4,3],codec_candidate_reference_updates=0,rate_bpp_monitoring=data['info']['bpp'],proxy_L1_monitoring=0.,rate_gradient='constant/no candidate codec path',proxy_gradient='Identity zero',raw_trace=trace)
    assert summary['total_loss'] is None or math.isfinite(summary['total_loss'])
    assert summary['gradient_L2']>0
    assert_reset(c)
    return (total,summary,component) if return_components else (total,summary)
