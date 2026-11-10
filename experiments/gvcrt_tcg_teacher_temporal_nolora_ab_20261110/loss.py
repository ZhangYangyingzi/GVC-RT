"""Content/output/Teacher-difference objectives with exactly-once global means."""
from common_s2 import *
from experiment_model import reconstruct,detach_history
from model_context import assert_reset
from temporal import backwarp,temporal_errors
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def norm(values):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in values if x is not None))

def latent_field(data,i,device):
    h,w=data['info']['native_shape'];lh,lw=h//16,w//16
    assert (h,w)==(256,256) and (lh,lw)==(16,16)
    flow=F.interpolate(data['GT_backward'][i].to(device).float(),size=(lh,lw),mode='bilinear',align_corners=False)
    flow=(flow*flow.new_tensor([lw/w,lh/h]).reshape(1,2,1,1)).detach()
    valid=-F.adaptive_max_pool2d(-data['GT_valid'][i].to(device).float(),(lh,lw))>.5
    yy,xx=torch.meshgrid(torch.arange(lh,device=device),torch.arange(lw,device=device),indexing='ij')
    px=xx[None]+flow[:,0];py=yy[None]+flow[:,1]
    valid=(valid&((px>=0)&(px<=lw-1)&(py>=0)&(py<=lh-1)).unsqueeze(1)).detach()
    return flow,valid

def window_masks(data,device):
    fields={i:latent_field(data,i,device) for i in range(49,64)}
    return dict(output_frames=sum(bool(data['GT_valid'][i].any()) for i in range(49,64)),latent_frames=sum(bool(fields[i][1].any()) for i in fields),fields=fields)

def latent_errors(s,prev,t,pt,flow,valid):
    count=int(valid.sum())
    if not count:return None,None,None
    warped,bounds=backwarp(prev,flow.detach());wt,_=backwarp(pt.detach(),flow.detach());mask=(valid&bounds).detach()
    ds=s-warped;dt=t.detach()-wt
    den=mask.sum()*s.shape[1]
    return ((ds-dt).abs()*mask).sum()/den,(ds.abs()*mask).sum()/den,(dt.abs()*mask).sum()/den

def backward_window(c,data,teacher,counts,coefficients,mode='total',reference=False,masks=None):
    device=c['device'];assert data['info']['training'] and data['info']['frames']==64
    assert teacher['Teacher_SHA256']==TEACHER_SHA and teacher['codec_identity']==data['info']['cache_identity']
    assert_reset(c);masks=window_masks(data,device) if masks is None else masks;nv=masks['output_frames'];nl=masks['latent_frames'];trace=[];objective=0.
    def gt(i):return data['GT'][i].permute(2,0,1)[None].to(device).float()/255
    def target(i):return teacher['targets'][i].to(device).detach()
    previous=c['core'].unit(data['original'][0].to(device));previous_z=None
    with torch.no_grad():
        for i in range(1,49):
            raw,previous_z=reconstruct(c,data['features'][i],data['info']['actual_qps'][i],latent=True);previous=c['core'].unit(raw)
    previous=previous.detach();previous_z=previous_z.detach()
    for first,last in [(49,52),(53,56),(57,60),(61,63)]:
        detach_history(c);previous=previous.detach();previous_z=previous_z.detach();segment=[]
        for i in range(first,last+1):
            out,z=reconstruct(c,data['features'][i],data['info']['actual_qps'][i],retain=True,latent=True);out=c['core'].unit(out);t=target(i);pt=target(i-1);g=gt(i)
            assert z.shape==t.shape==(1,18,16,16)
            cosine=(1-F.cosine_similarity(z.float(),t,dim=1,eps=1e-8)).mean()
            need_spatial=mode in ['total','content','LPIPS_VGG','DISTS']
            lp=checkpoint(lambda x,y:c['quality'][0](x,y,normalize=True).mean(),out,g,use_reentrant=False) if need_spatial else out.new_zeros(())
            ds=checkpoint(lambda x,y:c['quality'][1](x,y,require_grad=True).mean(),out,g,use_reentrant=False) if need_spatial else out.new_zeros(())
            rgb=F.l1_loss(out,g);content=lp+.5*ds+.1383589248954108*rgb+.2*cosine
            tr,te,valid_fraction=temporal_errors(out,previous,g,gt(i-1),data['GT_backward'][i].to(device),data['GT_valid'][i].to(device))
            flow,valid=masks['fields'][i];lt,sd,td=latent_errors(z,previous_z,t,pt,flow,valid)
            need_motion=mode in ['total','RAFT_motion'];motion=out.new_zeros(())
            if need_motion:
                flow_target=data['GT_forward'][i].to(device)
                def motion_fn(prev,cur,flow_target=flow_target):
                    a,b=c['transform'](prev,cur);return F.l1_loss(c['raft'](a,b)[-1],flow_target)
                motion=checkpoint(motion_fn,previous,out,use_reentrant=False)
            if mode=='total':
                value=(content+.05*motion)/(15*4)
                if tr is not None:value=value+coefficients['lambda_out']*(tr+.5*te)/(nv*counts['output'])
                if c['experiment_group']==GROUPS[1] and lt is not None:value=value+coefficients['lambda_latent']*lt/(nl*counts['latent'])
            elif mode=='content':value=content/15
            elif mode=='output_temporal':value=None if tr is None else (tr+.5*te)/nv
            elif mode=='latent_temporal':value=None if lt is None else lt/nl
            elif mode=='cosine':value=cosine/15
            elif mode=='LPIPS_VGG':value=lp/15
            elif mode=='DISTS':value=ds/15
            elif mode=='RAFT_motion':value=motion/15
            else:raise ValueError(mode)
            if value is not None:
                assert torch.isfinite(value),('NONFINITE_LOSS',mode,i);segment.append(value)
            trace.append(dict(frame=i,segment=[first,last],LPIPS_VGG=float(lp.detach()) if need_spatial else None,DISTS=float(ds.detach()) if need_spatial else None,RGB_L1=float(rgb.detach()),cosine=float(cosine.detach()),RAFT_motion=float(motion.detach()) if need_motion else None,RGB_MCResidual=None if tr is None else float(tr.detach()),Sobel_MCResidual=None if te is None else float(te.detach()),latent_temporal=None if lt is None else float(lt.detach()),student_latent_MC_change=None if sd is None else float(sd.detach()),Teacher_latent_MC_change=None if td is None else float(td.detach()),latent_valid_pixels=int(valid.sum()),RGB_valid_fraction=valid_fraction,student_RMS=float(z.detach().square().mean().sqrt()),Teacher_RMS=float(t.square().mean().sqrt())))
            previous,previous_z=out,z
        if segment:
            total=sum(segment);objective+=float(total.detach())
            if reference:
                gradients=torch.autograd.grad(total,c['params'],allow_unused=True)
                for p,g in zip(c['params'],gradients):
                    if g is not None:
                        if p.grad is None:p.grad=g.detach()
                        else:p.grad.add_(g.detach())
            else:total.backward()
        detach_history(c);previous=previous.detach();previous_z=previous_z.detach()
        if need_motion:c['raft'].corr_block.corr_pyramid=[x.detach() for x in c['raft'].corr_block.corr_pyramid]
        del segment,out,z,t,pt,lp,ds,content,cosine,rgb,tr,te,lt,sd,td,motion,value
    assert_reset(c)
    return dict(objective_contribution=objective,output_valid_frames=nv,latent_valid_frames=nl,output_missing_frames=15-nv,latent_missing_frames=15-nl,trace=trace,normalization='per-frame valid pixel/channel mean; per-window valid-frame mean; global4 content/RAFT and nonempty-window temporal means')
