"""Training-only temporal gradient and frozen-target diagnostics."""
from common_s2 import *
def main(rank=0):
    env();p=ROOT/'audits/preflight.json'
    if p.exists():assert load(p)['status']=='PASS';return
    import torch
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    from cache_teacher import paths
    from loss import latent_field,latent_errors
    from temporal import backwarp
    c=construct(GROUPS[1],True,rank);records=[]
    for row in plan_rows()[:8]:
        _,cp,_=cache_paths(row,row['external_qp'],True);_,tp,_=paths(row,row['external_qp']);d=torch.load(cp,map_location='cpu',weights_only=True,mmap=True);t=torch.load(tp,map_location='cpu',weights_only=True,mmap=True)
        assert_reset(c)
        with torch.no_grad():
            for i in range(1,50):reconstruct(c,d['features'][i],d['info']['actual_qps'][i])
        s=reconstruct(c,d['features'][50],d['info']['actual_qps'][50],True,True)[1]
        prev=s;s=reconstruct(c,d['features'][51],d['info']['actual_qps'][51],True,True)[1]
        fl,mask=latent_field(d,51,c['device']);tt=t['targets'][51].to(c['device']);pt=t['targets'][50].to(c['device']);error,_,_=latent_errors(s,prev,tt,pt,fl,mask);assert error is not None
        gs,gp=torch.autograd.grad(error,[s,prev],retain_graph=False);assert gs.abs().sum()>0 and gp.abs().sum()>0
        values=[]
        with torch.no_grad():
            for i in range(49,64):
                fl,valid=latent_field(d,i,c['device']);a=t['targets'][i].to(c['device']);b=t['targets'][i-1].to(c['device']);wb,_=backwarp(b,fl);err=(a-wb).abs().mean(1,keepdim=True)
                # RGB-equivalent flow length; thresholds only label diagnostic regions.
                magnitude=(fl*16).square().sum(1,keepdim=True).sqrt()
                item=dict(frame=i,valid_pixels=int(valid.sum()))
                for name,region in [('stationary',magnitude<.1),('moving',magnitude>=1.)]:
                    m=valid&region;n=int(m.sum());item[name+'_pixels']=n;item[name+'_Teacher_MC_change']=float(err[m].mean()) if n else None
                values.append(item)
        records.append(dict(sample_key=row['sample_key'],QP=row['external_qp'],previous_student_gradient_norm=float(gp.norm()),current_student_gradient_norm=float(gs.norm()),Teacher_requires_grad=tt.requires_grad,flow_requires_grad=fl.requires_grad,frames=values));assert_reset(c)
    dump(p,dict(status='PASS',windows=records,training_windows_only=True,temporal_stability_proven=False,latent_direct_TCG_gradient=False,gradient_components_from='gradient_calibration.json',geometry='matching RGB crop, no target interpolation/channel projection',time=time.time()))
if __name__=='__main__':main()
