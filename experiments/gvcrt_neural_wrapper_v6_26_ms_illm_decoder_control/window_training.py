"""Identical native STE path; 15 supervised P frames and detached references."""
import math
from io21 import *
def replay_rows(branch):return load(ROOT/'plans/shared.json')['updates']
def replay(row,branch,device):
    import cv2,numpy as np,torch
    from PIL import Image
    cv2.setNumThreads(1)
    d=row['domain'];suffix='';videos=load(ROOT/f'train_manifest_{d}{suffix}.json')['videos'];v=next(v for v in videos if v['filename' if d=='ulong' else 'name']==row['video'])
    assert row['source_path']==v['path' if d=='ulong' else 'source_path'];assert row['source_sha256']==v['sha256' if d=='ulong' else 'source_sha256']
    x,y=row['crop_x'],row['crop_y'];arrays=[];full=[];indices=row['canonical_indices'];assert indices==list(range(indices[0],indices[0]+64))
    if d=='ulong':
        cap=cv2.VideoCapture(v['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
        try:
            for i in indices:
                ok,bgr=cap.read();assert ok,(row['video'],i);a=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);full.append(hashlib.sha256(a.tobytes()).hexdigest());arrays.append(a[y:y+256,x:x+256].copy())
        finally:cap.release()
    else:
        for i in indices:
            p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:a=np.asarray(im,dtype=np.uint8)
            full.append(hashlib.sha256(a.tobytes()).hexdigest());arrays.append(a[y:y+256,x:x+256].copy())
    hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays];assert hashes==row['cropped_RGB_sha256'] and full==row['full_RGB_sha256'];assert len(arrays)==64
    return [torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).float()/255 for a in arrays]

def make_runner(v4,im,pm,models,quality,discriminator=None):
    import torch
    import torch.nn.functional as F
    common=module('rec22_common',V62/'fullqp_common.py');wrapper=models['wrapper'];params=[p for m in models.values() for p in m.parameters() if p.requires_grad]
    def run(frames,q,branch,details=True,rec_weight=None,mode='train',rec_scale=4096.,adv_weight=None,adv_scale=4096.):
        assert len(frames)==64 and branch in BRANCHES
        if rec_weight is None:rec_weight=0.1383589248954108
        requested_weight=adv_weight;adv_weight=0. if adv_weight is None else adv_weight
        cfg=load(ROOT/'config.json')
        if discriminator is not None:discriminator.g_mode()
        contrib={k:[torch.zeros_like(p) for p in params] for k in ('reconstruction','adversarial')};fakes=[];conditions=[];latent=[];den=15;beta=common.beta_q('schedule_s1p0',q);trace=[];warmup=[];capt=[];component_grads={name:[torch.zeros_like(p) for p in params] for name in ('base','adv')} if True else None
        assert all(p.dtype==torch.float32 and (p.grad is None or p.grad.dtype==torch.float32) for p in params)
        sums={k:0. for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','lambda_proxy_proxy_L1','reconstruction_L1','weighted_reconstruction_L1','projected_adv','weighted_projected_adv')}
        hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
        latent_hook=models['bridge'][0].register_forward_pre_hook(lambda _m,a:latent.append(a[0].detach()))
        def ref(rec):
            assert len(capt)==1
            with torch.no_grad():pm.add_ref_frame(capt.pop().detach(),(rec.detach()*2-1).half())
            assert not pm.dpb[0].feature.requires_grad and not pm.dpb[0].frame.requires_grad
        def add(grads,buffers,scale):
            for g,b in zip(grads,buffers):
                if g is not None:
                    value=g.detach().float()/scale;assert bool(torch.isfinite(value).all());b.add_(value)
        try:
            pm.clear_dpb();pm.set_curr_poc(0)
            with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),q)['x_hat'])
            for pos in range(1,49):
                with torch.no_grad():
                    latent.clear();qa=common.frame_qp(pm,q,pos);rec,rate,_=v4.joint_ste_forward(pm,wrapper(frames[pos]),qa);assert not rec.requires_grad;ref(rec)
                    warmup.append(dict(position=pos,actual_qp=qa,reference_age=pos,rec_sha256=hashlib.sha256(rec.cpu().contiguous().numpy().tobytes()).hexdigest(),feature_sha256=hashlib.sha256(pm.dpb[0].feature.cpu().contiguous().numpy().tobytes()).hexdigest(),no_grad=True,reference_detached=True))
            target=[common.frame_qp(pm,q,pos) for pos in range(49,64)]
            for pos,qa in zip(range(49,64),target):
                latent.clear();frame=frames[pos];proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
                lp=quality[0](rec,frame,normalize=True).mean();ds=quality[1](rec,frame,require_grad=True).mean();rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame);rl=F.l1_loss(rec,frame);dist=lp+.5*ds;base=dist+beta*rb+.01*pl
                assert len(latent)==1 and latent[0].shape[1]==1024
                condition=latent.pop().detach()
                if discriminator is not None and hasattr(discriminator,'set_target'):discriminator.set_target(frame)
                adv=discriminator.adversarial(rec,condition) if discriminator is not None else rec.new_zeros(())
                comp=dict(loss=base+rec_weight*rl+adv_weight*adv,projected_adv=adv,weighted_projected_adv=adv_weight*adv,LPIPS=lp,DISTS=ds,D=dist,rate_bpp=rb,beta_rate=beta*rb,proxy_L1=pl,lambda_proxy_proxy_L1=.01*pl,reconstruction_L1=rl,weighted_reconstruction_L1=rec_weight*rl)
                assert all(bool(torch.isfinite(t).all()) for t in comp.values())
                def restored(loss,scale,weight=1.):
                    gs=torch.autograd.grad(loss*(scale/den),params,retain_graph=True,allow_unused=True)
                    return [None if g is None else g.detach().float()/scale*weight for g in gs]
                rg=restored(rl,rec_scale,rec_weight)
                gp=torch.autograd.grad(base/den,params,retain_graph=True,allow_unused=True)
                add(gp,component_grads['base'],1.)
                for g,b in zip(rg,component_grads['base']):
                    if g is not None:b.add_(g)
                ga=restored(adv,adv_scale) if discriminator is not None else [None]*len(params)
                add(ga,component_grads['adv'],1.)
                for g,b in zip(rg,contrib['reconstruction']):
                    if g is not None:b.add_(g)
                if discriminator is not None:fakes.append(rec.detach());conditions.append(condition.detach().clone())
                for k,t in comp.items():sums[k]+=float(t.detach())/den
                h=hashlib.sha256(rec.detach().cpu().contiguous().numpy().tobytes()).hexdigest();ref(rec)
                trace.append(dict(window_position=pos,actual_qp=qa,reference_age=pos,I_position=0,normalization_denominator=den,reference_detached=True,reconstruction_sha256=h,feature_sha256=hashlib.sha256(pm.dpb[0].feature.cpu().contiguous().numpy().tobytes()).hexdigest(),**{k:float(t.detach()) for k,t in comp.items()}))
        finally:hook.remove();latent_hook.remove()
        assert len(trace)==15 and len(warmup)==48
        g0=component_grads['base'];ga=component_grads['adv']
        def norm(gs):return sum(float(g.double().square().sum()) for g in gs)**.5
        n0,na=norm(g0),norm(ga);assert math.isfinite(n0) and math.isfinite(na) and n0>0
        adv_weight=(cfg['rho']*n0/(na+cfg['gradient_eps']) if min(n0,na)>cfg['gradient_zero_threshold'] else 0.) if discriminator is not None and requested_weight is None else (requested_weight or 0.)
        ratio=adv_weight*na/n0;assert requested_weight is not None or ratio<=.05+1e-8
        dot=sum(float((g.double()*a.double()).sum()) for g,a in zip(g0,ga))
        sums.update(B1_gradient_norm=n0,unweighted_adv_gradient_norm=na,weighted_adv_gradient_norm=adv_weight*na,adv_B1_ratio=ratio,ceiling_triggered=0.,gradient_cosine=dot/(n0*na) if na else 0.,lambda_eff=adv_weight)
        off=0
        for k,m in models.items():
            n=len([p for p in m.parameters() if p.requires_grad]);nb,ng=norm(g0[off:off+n]),norm(ga[off:off+n]);off+=n
            sums[k+'_B1_gradient_norm']=nb;sums[k+'_adv_gradient_norm']=ng;sums[k+'_weighted_adv_B1_ratio']=adv_weight*ng/nb if nb else 0.
        sums['weighted_projected_adv']=adv_weight*sums['projected_adv'];sums['loss']+=sums['weighted_projected_adv']
        sums['gradient_control_zero_safety']=float(min(n0,na)<=cfg['gradient_zero_threshold']) if discriminator is not None else 0.
        if mode!='calibrate':
            for p,g,a in zip(params,g0,ga):p.grad=g.add(a,alpha=adv_weight)
        sums.update(beta_q=beta,lambda_q=common.lambda_q(q),weighted_DISTS=.5*sums['DISTS'],lambda_rec=rec_weight,lambda_adv=adv_weight,adversarial_unit_gradient_scale=adv_scale,branch_lambda_rec=rec_weight,reconstruction_backward_scale=(rec_scale/rec_weight if rec_weight else rec_scale),reconstruction_unit_gradient_scale=rec_scale)
        for name,buffers in contrib.items():
            off=0
            for k,m in models.items():
                n=len([p for p in m.parameters() if p.requires_grad]);sums[name+'_'+k+'_weighted_gradient_norm']=sum(float(g.double().square().sum()) for g in buffers[off:off+n])**.5;off+=n
            sums[name+'_global_weighted_gradient_norm']=sum(float(g.double().square().sum()) for g in buffers)**.5
        if mode=='calibrate':return sums,target,trace,warmup,component_grads
        return sums,target,trace,warmup,list(zip(fakes,conditions))
    return run
