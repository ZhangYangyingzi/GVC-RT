"""Identical native STE path; 15 supervised P frames and detached references."""
import math
from io23 import *
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
    common=module('rec22_common',V62/'fullqp_common.py');wrapper=models['wrapper'];params=[p for m in models.values() for p in m.parameters()]
    def run(frames,q,branch,details=True,rec_weight=None,mode='train',rec_scale=4096.):
        assert len(frames)==64 and branch in BRANCHES and discriminator is None
        if rec_weight is None:
            rec_weight=load(ROOT/'config.json')['branch_lambda_rec'][branch]
            assert rec_weight==WEIGHTS[branch]
        den=15;beta=common.beta_q('schedule_s1p0',q);trace=[];warmup=[];capt=[];component_grads={name:[torch.zeros_like(p) for p in params] for name in ('perc','rec')} if mode=='calibrate' else None
        assert all(p.dtype==torch.float32 and (p.grad is None or p.grad.dtype==torch.float32) for p in params)
        sums={k:0. for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','lambda_proxy_proxy_L1','reconstruction_L1','weighted_reconstruction_L1')}
        hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
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
                    qa=common.frame_qp(pm,q,pos);rec,rate,_=v4.joint_ste_forward(pm,wrapper(frames[pos]),qa);assert not rec.requires_grad;ref(rec)
                    warmup.append(dict(position=pos,actual_qp=qa,reference_age=pos,rec_sha256=hashlib.sha256(rec.cpu().contiguous().numpy().tobytes()).hexdigest(),feature_sha256=hashlib.sha256(pm.dpb[0].feature.cpu().contiguous().numpy().tobytes()).hexdigest(),no_grad=True,reference_detached=True))
            target=[common.frame_qp(pm,q,pos) for pos in range(49,64)]
            for pos,qa in zip(range(49,64),target):
                frame=frames[pos];proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
                lp=quality[0](rec,frame,normalize=True).mean();ds=quality[1](rec,frame,require_grad=True).mean();rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame);rl=F.l1_loss(rec,frame);dist=lp+.5*ds;base=dist+beta*rb+.01*pl
                comp=dict(loss=base+rec_weight*rl,LPIPS=lp,DISTS=ds,D=dist,rate_bpp=rb,beta_rate=beta*rb,proxy_L1=pl,lambda_proxy_proxy_L1=.01*pl,reconstruction_L1=rl,weighted_reconstruction_L1=rec_weight*rl)
                assert all(bool(torch.isfinite(t).all()) for t in comp.values())
                if mode=='calibrate':
                    gp=torch.autograd.grad(dist/den,params,retain_graph=True,allow_unused=True);add(gp,component_grads['perc'],1.);del gp
                    gr=torch.autograd.grad(rl*(rec_scale/den),params,allow_unused=True);add(gr,component_grads['rec'],rec_scale);del gr
                elif rec_weight!=0:
                    gs=torch.autograd.grad(rl*(rec_scale/den),params,retain_graph=True,allow_unused=True)
                    restored=[None if g is None else (g.detach().float()/rec_scale)*rec_weight for g in gs];del gs
                    (base/den).backward()
                    for p,g in zip(params,restored):
                        if g is not None:
                            assert bool(torch.isfinite(g).all()) and g.dtype==torch.float32
                            if p.grad is None:p.grad=g.clone()
                            else:p.grad.add_(g)
                    del restored
                else:(base/den).backward()
                for k,t in comp.items():sums[k]+=float(t.detach())/den
                h=hashlib.sha256(rec.detach().cpu().contiguous().numpy().tobytes()).hexdigest();ref(rec)
                trace.append(dict(window_position=pos,actual_qp=qa,reference_age=pos,I_position=0,normalization_denominator=den,reference_detached=True,reconstruction_sha256=h,feature_sha256=hashlib.sha256(pm.dpb[0].feature.cpu().contiguous().numpy().tobytes()).hexdigest(),**{k:float(t.detach()) for k,t in comp.items()}))
        finally:hook.remove()
        assert len(trace)==15 and len(warmup)==48
        sums.update(beta_q=beta,lambda_q=common.lambda_q(q),weighted_DISTS=.5*sums['DISTS'],lambda_rec=rec_weight,branch_lambda_rec=rec_weight,reconstruction_backward_scale=rec_scale,reconstruction_unit_gradient_scale=rec_scale)
        if mode=='calibrate':return sums,target,trace,warmup,component_grads
        return sums,target,trace,warmup,[]
    return run
