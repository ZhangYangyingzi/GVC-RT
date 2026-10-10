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
    common=module('ref21_common',V62/'fullqp_common.py');wrapper=models['wrapper']
    def run(frames,q,branch,details=True,adv_weight=.01,adv_only=False,adv_scale=4096.):
        assert len(frames)==64 and branch in BRANCHES
        den=15;beta=common.beta_q('schedule_s1p0',q);origin=0 if branch=='B' else 48
        target_qps=[common.frame_qp(pm,q,p-origin) for p in range(49,64)]
        assert target_qps==[common.frame_qp(pm,q,p) for p in range(49,64)]
        trace=[];warmup=[];capt=[];fakes=[]
        hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
        sums={k:0. for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','lambda_proxy_proxy_L1','L_adv','weighted_adv')}
        if discriminator is not None:
            discriminator.eval().requires_grad_(False)
            d_before={k:t.detach().clone() for k,t in discriminator.state_dict().items()}
        def ref(rec):
            assert len(capt)==1
            with torch.no_grad():pm.add_ref_frame(capt.pop().detach(),(rec.detach()*2-1).half())
            assert not pm.dpb[0].feature.requires_grad and not pm.dpb[0].frame.requires_grad
        try:
            pm.clear_dpb();pm.set_curr_poc(0)
            with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[origin]),q)['x_hat'])
            for pos in range(origin+1,49):
                with torch.no_grad():
                    rec,rate,_=v4.joint_ste_forward(pm,wrapper(frames[pos]),common.frame_qp(pm,q,pos))
                    assert not rec.requires_grad
                    ref(rec)
                    warmup.append(dict(position=pos,actual_qp=common.frame_qp(pm,q,pos),reference_age=pos,rec_sha256=hashlib.sha256(rec.cpu().contiguous().numpy().tobytes()).hexdigest(),feature_sha256=hashlib.sha256(pm.dpb[0].feature.cpu().contiguous().numpy().tobytes()).hexdigest(),no_grad=True,reference_detached=True))
            for pos,qactual in zip(range(49,64),target_qps):
                frame=frames[pos];proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,qactual)
                lp=quality[0](rec,frame,normalize=True).mean();ds=quality[1](rec,frame,require_grad=True).mean()
                rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame);dist=lp+.5*ds
                adv=-discriminator(rec*2-1).mean() if discriminator is not None else rec.new_zeros(())
                weighted=adv*adv_weight if branch=='C' else rec.new_zeros(())
                comp=dict(loss=dist+beta*rb+.01*pl+weighted,LPIPS=lp,DISTS=ds,D=dist,rate_bpp=rb,beta_rate=beta*rb,proxy_L1=pl,lambda_proxy_proxy_L1=.01*pl,L_adv=adv,weighted_adv=weighted)
                assert all(bool(torch.isfinite(t).all()) for t in comp.values())
                if branch=='C' and adv_weight!=0:
                    params=[p for m in models.values() for p in m.parameters()]
                    assert all(p.dtype==torch.float32 and (p.grad is None or p.grad.dtype==torch.float32) for p in params)
                    scaled=torch.autograd.grad(weighted*(adv_scale/den),params,retain_graph=not adv_only,allow_unused=True)
                    restored=[None if g is None else g.detach().float()/adv_scale for g in scaled]
                    assert all(g is None or bool(torch.isfinite(g).all()) for g in restored)
                    if not adv_only:((dist+beta*rb+.01*pl)/den).backward()
                    with torch.no_grad():
                        for p,g in zip(params,restored):
                            if g is not None:
                                if p.grad is None:p.grad=g.clone()
                                else:p.grad.add_(g)
                    del scaled,restored
                else:
                    ((weighted if adv_only else comp['loss'])/den).backward()
                for k,t in comp.items():sums[k]+=float(t.detach())/den
                rec_hash=hashlib.sha256(rec.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                if discriminator is not None:fakes.append(rec.detach())
                ref(rec)
                trace.append(dict(window_position=pos,actual_qp=qactual,reference_age=pos-origin,I_position=origin,normalization_denominator=den,reference_detached=True,reconstruction_sha256=rec_hash,**{k:float(t.detach()) for k,t in comp.items()}))
        finally:hook.remove()
        if discriminator is not None:
            assert all(torch.equal(t,d_before[k]) for k,t in discriminator.state_dict().items())
            assert all(p.grad is None for p in discriminator.parameters())
        assert len(trace)==15 and len(warmup)==(48 if branch=='B' else 0)
        sums.update(beta_q=beta,lambda_q=common.lambda_q(q),weighted_DISTS=.5*sums['DISTS'])
        return sums,target_qps,trace,warmup,fakes
    return run
