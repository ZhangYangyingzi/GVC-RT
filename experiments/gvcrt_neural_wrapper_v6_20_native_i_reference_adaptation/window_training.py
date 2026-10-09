"""Identical native STE path; 15 supervised P frames and detached references."""
import math
from io20 import *
def replay_rows(branch):return load(ROOT/'plans/AB.json')['updates']
def replay(row,branch,device):
    import cv2,numpy as np,torch
    from PIL import Image
    d=row['domain'];suffix='';videos=load(ROOT/f'train_manifest_{d}{suffix}.json')['videos'];v=next(v for v in videos if v['filename' if d=='ulong' else 'name']==row['video'])
    assert row['source_path']==v['path' if d=='ulong' else 'source_path'];assert row['source_sha256']==v['sha256' if d=='ulong' else 'source_sha256']
    x,y=row['crop_x'],row['crop_y'];arrays=[];full=[];indices=row['canonical_indices'];assert indices==list(range(indices[0],indices[0]+16))
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
    hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays];assert hashes==row['cropped_RGB_sha256'] and full==row['full_RGB_sha256'];assert len(arrays)==16
    return [torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).float()/255 for a in arrays]
def make_runner(v4,im,pm,models,quality):
    import torch
    import torch.nn.functional as F
    common=module('long18_common',V62/'fullqp_common.py');wrapper=models['wrapper'];cfg=load(ROOT/'config.json')
    def run(frames,q,branch,details=False):
        n=len(frames);assert n in (4,16);den=n-1;beta=common.beta_q('schedule_s1p0',q)
        actual=[common.frame_qp(pm,q,i) for i in range(n)];resets=[0]
        assert branch in BRANCHES and resets==[0];trace=[];capt=[]
        hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o));sums={k:0.0 for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','lambda_proxy_proxy_L1')}
        try:
            for pos in range(1,n):
                if pos-1 in resets:
                    pm.clear_dpb();pm.set_curr_poc(0)
                    with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[pos-1] if branch=='native_i_adapt' else wrapper(frames[pos-1])),q)['x_hat'])
                frame=frames[pos];proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,actual[pos])
                lp=quality[0](rec,frame,normalize=True).mean();ds=quality[1](rec,frame,require_grad=True).mean();dist=lp+.5*ds
                rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame)
                components=dict(loss=dist+beta*rb+.01*pl,LPIPS=lp,DISTS=ds,D=dist,rate_bpp=rb,beta_rate=beta*rb,proxy_L1=pl,lambda_proxy_proxy_L1=.01*pl)
                assert all(bool(torch.isfinite(x).all()) for x in components.values())
                (components['loss']/den).backward()
                for k,x in components.items():sums[k]+=float(x.detach())/den
                assert len(capt)==1
                with torch.no_grad():pm.add_ref_frame(capt[-1].detach(),(rec.detach()*2-1).half())
                capt.clear()
                trace.append(dict(window_position=pos,actual_qp=actual[pos],reference_reset_before=pos-1 in resets,I_reference_position=pos-1 if pos-1 in resets else None,I_external_qp=q if pos-1 in resets else None,normalization_denominator=den,reference_detached=True,**({k:float(x.detach()) for k,x in components.items()} if details else {})))
        finally:hook.remove()
        sums.update(beta_q=beta,lambda_q=common.lambda_q(q),rate_to_distortion_ratio=sums['beta_rate']/sums['D'],weighted_DISTS=.5*sums['DISTS'])
        assert [r['window_position'] for r in trace]==list(range(1,n))
        return sums,actual,trace
    return run
