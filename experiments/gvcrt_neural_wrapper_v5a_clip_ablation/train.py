import argparse
import fcntl
import hashlib
import math
import os
import random
import cv2
import torch
import torch.nn.functional as F
from v5_utils import ROOT,V4,V41,BRANCHES,sha,load,dump,read,write,module

v4=module('v5_source_training',V4/'train.py')

def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def state_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def sample_clip(records,rng,device,clip_length,crop=256):
    decoded=0;rejected=0
    for _ in range(20):
        record=rng.choice(records);cap=cv2.VideoCapture(record['path'])
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if count<clip_length or width<crop or height<crop:cap.release();rejected+=1;continue
        start=rng.randrange(count-clip_length+1);x=rng.randrange(width-crop+1);y=rng.randrange(height-crop+1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[];indices=[]
        for position in range(clip_length):
            ok,bgr=cap.read()
            if not ok:break
            decoded+=1;indices.append(int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1)
            rgb=cv2.cvtColor(bgr[y:y+crop,x:x+crop],cv2.COLOR_BGR2RGB).copy()
            frames.append(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device).float()/255)
        cap.release()
        if len(frames)==clip_length and indices==list(range(start,start+clip_length)):
            return frames,dict(video=record['filename'],start=start,crop_x=x,crop_y=y,width=width,height=height,
                clip_length=clip_length,first_frame_index=indices[0],last_frame_index=indices[-1],consecutive_frames=True,
                raw_decoded_frames=decoded,rejected_sample_attempts=rejected)
        rejected+=1
    raise RuntimeError('failed to decode a complete consecutive clip without padding')

def save(path,branch,step,cfg,models,optimizer,rng,hashes):
    payload=dict(schema='gvcrt_v5a',branch=branch,stage=branch,step=step,global_step=20000+step,beta=cfg['beta'],
        clip_length=BRANCHES[branch]['clip_length'],optimized_P_frames=step*(BRANCHES[branch]['clip_length']-1),
        **{n:m.state_dict() for n,m in models.items()},optimizer=optimizer.state_dict(),initial_hashes=hashes,
        sampling_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
        source_checkpoint=cfg['source_checkpoint'],source_checkpoint_sha256=sha(cfg['source_checkpoint']))
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    parser.add_argument('--branch',choices=tuple(BRANCHES),required=True);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    lock=open(ROOT/'parts'/f'train_{args.branch}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=load('config.json');device=torch.device('cuda:0');branch=BRANCHES[args.branch]
    clip_length=branch['clip_length'];num_p_frames=clip_length-1;total=branch['updates']
    assert total*num_p_frames==21000 and cfg['beta']==17.302782275540444
    source=cfg['source_checkpoint'];assert sha(source)==load('source_snapshot.json')[source]
    candidates=sorted((ROOT/'checkpoints'/args.branch).glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    resume=candidates[-1] if candidates else source;cp=torch.load(resume,map_location='cpu',weights_only=True)
    start=int(cp['step']) if candidates else 0
    assert (cp['branch']==args.branch if candidates else cp['step']==20000) and cp['beta']==cfg['beta']
    wrapper=v4.NeuralWrapper().to(device).float().train()
    i_model,p_model=v4.load_models(device);i_model.requires_grad_(False);p_model.requires_grad_(False)
    bridge=p_model.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=p_model.recon_generation_net.decoder.float().train().requires_grad_(True)
    models={'wrapper':wrapper,'bridge':bridge,'generator':generator}
    params={n:list(m.parameters()) for n,m in models.items()}
    for n,m in models.items():m.load_state_dict(cp[n],strict=True)
    opt=cfg['optimizer'];optimizer=torch.optim.AdamW([dict(params=ps,lr=opt[n+'_lr'],name=n) for n,ps in params.items()],weight_decay=opt['weight_decay'])
    assert cp['optimizer']['state'],'empty AdamW state'
    optimizer.load_state_dict(cp['optimizer']);assert equal(optimizer.state_dict(),cp['optimizer'])
    for group in optimizer.param_groups:assert group['lr']==opt[group['name']+'_lr'] and group['weight_decay']==opt['weight_decay']
    for state in optimizer.state.values():
        assert int(state['step'])==20000+start
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    ids={id(p) for ps in params.values() for p in ps}
    frozen=sum(p.numel() for p in i_model.parameters() if p.requires_grad)+sum(p.numel() for p in p_model.parameters() if p.requires_grad and id(p) not in ids)
    assert frozen==0
    hashes={'compression':v4.compression_hash(i_model,p_model),**{n:v4.module_hash(m) for n,m in models.items()}}
    assert hashes['compression']==cp['initial_hashes']['compression']
    initial_hashes=cp['initial_hashes'] if candidates else hashes
    quality=v4.quality_models(device);rng=random.Random();rng.setstate(cp['sampling_rng_state']);torch.set_rng_state(cp['torch_rng_state']);torch.cuda.set_rng_state(cp['cuda_rng_state'])
    restored=equal(rng.getstate(),cp['sampling_rng_state']) and equal(torch.get_rng_state(),cp['torch_rng_state']) and equal(torch.cuda.get_rng_state(),cp['cuda_rng_state'])
    assert restored
    audit=dict(status='PASS',branch=args.branch,source_checkpoint=source,source_checkpoint_sha256=sha(source),source_step=20000,
        actual_resume_step=start,optimizer_state_restored=True,optimizer_state_count=len(optimizer.state),
        optimizer_state_hash=state_hash(optimizer.state_dict()),hashes=hashes,
        **{n+'_equal_to_checkpoint':equal(m.state_dict(),cp[n]) for n,m in models.items()},
        trainable_parameters={n:sum(p.numel() for p in ps if p.requires_grad) for n,ps in params.items()},
        compression_trainable_parameters=frozen,saved_rng_restored=restored,
        missing_rng_states=[k for k in ('python_rng_state','numpy_rng_state') if k not in cp],
        clip_length=clip_length,loss_denominator=num_p_frames,expected_updates=total,gpu=args.gpu)
    dump(f'parts/init_{args.branch}.json' if start==0 else f'parts/resume_{args.branch}_{start}.json',audit)
    del cp
    if not candidates:save(ROOT/'checkpoints'/args.branch/'step_0000.pt',args.branch,0,cfg,models,optimizer,rng,initial_hashes)
    records=load('train_manifest.json')['videos'];assert len(records)==256
    logpath=ROOT/'training_logs'/f'training_{args.branch}.csv';pospath=ROOT/'training_logs'/f'positions_{args.branch}.csv'
    history=[r for r in read(logpath) if int(r['step'])<=start] if logpath.exists() else []
    positions=[r for r in read(pospath) if int(r['step'])<=start] if pospath.exists() else []
    assert len(history)==start and len(positions)==start*num_p_frames
    temporal=load(f'parts/temporal_{args.branch}.json') if (ROOT/'parts'/f'temporal_{args.branch}.json').exists() else dict(branch=args.branch,tests=[],status='PENDING')
    flat=[p for ps in params.values() for p in ps]
    print('INITIALIZED',args.branch,'step',start,'optimizer_states',len(optimizer.state),'denominator',num_p_frames,'gpu',args.gpu,flush=True)
    for step in range(start+1,total+1):
        frames,plan=sample_clip(records,rng,device,clip_length);qp=rng.randrange(4)
        p_model.clear_dpb();p_model.set_curr_poc(0)
        with torch.no_grad():
            first_proxy=wrapper(frames[0]);encoded=i_model.compress(v4.codec_input(first_proxy),qp)
            p_model.add_ref_frame(None,encoded['x_hat'])
        assert not first_proxy.requires_grad and not encoded['x_hat'].requires_grad
        optimizer.zero_grad(set_to_none=True);captured=[]
        hook=p_model.dec.register_forward_hook(lambda _m,_a,out:captured.append(out))
        sums={k:0. for k in ('loss','LPIPS','DISTS','R_est_bpp','proxy_L1','PSNR','MS_SSIM')}
        previous_reconstruction=previous_feature=None;audit_step=step==1 or step%1000==0 or step==total
        try:
            for position,frame in enumerate(frames[1:],1):
                proxy=wrapper(frame);reconstruction,rate,_=v4.joint_ste_forward(p_model,proxy,qp)
                distortion,lpips,dists=v4.perceptual(reconstruction,frame,quality)
                rate_bpp=rate/(frame.shape[-2]*frame.shape[-1]);l1=F.l1_loss(proxy,frame)
                frame_objective=distortion+cfg['beta']*rate_bpp+cfg['lambda_proxy']*l1
                loss=frame_objective/num_p_frames
                assert torch.isfinite(loss)
                if audit_step and previous_reconstruction is not None:
                    grads=torch.autograd.grad(loss,(previous_reconstruction,previous_feature),allow_unused=True,retain_graph=True)
                    assert all(g is None for g in grads),'future loss connected to previous DPB tensors'
                    temporal['tests'].append(dict(step=step,position=position,previous_reconstruction_gradient='None',previous_feature_gradient='None'))
                loss.backward()
                with torch.no_grad():
                    p_model.add_ref_frame(captured[-1].detach(),(reconstruction.detach()*2-1).half())
                for entry in p_model.dpb:
                    for tensor in vars(entry).values():
                        if torch.is_tensor(tensor):assert not tensor.requires_grad and tensor.grad_fn is None
                row=dict(branch=args.branch,step=step,global_step=20000+step,position=position,qp=qp,
                    R_est_bpp=float(rate_bpp),LPIPS=float(lpips),DISTS=float(dists),proxy_L1=float(l1),
                    PSNR=-10*math.log10(max(float((reconstruction.detach()-frame).square().mean()),1e-15)),
                    MS_SSIM=v4.ms_ssim_rgb(reconstruction.detach(),frame),frame_objective=float(frame_objective))
                assert all(math.isfinite(float(row[k])) for k in ('R_est_bpp','LPIPS','DISTS','proxy_L1','PSNR','MS_SSIM','frame_objective'))
                positions.append(row);sums['loss']+=float(loss)
                for key in sums:
                    if key!='loss':sums[key]+=row[key]/num_p_frames
                previous_reconstruction=reconstruction;previous_feature=captured[-1]
        finally:hook.remove()
        grads={n+'_gradient_norm':v4.grad_norm(ps) for n,ps in params.items()}
        norm=float(torch.nn.utils.clip_grad_norm_(flat,opt['gradient_clip_norm']))
        assert norm>0 and all(math.isfinite(x) for x in [norm,*sums.values(),*grads.values()])
        optimizer.step()
        history.append(dict(branch=args.branch,step=step,global_step=20000+step,beta=cfg['beta'],qp=qp,
            optimized_P_frames=step*num_p_frames,num_p_frames=num_p_frames,**plan,**sums,**grads,all_finite=True))
        if audit_step:
            temporal.update(status='PASS',temporal_gradient_through_DPB=False,BPTT_disabled=True,DPB_tensors_detached=True,first_frame_no_grad=True)
            dump(f'parts/temporal_{args.branch}.json',temporal)
        if step==start+1 or step%100==0 or step==total:
            write(logpath,history);write(pospath,positions)
            print(args.branch,f'{step}/{total}','P_frames',step*num_p_frames,'loss',sums['loss'],flush=True)
        if step%1000==0 or step==total:
            assert v4.compression_hash(i_model,p_model)==initial_hashes['compression']
            save(ROOT/'checkpoints'/args.branch/f'step_{step:04d}.pt',args.branch,step,cfg,models,optimizer,rng,initial_hashes)
    assert len(history)==total and len(positions)==21000
    dump(f'parts/train_{args.branch}_done.json',dict(status='PASS',branch=args.branch,updates=total,optimized_P_frames=21000,
        raw_decoded_frames=sum(int(r['raw_decoded_frames']) for r in history),clip_length=clip_length,loss_denominator=num_p_frames,
        initial_hashes=initial_hashes,final_hashes={'compression':v4.compression_hash(i_model,p_model),**{n:v4.module_hash(m) for n,m in models.items()}},
        compression_hash_unchanged=v4.compression_hash(i_model,p_model)==initial_hashes['compression']))
    print('TRAIN PASS',args.branch,flush=True)

if __name__=='__main__':main()
