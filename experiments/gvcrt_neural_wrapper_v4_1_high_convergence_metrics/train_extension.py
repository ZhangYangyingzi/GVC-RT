import argparse
import fcntl
import importlib.util
import json
import math
import os
import random
import torch
import torch.nn.functional as F
from experiment_utils import ROOT, V4, dump, read, write, sha

spec=importlib.util.spec_from_file_location('v4_training',V4/'train.py')
v4=importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)

def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    lock=open(ROOT/'parts/train.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/'config.json').read_text()); device=torch.device('cuda:0')
    checkpoints=sorted((ROOT/'checkpoints/beta_high').glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    path=checkpoints[-1]; cp=torch.load(path,map_location='cpu',weights_only=True)
    start=int(cp['step']); assert start in (10000,15000,20000) and cp['branch']=='beta_high' and cp['beta']==cfg['beta']
    wrapper=v4.NeuralWrapper().to(device).float().train()
    i_model,p_model=v4.load_models(device); i_model.requires_grad_(False); p_model.requires_grad_(False)
    bridge=p_model.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=p_model.recon_generation_net.decoder.float().train().requires_grad_(True)
    models={'wrapper':wrapper,'bridge':bridge,'generator':generator}
    for name,m in models.items(): m.load_state_dict(cp[name],strict=True)
    param_lists={name:list(m.parameters()) for name,m in models.items()}
    optcfg=cfg['optimizer']
    optimizer=torch.optim.AdamW([dict(params=ps,lr=optcfg[name+'_lr'],name=name) for name,ps in param_lists.items()],weight_decay=optcfg['weight_decay'])
    audit=dict(source_checkpoint=cfg['source_checkpoint'],source_step=10000,actual_resume_checkpoint=str(path),actual_resume_step=start,
               optimizer_state_restored=False,optimizer_parameter_state_count=len(cp.get('optimizer',{}).get('state',{})))
    try:
        assert cp['optimizer']['state'], 'empty optimizer state'
        optimizer.load_state_dict(cp['optimizer'])
        assert equal(optimizer.state_dict(),cp['optimizer']), 'optimizer state differs after restore'
        for group in optimizer.param_groups:
            assert group['lr']==optcfg[group['name']+'_lr'] and group['weight_decay']==optcfg['weight_decay']
        for state in optimizer.state.values():
            assert all(k in state for k in ('exp_avg','exp_avg_sq','step'))
            assert int(state['step'])==start
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        audit['optimizer_state_restored']=True
    except Exception as exc:
        audit.update(status='FAIL',error=str(exc)); dump('resume_audit.json',audit); raise
    for name,m in models.items(): audit[name+'_equal_before_resume']=equal(m.state_dict(),cp[name])
    assert all(audit[n+'_equal_before_resume'] for n in models)
    trainable={id(p) for ps in param_lists.values() for p in ps}
    frozen_count=sum(p.numel() for p in i_model.parameters() if p.requires_grad)+sum(p.numel() for p in p_model.parameters() if p.requires_grad and id(p) not in trainable)
    assert frozen_count==0
    hashes={'compression':v4.compression_hash(i_model,p_model),**{name:v4.module_hash(m) for name,m in models.items()}}
    assert hashes['compression']==cp['initial_hashes']['compression']
    quality=v4.quality_models(device)
    records=json.loads((ROOT/'train_manifest.json').read_text())['videos']; assert len(records)==256
    history_path=ROOT/'training_logs/training_beta_high_extension.csv'
    history=[r for r in read(history_path) if int(r['step'])<=start] if history_path.exists() else []
    assert len(history)==start-10000
    rng=random.Random()
    for key,label,setter,getter in [('sampling_rng_state','sampling_rng_restored',rng.setstate,rng.getstate),
                                   ('torch_rng_state','torch_rng_restored',torch.set_rng_state,torch.get_rng_state),
                                   ('cuda_rng_state','cuda_rng_restored',torch.cuda.set_rng_state,torch.cuda.get_rng_state)]:
        if key in cp: setter(cp[key]); audit[label]=equal(getter(),cp[key]); assert audit[label]
        else: audit[label]='missing'
    missing=[k for k in ('python_rng_state','numpy_rng_state') if k not in cp]
    audit.update(python_global_rng='missing',numpy_rng='missing',missing_rng_states=missing,
                 compression_trainable_parameters=frozen_count,
                 trainable_parameter_counts={n:sum(p.numel() for p in ps) for n,ps in param_lists.items()},
                 compression_hash=hashes['compression'],status='PASS_WITH_MISSING_RNG' if missing else 'PASS')
    audit_path='resume_audit.json' if start==10000 else f'parts/resume_audit_step_{start}.json'
    dump(audit_path,audit)
    initial_hashes=cp.get('extension_initial_hashes',hashes); del cp
    dump(f'parts/parameter_audit_step_{start}.json',dict(hashes=hashes,compression_trainable_parameters=frozen_count,trainable=audit['trainable_parameter_counts']))
    print(f'RESUMED step={start} optimizer_states={audit["optimizer_parameter_state_count"]} gpu={args.gpu}',flush=True)
    all_params=[p for ps in param_lists.values() for p in ps]
    for step in range(start+1,20001):
        frames,plan=v4.sample_clip(records,rng,device); qp=rng.randrange(4)
        p_model.clear_dpb(); p_model.set_curr_poc(0)
        with torch.no_grad():
            first_proxy=wrapper(frames[0]); encoded=i_model.compress(v4.codec_input(first_proxy),qp)
            p_model.add_ref_frame(None,encoded['x_hat'])
        optimizer.zero_grad(set_to_none=True); captured=[]
        hook=p_model.dec.register_forward_hook(lambda _m,_a,out:captured.append(out))
        sums={k:0. for k in ('loss','LPIPS','DISTS','R_est_bpp','proxy_L1','PSNR','MS_SSIM')}
        try:
            for frame in frames[1:]:
                proxy=wrapper(frame); reconstruction,rate,_=v4.joint_ste_forward(p_model,proxy,qp)
                distortion,lpips,dists=v4.perceptual(reconstruction,frame,quality)
                rate_bpp=rate/(frame.shape[-2]*frame.shape[-1]); l1=F.l1_loss(proxy,frame)
                loss=(distortion+cfg['beta']*rate_bpp+cfg['lambda_proxy']*l1)/3
                if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss {step}')
                loss.backward()
                with torch.no_grad():
                    p_model.add_ref_frame(captured[-1].detach(),(reconstruction.detach()*2-1).half())
                sums['loss']+=float(loss); sums['LPIPS']+=float(lpips)/3; sums['DISTS']+=float(dists)/3
                sums['R_est_bpp']+=float(rate_bpp)/3; sums['proxy_L1']+=float(l1)/3
                sums['PSNR']+=-10*math.log10(max(float((reconstruction.detach()-frame).square().mean()),1e-15))/3
                sums['MS_SSIM']+=v4.ms_ssim_rgb(reconstruction.detach(),frame)/3
        finally: hook.remove()
        grads={n+'_gradient_norm':v4.grad_norm(ps) for n,ps in param_lists.items()}
        norm=float(torch.nn.utils.clip_grad_norm_(all_params,optcfg['gradient_clip_norm']))
        assert norm>0 and all(math.isfinite(x) for x in [norm,*grads.values(),*sums.values()])
        optimizer.step()
        history.append(dict(step=step,beta=cfg['beta'],qp=qp,**plan,**sums,**grads,all_finite=True))
        if step==start+1 or step%100==0:
            write(history_path,history); print(f'beta_high {step}/20000 loss={sums["loss"]:.6f}',flush=True)
        if step in (15000,20000):
            assert v4.compression_hash(i_model,p_model)==hashes['compression']
            payload=dict(schema='gvcrt_v4_1',branch='beta_high',stage='beta_high',step=step,beta=cfg['beta'],
                         **{n:m.state_dict() for n,m in models.items()},optimizer=optimizer.state_dict(),
                         initial_hashes=hashes,extension_initial_hashes=initial_hashes,
                         sampling_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state())
            path=ROOT/'checkpoints/beta_high'/f'step_{step}.pt'; temporary=path.with_suffix('.tmp')
            torch.save(payload,temporary); temporary.replace(path)
            dump(f'parts/parameter_audit_step_{step}.json',dict(compression_hash=v4.compression_hash(i_model,p_model),
                 **{n+'_hash':v4.module_hash(m) for n,m in models.items()},status='PASS'))
    write(history_path,history)
    assert len(history)==10000 and [int(r['step']) for r in history]==list(range(10001,20001))
    assert sha(cfg['source_checkpoint'])==json.loads((ROOT/'source_snapshot.json').read_text())[cfg['source_checkpoint']]
    dump('parts/train_done.json',dict(status='PASS',step=20000,extension_updates=10000,compression_hash_unchanged=v4.compression_hash(i_model,p_model)==hashes['compression']))

if __name__=='__main__': main()
