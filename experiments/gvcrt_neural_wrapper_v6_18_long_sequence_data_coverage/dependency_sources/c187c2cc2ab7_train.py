"""V4.1 update loop, unchanged except expanded data/QP domain and duration."""
import argparse,fcntl,math,random
from v61_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),default=4);args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import torch.nn.functional as F
    lock=(ROOT/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=load(ROOT/'config.json');assert load(ROOT/'data_split_integrity.json')['status']=='PASS';frozen()
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    v4=module('v61_frozen_v4_training',V4/'train.py');device=torch.device('cuda:0');assert torch.cuda.device_count()==1
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    im.requires_grad_(False);pm.requires_grad_(False);bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator);params={k:list(m.parameters()) for k,m in models.items()}
    oc=cfg['optimizer'];optimizer=torch.optim.AdamW([dict(params=ps,lr=oc[k+'_lr'],name=k) for k,ps in params.items()],weight_decay=oc['weight_decay'])
    available=list((ROOT/'checkpoints').glob('additional_*.pt'))+list((ROOT/'checkpoints/recovery').glob('additional_*.pt'))
    source=max(available,key=lambda p:int(p.stem.split('_')[-1])) if available else Path(cfg['source_checkpoint'])
    cp=torch.load(source,map_location='cpu',weights_only=True);start=int(cp.get('additional_updates',0))
    assert cp['step']==20000+start and cp['beta']==cfg['beta']
    for k,m in models.items():m.load_state_dict(cp[k],strict=True)
    optimizer.load_state_dict(cp['optimizer'])
    for group in optimizer.param_groups:assert group['lr']==oc[group['name']+'_lr'] and group['weight_decay']==oc['weight_decay']
    for state in optimizer.state.values():assert int(state['step'])==20000+start and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    trainable={id(p) for ps in params.values() for p in ps}
    assert sum(p.numel() for p in im.parameters() if p.requires_grad)==0
    assert sum(p.numel() for p in pm.parameters() if p.requires_grad and id(p) not in trainable)==0
    compression=v4.compression_hash(im,pm);assert compression==cp['initial_hashes']['compression']
    quality=v4.quality_models(device)
    rng=random.Random();rng.setstate(cp['sampling_rng_state']);torch.set_rng_state(cp['torch_rng_state']);torch.cuda.set_rng_state(cp['cuda_rng_state'])
    initial_hashes=cp['initial_hashes'];del cp
    records=load(ROOT/'train_manifest_1024.json')['videos'];assert len(records)==1024
    logpath=ROOT/'training_log.jsonl';history=[]
    if logpath.exists():
        lines=logpath.read_text().splitlines();complete=[]
        for line in lines:
            try:r=json.loads(line)
            except json.JSONDecodeError:break
            if int(r['additional_update'])<=start:complete.append(r)
        if len(complete)!=len(lines):
            archive=ROOT/'logs'/f'training_recovery_{time.time_ns()}.jsonl';logpath.replace(archive)
            with logpath.open('w') as out:
                for r in complete:out.write(json.dumps(r,allow_nan=False)+'\n')
        history=complete
    assert [r['additional_update'] for r in history]==list(range(1,start+1))
    histogram={q:sum(int(r['qp'])==q for r in history) for q in range(10)}
    dump(ROOT/'parts'/f'resume_audit_{start}_{time.time_ns()}.json',dict(status='PASS',source=str(source),source_sha256=sha(source),additional_update=start,global_step=20000+start,
        optimizer_restored=True,sampling_rng_restored=True,compression_hash=compression,compression_trainable_parameters=0,
        trainable_parameter_counts={k:sum(p.numel() for p in ps) for k,ps in params.items()},physical_gpu=args.gpu))
    def save(step,recovery=False):
        assert v4.compression_hash(im,pm)==compression
        path=(ROOT/'checkpoints/recovery'/f'additional_{step:05d}.pt') if recovery else checkpoint(step);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():return
        payload=dict(schema='gvcrt_v6_1',branch='beta_high',stage='beta_high',step=20000+step,additional_updates=step,source_step=20000,beta=cfg['beta'],
            **{k:m.state_dict() for k,m in models.items()},optimizer=optimizer.state_dict(),initial_hashes=initial_hashes,
            sampling_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),qp_histogram=histogram,
            source_checkpoint_sha256=cfg['source_checkpoint_sha256'],config_sha256=sha(ROOT/'config.json'),train_manifest_sha256=sha(ROOT/'train_manifest_1024.json'))
        temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
        if not recovery:
            record=dict(path=str(path),sha256=sha(path),additional_updates=step,global_step=20000+step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},compression_hash=compression)
            dump(ROOT/'parts'/f'checkpoint_{step:05d}.json',record)
            existing=load(ROOT/'checkpoint_hashes.json') if (ROOT/'checkpoint_hashes.json').exists() else {};existing[str(step)]=record;dump(ROOT/'checkpoint_hashes.json',existing)
    if not available:save(0)
    all_params=[p for ps in params.values() for p in ps]
    with logpath.open('a',buffering=1) as log:
        for additional in range(start+1,40001):
            begin=time.time();frames,plan=v4.sample_clip(records,rng,device);qp=rng.randrange(10)
            pm.clear_dpb();pm.set_curr_poc(0)
            with torch.no_grad():
                encoded=im.compress(v4.codec_input(wrapper(frames[0])),qp);pm.add_ref_frame(None,encoded['x_hat'])
            optimizer.zero_grad(set_to_none=True);captured=[];hook=pm.dec.register_forward_hook(lambda _m,_a,out:captured.append(out))
            sums={k:0.0 for k in ('loss','LPIPS','DISTS','R_est_bpp','proxy_L1','PSNR','MS_SSIM')}
            try:
                for frame in frames[1:]:
                    proxy=wrapper(frame);reconstruction,rate,_=v4.joint_ste_forward(pm,proxy,qp)
                    distortion,lpips,dists=v4.perceptual(reconstruction,frame,quality);rate_bpp=rate/(frame.shape[-2]*frame.shape[-1]);l1=F.l1_loss(proxy,frame)
                    loss=(distortion+cfg['beta']*rate_bpp+cfg['lambda_proxy']*l1)/3
                    assert torch.isfinite(loss),additional;loss.backward()
                    with torch.no_grad():pm.add_ref_frame(captured[-1].detach(),(reconstruction.detach()*2-1).half())
                    sums['loss']+=float(loss.detach());sums['LPIPS']+=float(lpips.detach())/3;sums['DISTS']+=float(dists.detach())/3
                    sums['R_est_bpp']+=float(rate_bpp.detach())/3;sums['proxy_L1']+=float(l1.detach())/3
                    sums['PSNR']+=-10*math.log10(max(float((reconstruction.detach()-frame).square().mean()),1e-15))/3;sums['MS_SSIM']+=v4.ms_ssim_rgb(reconstruction.detach(),frame)/3
            finally:hook.remove()
            grads={k+'_gradient_norm':v4.grad_norm(ps) for k,ps in params.items()};norm=float(torch.nn.utils.clip_grad_norm_(all_params,oc['gradient_clip_norm']))
            assert norm>0 and all(math.isfinite(x) for x in [norm,*grads.values(),*sums.values()]);optimizer.step();histogram[qp]+=1
            r=dict(additional_update=additional,global_step=20000+additional,qp=qp,actual_i_qp=qp,actual_p_qps=[qp]*3,beta=cfg['beta'],**plan,**sums,**grads,all_finite=True,elapsed_seconds=time.time()-begin)
            log.write(json.dumps(r,allow_nan=False)+'\n')
            if additional%100==0 or additional==start+1:
                dump(ROOT/'training_status.json',dict(status='RUNNING',additional_updates=additional,target=40000,global_step=20000+additional,updated_unix=time.time(),last_loss=sums['loss'],qp_histogram=histogram))
                write(ROOT/'training_qp_histogram.csv',[dict(external_qp=q,count=histogram[q],fraction=histogram[q]/additional,expected_probability=.1) for q in range(10)])
                print('TRAIN',additional,'/40000',sums['loss'],flush=True);log.flush()
            if additional in cfg['checkpoint_additional_updates']:save(additional)
            elif additional%1000==0:save(additional,recovery=True)
    assert sum(histogram.values())==40000
    from scipy.stats import chisquare
    chi,pvalue=chisquare([histogram[q] for q in range(10)]);valid=max(abs(histogram[q]-4000)/4000 for q in range(10))<=.1 and pvalue>=1e-6
    audit=load(ROOT/'qp_semantics_audit.json');audit.update(status='PASS' if valid else 'FAIL',per_qp_sampling_counts=histogram,chi_square=float(chi),chi_square_pvalue=float(pvalue),uniformity_rule='Each count within 10% of 4000 and chi-square p >=1e-6');dump(ROOT/'qp_semantics_audit.json',audit);assert valid
    assert v4.compression_hash(im,pm)==compression;frozen()
    dump(ROOT/'parts/train_done.json',dict(status='PASS',additional_updates=40000,global_step=60000,compression_unchanged=True,qp_histogram_valid=True))
    dump(ROOT/'training_status.json',dict(status='PASS',additional_updates=40000,target=40000,global_step=60000,qp_histogram=histogram))
if __name__=='__main__':main()
