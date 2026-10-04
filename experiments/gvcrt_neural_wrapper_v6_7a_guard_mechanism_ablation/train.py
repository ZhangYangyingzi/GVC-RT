"""Matched source and sequence; D/E jointly train P/B/G with one calibrated extra guard."""
import argparse,fcntl,math,random,statistics,traceback
from v67_io import *
from objective import bind

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--calibrate',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();br=ROOT/'branches'/a.branch
    lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cuda:0');assert torch.cuda.device_count()==1
    v4=module('v67_training_v4',V4/'train.py');torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=cfg['force_zero_thres']);im.requires_grad_(False);pm.requires_grad_(False)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    state=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True);assert state['step']==20000
    for k,m in models.items():m.load_state_dict(state[k],strict=True)
    del state
    initial={k:v4.module_hash(m) for k,m in models.items()};compression=v4.compression_hash(im,pm)
    ref=load(ROOT/'audits/source_checkpoint_audit.json')['initialization'];assert initial==ref['module_hashes'] and compression==ref['compression_hash']
    params=[p for m in models.values() for p in m.parameters() if p.requires_grad];ids={id(p) for p in params}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    oc=cfg['optimizer'];optimizer=torch.optim.AdamW([dict(params=[p for p in m.parameters() if p.requires_grad],lr=oc[k+'_lr'],name=k) for k,m in models.items() if any(p.requires_grad for p in m.parameters())],weight_decay=oc['weight_decay'])
    init=dict(status='PASS',branch=a.branch,module_hashes=initial,compression_hash=compression,source_checkpoint_sha256=cfg['source_checkpoint_sha256'],fresh_optimizer=True,trainable={k:any(p.requires_grad for p in m.parameters()) for k,m in models.items()})
    dump(br/'initialization_audit.json',init)
    quality=v4.quality_models(device);plans=load(ROOT/'audits/training_sequence_equivalence.json')['plans'];qps=load(ROOT/'qp_sequence_audit.json')['sequence']
    baseline_rows=[json.loads(x) for x in (V64/'training_logs/vimeo_only.jsonl').read_text().splitlines()]
    def checkplan(plan,step):
        expected=plans[step-1];assert plan['frame_sha256']==baseline_rows[step-1]['frame_sha256'];assert all(plan[k]==expected[k] for k in ('sample_id','start','crop_x','crop_y'));assert qps[step-1]==expected['external_qp']
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    if a.calibrate:
        from calibration import run_calibration
        run_calibration(v4,im,pm,wrapper,quality,cfg,models,initial,compression,optimizer,device)
        return
    cal=load(ROOT/'audits/guard_gradient_calibration.json');assert cal['status']=='PASS'
    weight=cal[a.branch[0]]['lambda'];assert .095<=cal[a.branch[0]]['weighted_over_main']<=.105
    fn=bind(v4,im,pm,wrapper,quality,cfg,weight=weight,branch=a.branch)
    if a.smoke:
        frames,plan=sample_clip(random.Random(cfg['sample_seed']),device,v4);checkplan(plan,1);rows=[]
        for q in (0,9):
            # Verify instrumentation with guard disabled is identical to the original objective and gradients.
            optimizer.zero_grad(set_to_none=True);reference=bind(v4,im,pm,wrapper,quality,cfg,original=True,branch=a.branch);refsum,refq=reference(frames,q)
            grads=[p.grad.detach().clone() if p.grad is not None else None for p in params]
            optimizer.zero_grad(set_to_none=True);zero=bind(v4,im,pm,wrapper,quality,cfg,weight=0.0,branch=a.branch);sums,actual=zero(frames,q)
            assert actual==refq and all(math.isclose(sums[k],z,rel_tol=1e-6,abs_tol=1e-8) for k,z in refsum.items())
            assert all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.allclose(g,p.grad,rtol=1e-5,atol=1e-7)) for p,g in zip(params,grads))
            optimizer.zero_grad(set_to_none=True);sums,actual=fn(frames,q)
            norms={k:v4.grad_norm(m.parameters()) for k,m in models.items()};assert norms['wrapper']>0
            assert all(math.isfinite(x) for x in norms.values());rows.append(dict(QP=q,base_objective_gradient_equivalence=True,**norms,**sums))
        optimizer.zero_grad(set_to_none=True)
        guard_fn=bind(v4,im,pm,wrapper,quality,cfg,weight=weight,branch=a.branch,guard_only=True)
        guard_fn(frames,9)
        guard_norms={k:v4.grad_norm(m.parameters()) for k,m in models.items()}
        assert guard_norms['wrapper']>0 and all(math.isfinite(z) for z in guard_norms.values())
        if a.branch==E:assert all(z>0 for z in guard_norms.values())
        optimizer.zero_grad(set_to_none=True);fn(frames,9)
        torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']);optimizer.step();assert v4.compression_hash(im,pm)==compression
        changed={k:initial[k]!=v4.module_hash(m) for k,m in models.items()};assert changed==dict(wrapper=True,bridge=True,generator=True)
        dump(br/'smoke_audit.json',dict(status='PASS',rows=rows,disposable_step=True,changes=changed,receiver_input_gradient_path_valid=True,extra_guard_gradient_norms=guard_norms));print('SMOKE PASS',a.branch,flush=True);return
    assert load(br/'smoke_audit.json')['status']=='PASS'
    rng=random.Random(cfg['sample_seed']);prior=br/'checkpoint_hashes.json';hashes=load(prior) if prior.exists() else {};start=0
    logpath=ROOT/'training_logs'/f'{a.branch}.jsonl';csvpath=ROOT/'training_logs'/f'{a.branch}.csv';code={p.name:sha(p) for p in ROOT.glob('*.py')}
    if hashes:
        start=max(map(int,hashes));cp=checkpoint(a.branch,start);assert sha(cp)==hashes[str(start)]['sha256'];s=torch.load(cp,map_location='cpu',weights_only=True)
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['lambda_guard']==weight
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        optimizer.load_state_dict(s['optimizer']);rng.setstate(s['sample_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state']);del s
    rows=[json.loads(x) for x in logpath.read_text().splitlines()] if logpath.exists() else []
    if len(rows)>start:
        dump(br/'logs'/f'pre_resume_{time.time_ns()}.json',rows);rows=[r for r in rows if r['step']<=start]
        logpath.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert [r['step'] for r in rows]==list(range(1,start+1))
    if rows:write(csvpath,rows)
    elif csvpath.exists():
        dump(br/'logs'/f'pre_resume_csv_{time.time_ns()}.json',read(csvpath));csvpath.write_text('')
    def save(step):
        assert v4.compression_hash(im,pm)==compression;path=checkpoint(a.branch,step);assert not path.exists()
        s=dict(step=step,branch=a.branch,**{k:m.state_dict() for k,m in models.items()},optimizer=optimizer.state_dict(),initialization_hashes=initial,compression_hash=compression,sample_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),lambda_guard=weight,code_hashes=code,config_sha256=sha(ROOT/'config.json'))
        tmp=path.with_suffix('.tmp');torch.save(s,tmp);tmp.replace(path);hashes[str(step)]=dict(path=str(path),sha256=sha(path),step=step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},compression_hash=compression);dump(prior,hashes)
    if not hashes:save(0)
    begin=time.time()
    with logpath.open('a',buffering=1) as log:
        for step in range(start+1,1001):
            frames,plan=sample_clip(rng,device,v4);checkplan(plan,step);optimizer.zero_grad(set_to_none=True);sums,actual=fn(frames,qps[step-1]);assert actual==baseline_rows[step-1]['actual_qps']
            norms={k:v4.grad_norm(m.parameters()) for k,m in models.items()};assert norms['wrapper']>0 and all(math.isfinite(z) for z in norms.values())
            norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm);optimizer.step()
            pn={k:math.sqrt(sum(float(p.detach().double().square().sum()) for p in m.parameters())) for k,m in models.items()};assert all(math.isfinite(z) for z in pn.values())
            row=dict(branch=a.branch,step=step,update=step,external_qp=qps[step-1],actual_qps=actual,**plan,**sums,total_loss=sums['loss'],R_est_bpp=sums['rate_bpp'],lambda_guard=weight,gradient_norm=norm,**{k+'_gradient_norm':z for k,z in norms.items()},**{k+'_parameter_norm':z for k,z in pn.items()},receiver_parameter_gradient_status='trainable',input_gradient_path_valid=True)
            log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            # Every update is visible in the requested CSV.
            with csvpath.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));
                if step==1:w.writeheader()
                w.writerow(row)
            if step in STEPS:log.flush();os.fsync(log.fileno());save(step)
            if step%10==0 or step==start+1:
                dump(br/'training_status.json',dict(status='RUNNING',update=step,target=1000,pid=os.getpid(),gpu=a.gpu,seconds_per_update=(time.time()-begin)/(step-start),updated_unix=time.time()));print('TRAIN',a.branch,step,'/1000',flush=True)
    final={k:v4.module_hash(m) for k,m in models.items()};changes={k:final[k]!=initial[k] for k in initial};assert changes==dict(wrapper=True,bridge=True,generator=True)
    assert v4.compression_hash(im,pm)==compression
    dump(ROOT/'audits'/f'parameter_update_audit_{a.branch[0]}.json',dict(status='PASS',before=dict(initial,compression_core=compression),after=dict(final,compression_core=compression),changed=dict(changes,compression_core=False),updates=1000))
    frozen();dump(br/'training_status.json',dict(status='PASS',update=1000,target=1000,pid=os.getpid(),gpu=a.gpu,updated_unix=time.time()));print('TRAIN PASS',a.branch,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'training_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
