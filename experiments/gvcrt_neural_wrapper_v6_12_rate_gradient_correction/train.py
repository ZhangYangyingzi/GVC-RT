"""Exact paired continuation, FP32 core optimizer, separately clipped parameter groups."""
import argparse,fcntl,random,math,traceback
from v68_io import *
from model_runtime import setup,restore_rng,norm,core_parameters,precision_metadata,inference_state
from objective import Objective

def atomic_torch(path,state):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def gradients(core):
    out={}
    for n,p in core.items():
        key=n.split('.')[0];r=out.setdefault(key,dict(parameters=0,active=0,nonzero=0,grad_squared=0.,inactive_names=[]))
        r['parameters']+=1
        if p.grad is None:r['inactive_names'].append(n)
        else:
            r['active']+=1;v=float(p.grad.detach().double().square().sum());r['grad_squared']+=v;r['nonzero']+=int(v>0)
    for r in out.values():r['gradient_norm']=math.sqrt(r.pop('grad_squared'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();assert load(ROOT/'audits/objective_smoke.json')['status']=='PASS'
    br=ROOT/'branches'/a.branch;lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cuda:0');joint=a.branch=='J_scale_ste'
    v4,im,pm,models,opt,quality,source=setup(device,joint=joint)
    core=core_parameters(pm,models);params=[p for m in models.values() for p in m.parameters()]
    initial={k:v4.module_hash(m) for k,m in models.items()};ih=v4.module_hash(im);qh=[v4.module_hash(m) for m in quality]
    code={p.name:sha(p) for p in ROOT.glob('*.py')};plans=load(ROOT/'audits/continuation_training_plans.json')['plans']
    init=dict(status='PASS',source_sha256=sha(SOURCE),module_hashes=initial,plans_sha256=sha(ROOT/'audits/continuation_training_plans.json'),source_optimizer_hash=state_hash(source['optimizer']),rng_hashes={k:state_hash(source[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')})
    dump(br/'initialization_audit.json',init)
    fn=Objective(v4,im,pm,models,quality,cfg);rng=random.Random();rng.setstate(source['sample_rng_state']);start=1000
    latest=br/'parts/latest.pt';hashpath=br/'checkpoint_hashes.json';index=load(hashpath) if hashpath.exists() else {}
    candidates=[x for x in [latest,*list((br/'checkpoints').glob('step_*.pt'))] if x.exists()]
    if candidates:
        # File names are monotonic; latest is atomically persisted every 50 updates.
        selected=max(candidates,key=lambda x:x.stat().st_mtime_ns);s=torch.load(selected,map_location='cpu',weights_only=True);start=s['step']
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['training_plan_sha256']==init['plans_sha256']
        pm.load_state_dict(s['full_p_master'],strict=True)
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);rng.setstate(s['sample_rng_state']);restore_rng(s);random.setstate(s['python_rng_state']);del s
    else:restore_rng(source)
    del source
    end=cfg['absolute_steps'][1]
    logs=ROOT/'training_logs'/f'{a.branch}.jsonl';csvpath=logs.with_suffix('.csv')
    rows=[json.loads(x) for x in logs.read_text().splitlines()] if logs.exists() else []
    if any(r['absolute_step']>start for r in rows):
        dump(br/'logs'/f'pre_resume_{time.time_ns()}.json',rows);rows=[r for r in rows if r['absolute_step']<=start]
        logs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert [r['absolute_step'] for r in rows]==list(range(1001,start+1))
    if rows:write(csvpath,rows)
    def save(step):
        assert v4.module_hash(im)==ih and [v4.module_hash(m) for m in quality]==qh
        precision=precision_metadata(im,pm,models,v4)
        if not joint:assert precision['compression_hash']==cfg['compression_hash']
        state=dict(step=step,branch=a.branch,**{k:m.state_dict() for k,m in models.items()},full_p_master=pm.state_dict(),
            fp32_core_master={n:p for n,p in core.items()},optimizer=opt.state_dict(),sample_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),python_rng_state=random.getstate(),training_plan_position=step-1000,
            **precision,lambda_struct=cfg['lambda_struct'],code_hashes=code,config_sha256=sha(ROOT/'config.json'),training_plan_sha256=init['plans_sha256'],source_B1000_sha256=init['source_sha256'],I_checkpoint=str(REPO/'checkpoints/GVC-RT_I.pt'))
        atomic_torch(latest,state)
        if step in cfg['checkpoint_steps']:
            path=checkpoint(a.branch,step)
            if not path.exists():atomic_torch(path,state)
            meta=dict(path=str(path),sha256=sha(path),step=step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},**precision)
            if step in cfg['evaluation_steps']:
                inf=ROOT/'evaluation/checkpoints'/f'{tag(a.branch)}{step}.pt'
                if not inf.exists():atomic_torch(inf,inference_state(im,pm,models,v4,joint=joint))
                meta['inference']=dict(path=str(inf),sha256=sha(inf),step=step,module_hashes=meta['module_hashes'],**precision)
            index[str(step)]=meta;dump(hashpath,index)
    begin=time.time()
    with logs.open('a',buffering=1) as stream:
        for step in range(start+1,end+1):
            tic=time.time();k=step-1000;expected=plans[k-1];torch.cuda.reset_peak_memory_stats()
            frames,plan=sample_clip(rng,device,v4)
            assert all(plan[x]==expected[x] for x in ('sample_id','start','crop_x','crop_y','temporal_indices','frame_sha256'))
            assert plan['source_frame_rgb_sha256']==load(ROOT/'audits/training_rgb_manifest.json')['plans'][k-1]['source_frame_rgb_sha256']
            measured=k in cfg['gradient_check_steps']
            before={n:p.detach().cpu().clone() for n,p in core.items()} if measured else None
            opt.zero_grad(set_to_none=True);sums,actual=fn(frames,expected['external_qp'],measure=measured)
            assert actual==expected['actual_qps']
            norms={n:norm([p.grad for p in m.parameters()]) for n,m in models.items()}
            assert all(math.isfinite(v) and v>0 for v in norms.values())
            detail=gradients(core) if measured else None
            total=float(torch.nn.utils.clip_grad_norm_(params,1.));cn=float(torch.nn.utils.clip_grad_norm_(list(core.values()),1.)) if joint else 0.
            assert math.isfinite(total) and math.isfinite(cn) and (not joint or cn>0)
            opt.step()
            if joint:
                assert all(p.dtype==torch.float32 and all(v.dtype==torch.float32 for v in opt.state[p].values() if torch.is_tensor(v)) for p in core.values() if p in opt.state)
            if measured:
                changes={}
                for n,p in core.items():
                    x=p.detach().cpu();old=before[n];key=n.split('.')[0]
                    r=changes.setdefault(key,dict(master_changed=0,deployment_changed=0,max_master_delta=0.))
                    r['master_changed']+=int(not torch.equal(x,old));r['deployment_changed']+=int(not torch.equal(x.half(),old.half()));r['max_master_delta']=max(r['max_master_delta'],float((x.float()-old.float()).abs().max()))
                assert v4.module_hash(im)==ih and [v4.module_hash(m) for m in quality]==qh
                dump(ROOT/'audits/gradient_checks'/f'{a.branch}_{k}.json',dict(status='PASS',continuation_step=k,modules=detail,updates=changes,PBG_gradient_norms=norms,loss_components=fn.gradient_measurements,PBG_preclip_norm=total,core_preclip_norm=cn,PBG_clip_factor=min(1.,1/(total+1e-6)),core_clip_factor=min(1.,1/(cn+1e-6)),I_and_quality_unchanged=True,extra_optimizer_updates=0,**precision_metadata(im,pm,models,v4)))
                del before
            row=dict(branch=a.branch,**expected,**sums,**{n+'_gradient_norm':v for n,v in norms.items()},**{'current_'+g['name']+'_lr':g['lr'] for g in opt.param_groups},PBG_preclip_norm=total,core_preclip_norm=cn,PBG_clip_factor=min(1.,1/(total+1e-6)),core_clip_factor=min(1.,1/(cn+1e-6)),elapsed_seconds=time.time()-tic,peak_reserved=torch.cuda.max_memory_reserved(),finite=True,optimizer_update_count=k)
            row['source_frame_rgb_sha256']=plan['source_frame_rgb_sha256']
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float));stream.write(json.dumps(row,allow_nan=False)+'\n')
            with csvpath.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if step==1001:w.writeheader()
                w.writerow({n:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for n,v in row.items()})
            if k%50==0:stream.flush();os.fsync(stream.fileno());save(step)
            if k==1 or k%5==0:
                dump(br/'training_status.json',dict(status='RUNNING',absolute_step=step,continuation_step=k,target=end,gpu=a.gpu,PID=os.getpid(),seconds_per_update=(time.time()-begin)/(step-start),updated_unix=time.time()));print('TRAIN',a.branch,step,flush=True)
    save(end)
    dump(br/'training_status.json',dict(status='PASS',absolute_step=end,continuation_step=cfg['updates'],gpu=a.gpu,PID=os.getpid(),updated_unix=time.time()))
    dump(br/'training_integrity.json',dict(status='PASS',compression_core_frozen=not joint,updates=cfg['updates'],I_and_quality_unchanged=True,**precision_metadata(im,pm,models,v4)))
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'logs/failures'/f'train_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
