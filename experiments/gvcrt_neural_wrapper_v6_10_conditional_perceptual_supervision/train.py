"""Three resumable branches, disposable D warmup, atomic full-state checkpoints."""
import argparse,fcntl,random,math,traceback
from v68_io import *
from model_runtime import setup,restore_rng,norm
from objective import Objective,discriminator_update
from discriminator import create

def atomic_torch(path,state):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();assert load(ROOT/'audits/objective_smoke.json')['status']=='PASS'
    br=ROOT/'branches'/a.branch;lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cuda:0');v4,im,pm,models,opt,quality,source=setup(device)
    initial={k:v4.module_hash(m) for k,m in models.items()};params=[p for m in models.values() for p in m.parameters()]
    source_audit=load(ROOT/'audits/source_B1000_audit.json')
    assert initial==source_audit['module_hashes'] and optimizer_summary(opt.state_dict())==source_audit['optimizer']
    init=dict(status='PASS',source_sha256=sha(SOURCE),module_hashes=initial,
        optimizer=optimizer_summary(opt.state_dict()),rng_hashes=source_audit['rng_hashes'],
        plans_sha256=sha(ROOT/'audits/continuation_training_plans.json'))
    dump(br/'initialization_audit.json',init)
    fn=Objective(v4,im,pm,models,quality,cfg);disc=dopt=None
    if a.branch!='C_plain':
        disc=create(cfg['discriminator']['feature_channels'],device,cfg['discriminator']['seed'])
        dopt=torch.optim.Adam(disc.parameters(),lr=1e-4,betas=(0.,.99),weight_decay=0.)
        dump(br/'discriminator_initialization.json',dict(hash=state_hash(disc.state_dict()),architecture=str(disc),
            seed=cfg['discriminator']['seed']))
    code={p.name:sha(p) for p in ROOT.glob('*.py')};plans=load(ROOT/'audits/continuation_training_plans.json')['plans']
    rng=random.Random();rng.setstate(source['sample_rng_state']);start=1000
    def state_at(step):
        return dict(step=step,branch=a.branch,**{k:m.state_dict() for k,m in models.items()},
            optimizer=opt.state_dict(),discriminator=None if disc is None else disc.state_dict(),
            discriminator_optimizer=None if dopt is None else dopt.state_dict(),
            sample_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
            python_rng_state=random.getstate(),training_plan_position=step-1000,
            compression_hash=cfg['compression_hash'],lambda_struct=cfg['lambda_struct'],
            code_hashes=code,config_sha256=sha(ROOT/'config.json'),training_plan_sha256=init['plans_sha256'],
            source_B1000_sha256=init['source_sha256'])
    latest=br/'parts/latest.pt';hashpath=br/'checkpoint_hashes.json'
    index=load(hashpath) if hashpath.exists() else {}
    # Prefer the newest recoverable state, including unindexed atomically saved named checkpoints.
    candidates=[p for p in [latest,*list((br/'checkpoints').glob('step_*.pt'))] if p.exists()]
    if candidates:
        states=[(p,torch.load(p,map_location='cpu',weights_only=True)) for p in candidates]
        path,s=max(states,key=lambda z:z[1]['step']);start=s['step']
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['training_plan_sha256']==init['plans_sha256']
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);rng.setstate(s['sample_rng_state']);restore_rng(s);random.setstate(s['python_rng_state'])
        if disc is not None:disc.load_state_dict(s['discriminator']);dopt.load_state_dict(s['discriminator_optimizer'])
        del states,s
    elif disc is not None:
        warm=br/'parts/warmup.pt';wstart=0;wr=random.Random();wr.setstate(source['sample_rng_state']);wrows=[]
        if warm.exists():
            w=torch.load(warm,map_location='cpu',weights_only=True);assert w['code_hashes']==code
            disc.load_state_dict(w['discriminator']);dopt.load_state_dict(w['optimizer']);wr.setstate(w['sample_rng_state'])
            wstart=w['step'];wrows=w['rows']
        for k in range(wstart,100):
            frames,plan=sample_clip(wr,device,v4);expected=plans[k]
            assert all(plan[x]==expected[x] for x in ('sample_id','start','crop_x','crop_y','frame_sha256'))
            with torch.random.fork_rng(devices=[0]):records=fn.collect(frames,expected['external_qp'])
            row=discriminator_update(disc,dopt,records,a.branch,expected['external_qp'])
            wrows.append(dict(warmup_step=k+1,**expected,**row))
            if (k+1)%10==0:
                atomic_torch(warm,dict(step=k+1,discriminator=disc.state_dict(),optimizer=dopt.state_dict(),
                    sample_rng_state=wr.getstate(),rows=wrows,code_hashes=code))
                write(br/'logs/discriminator_warmup.csv',wrows)
                dump(br/'training_status.json',dict(status='RUNNING',phase='discriminator_warmup',warmup_step=k+1,target=100,gpu=a.gpu,PID=os.getpid(),updated_unix=time.time()))
                print('WARMUP',a.branch,k+1,flush=True)
        assert initial=={k:v4.module_hash(m) for k,m in models.items()}
        assert optimizer_summary(opt.state_dict())==source_audit['optimizer']
        assert v4.compression_hash(im,pm)==cfg['compression_hash']
        dump(br/'warmup_integrity.json',dict(status='PASS',updates=100,generator_weights_unchanged=True,optimizer_unchanged=True))
        fn.records=[];del records,frames;restore_rng(source)
    else:restore_rng(source)
    if disc is not None:disc.eval().requires_grad_(False)
    logs=ROOT/'training_logs'/f'{a.branch}.jsonl'
    rows=[json.loads(x) for x in logs.read_text().splitlines()] if logs.exists() else []
    if any(r['absolute_step']>start for r in rows):
        dump(br/'logs'/f'pre_resume_{time.time_ns()}.json',rows)
        rows=[r for r in rows if r['absolute_step']<=start]
        logs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert [r['absolute_step'] for r in rows]==list(range(1001,start+1))
    csvpath=ROOT/'training_logs'/f'{a.branch}.csv'
    if rows:write(csvpath,rows)
    begin=time.time()
    def save(step):
        assert v4.compression_hash(im,pm)==cfg['compression_hash']
        state=state_at(step);atomic_torch(latest,state)
        if step in (1250,1500,2000):
            path=checkpoint(a.branch,step)
            if not path.exists():atomic_torch(path,state)
            index[str(step)]=dict(path=str(path),sha256=sha(path),step=step,
                module_hashes={k:v4.module_hash(m) for k,m in models.items()},compression_hash=cfg['compression_hash'])
            if step in (1500,2000):
                inference=ROOT/'evaluation/checkpoints'/f'{tag(a.branch)}{step}.pt'
                if not inference.exists():
                    atomic_torch(inference,{k:m.state_dict() for k,m in models.items()})
                index[str(step)]['inference']=dict(path=str(inference),sha256=sha(inference),module_hashes=index[str(step)]['module_hashes'],step=step)
            dump(hashpath,index)
    with logs.open('a',buffering=1) as stream:
        for step in range(start+1,2001):
            k=step-1000;expected=plans[k-1];frames,plan=sample_clip(rng,device,v4)
            assert all(plan[x]==expected[x] for x in ('sample_id','start','crop_x','crop_y','temporal_indices','frame_sha256'))
            opt.zero_grad(set_to_none=True)
            dr=dict(L_D=0.,real_score=0.,fake_score=0.,discriminator_gradient_norm=0.,discriminator_lr=0.)
            if disc is not None:
                with torch.random.fork_rng(devices=[0]):records=fn.collect(frames,expected['external_qp'])
                dr=discriminator_update(disc,dopt,records,a.branch,expected['external_qp'])
                fn.records=[];del records
            weight=0. if disc is None else .01*min(k/100,1.)
            measured=k in (1,100,500,1000)
            dh=state_hash(disc.state_dict()) if measured and disc is not None else None
            sums,actual=fn(frames,expected['external_qp'],a.branch,disc,weight,measured)
            assert actual==expected['actual_qps']
            norms={n:v4.grad_norm(m.parameters()) for n,m in models.items()}
            assert all(math.isfinite(v) and v>0 for v in norms.values())
            assert disc is None or all(p.grad is None for p in disc.parameters())
            total=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(total)
            opt.step()
            if measured:
                if disc is not None:assert state_hash(disc.state_dict())==dh
                dump(br/'parts'/f'gradient_components_{k}.json',dict(joint_update=k,measurements=fn.gradient_measurements,lambda_adv=weight,extra_optimizer_updates=0))
            row=dict(branch=a.branch,**expected,**sums,**dr,
                **{n+'_gradient_norm':v for n,v in norms.items()},
                **{'current_'+g['name']+'_lr':g['lr'] for g in opt.param_groups})
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            with csvpath.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if step==1001:writer.writeheader()
                writer.writerow(row)
            if k%50==0:stream.flush();os.fsync(stream.fileno());save(step)
            if k==1 or k%5==0:
                dump(br/'training_status.json',dict(status='RUNNING',phase='joint_training',absolute_step=step,continuation_step=k,target=2000,gpu=a.gpu,PID=os.getpid(),seconds_per_update=(time.time()-begin)/(step-start),updated_unix=time.time()))
                print('TRAIN',a.branch,step,flush=True)
    save(2000)
    assert v4.compression_hash(im,pm)==cfg['compression_hash']
    dump(br/'training_status.json',dict(status='PASS',absolute_step=2000,continuation_step=1000,gpu=a.gpu,PID=os.getpid(),updated_unix=time.time()))
    dump(br/'training_integrity.json',dict(status='PASS',compression_core_frozen=True,updates=1000,module_hashes={k:v4.module_hash(m) for k,m in models.items()}))
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'logs/failures'/f'train_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
