"""Exactly 5000 plan-indexed updates, complete restoration, fixed validation."""
import argparse,fcntl,random,math,traceback
import torch
from v68_io import *
from io_utils import command
from model_runtime import setup,core_parameters,norm,precision,inference_state,atomic_torch
from objective import Objective
from data import sample
def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2);cfg=frozen();assert load(ROOT/'audits/objective_smoke.json')['status']=='PASS'
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    br=ROOT/'branches'/a.branch;lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cuda:0');v4,im,pm,models,opt,quality,teacher=setup(device)
    core=core_parameters(pm,models);wb=[p for k in ('wrapper','bridge') for p in models[k].parameters()]
    constant=dict(I=v4.module_hash(im),generator=v4.module_hash(models['generator']),teacher=v4.module_hash(teacher),quality=[v4.module_hash(m) for m in quality])
    code={p.name:sha(p) for p in ROOT.glob('*.py')};plans=load(ROOT/'manifests/training_plans.json')['plans'];ph=sha(ROOT/'manifests/training_plans.json')
    init=dict(status='PASS',native_checkpoints=cfg['init_checkpoints'],fresh_optimizer=True,optimizer_initial_state_entries=len(opt.state),module_hashes={k:v4.module_hash(m) for k,m in models.items()},core_hash=state_hash({n:p.detach() for n,p in core.items()}),optimizer_settings=optimizer_summary(opt.state_dict()),RNG=dict(torch=state_hash(torch.get_rng_state()),cuda=state_hash(torch.cuda.get_rng_state()),python=state_hash(random.getstate())))
    dump(br/'initialization_audit.json',init)
    fn=Objective(v4,im,pm,models,quality,teacher,cfg['lambda_cosine'][a.branch])
    latest=br/'parts/latest.pt';index=load(br/'checkpoint_hashes.json') if (br/'checkpoint_hashes.json').exists() else {};start=0
    candidates=[f for f in [latest,*list((br/'checkpoints').glob('step_*.pt'))] if f.exists()]
    if candidates:
        path=max(candidates,key=lambda p:p.stat().st_mtime_ns);s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['training_plan_sha256']==ph
        pm.load_state_dict(s['full_p_master'],strict=True);models['wrapper'].load_state_dict(s['wrapper'],strict=True);opt.load_state_dict(s['optimizer']);start=s['step']
        torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state']);random.setstate(s['python_rng_state']);del s
    def unchanged():
        assert constant==dict(I=v4.module_hash(im),generator=v4.module_hash(models['generator']),teacher=v4.module_hash(teacher),quality=[v4.module_hash(m) for m in quality])
    def save(step):
        unchanged();prec=precision(im,pm,models,v4)
        s=dict(step=step,optimizer_update_count=step,branch=a.branch,**{k:m.state_dict() for k,m in models.items()},full_p_master=pm.state_dict(),fp32_core_master={n:p for n,p in core.items()},optimizer=opt.state_dict(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),python_rng_state=random.getstate(),plan_position=step,training_plan_sha256=ph,config_sha256=sha(ROOT/'config.json'),code_hashes=code,frozen_hashes=constant,**prec)
        atomic_torch(latest,s)
        if step in cfg['checkpoint_steps']:
            path=checkpoint(a.branch,step)
            if not path.exists():atomic_torch(path,s)
            inf=ROOT/'evaluation/checkpoints'/f'{tag(a.branch)}{step}.pt'
            if not inf.exists():atomic_torch(inf,inference_state(im,pm,models,v4))
            meta=dict(path=str(path),sha256=sha(path),step=step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},**prec)
            meta['inference']=dict(path=str(inf),sha256=sha(inf),step=step,module_hashes=meta['module_hashes'],**prec)
            index[str(step)]=meta;dump(br/'checkpoint_hashes.json',index)
    validation=ROOT/'validation'/a.branch;validation.mkdir(parents=True,exist_ok=True)
    def validate(step):
        output=validation/f'step_{step}.json'
        if output.exists():assert load(output)['status']=='PASS';return
        before=state_hash(opt.state_dict());rows=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            for plan in load(ROOT/'manifests/validation_manifest.json')['samples']:
                frames,hashes=sample(plan,device)
                for q in (0,4,9):
                    values,actual=fn(frames,q,backward=False)
                    rows.append(dict(branch=a.branch,step=step,domain=plan['domain'],sample_index=plan['index'],external_qp=q,actual_qps=actual,frame_rgb_sha256=hashes,**values))
        assert before==state_hash(opt.state_dict());unchanged()
        dump(output,dict(status='PASS',protocol='frozen independent 32 clips; estimated training objective, not final quality selection',optimizer_steps=0,rows=rows));write(output.with_suffix('.csv'),rows)
        merged=[]
        for p in sorted(validation.glob('step_*.json'),key=lambda p:int(p.stem.split('_')[1])):merged.extend(load(p)['rows'])
        write(ROOT/'results'/f'validation_{tag(a.branch)}.csv',merged)
    logpath=ROOT/'training_logs'/f'{a.branch}.jsonl';csvpath=logpath.with_suffix('.csv')
    rows=[json.loads(x) for x in logpath.read_text().splitlines()] if logpath.exists() else []
    if any(r['step']>start for r in rows):
        dump(br/'logs'/f'pre_resume_{time.time_ns()}.json',rows);rows=[r for r in rows if r['step']<=start]
        temp=logpath.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(logpath)
    assert [r['step'] for r in rows]==list(range(1,start+1))
    plan_fields=list(dict.fromkeys(k for p in plans for k in p))
    csv_fields=None
    if rows:
        csv_fields=['branch',*plan_fields,*[k for k in rows[0] if k not in plan_fields and k!='branch']]
        from io_utils import write as csv_write
        csv_write(csvpath,rows,fields=csv_fields)
    if start==0:save(0)
    if start in cfg['checkpoint_steps']:validate(start)
    begin=time.time()
    with logpath.open('a',buffering=1) as stream:
        for step in range(start+1,5001):
            tic=time.time();plan=plans[step-1];assert plan['step']==step;frames,hashes=sample(plan,device);measured=step in cfg['gradient_check_steps']
            before={n:p.detach().cpu().clone() for n,p in core.items()} if measured else None
            opt.zero_grad(set_to_none=True);values,actual=fn(frames,plan['external_qp'],measure=measured);assert actual==plan['actual_qps']
            norms={k:norm([p.grad for p in m.parameters()]) for k,m in models.items()}
            wb_norm=float(torch.nn.utils.clip_grad_norm_(wb,1.));core_norm=float(torch.nn.utils.clip_grad_norm_(list(core.values()),1.))
            assert all(math.isfinite(n) for n in [wb_norm,core_norm,*norms.values()]);assert norms['generator']==0.
            assert all(all(v['finite'] for v in frame.values()) for frame in fn.cosine_gradient_frames)
            opt.step()
            assert all(v.dtype==torch.float32 for ps in opt.state.values() for k,v in ps.items() if torch.is_tensor(v))
            if measured:
                unchanged();changes=[n for n,p in core.items() if not torch.equal(p.detach().cpu(),before[n])]
                dump(ROOT/'audits/gradient_checks'/f'{a.branch}_{step}.json',dict(status='PASS',step=step,observations=fn.observations,core_changed=changes,generator_frozen=True,teacher_frozen=True,extra_optimizer_steps=0,**precision(im,pm,models,v4)));del before
            row=dict(branch=a.branch,**plan,**values,frame_rgb_sha256=hashes,optimizer_update_count=step,gradients=norms,cosine_gradient_per_P_frame=fn.cosine_gradient_frames,wrapper_bridge_preclip_norm=wb_norm,compression_core_preclip_norm=core_norm,wrapper_bridge_clip_factor=min(1.,1/(wb_norm+1e-6)),core_clip_factor=min(1.,1/(core_norm+1e-6)),learning_rates={g['name']:g['lr'] for g in opt.param_groups},finite=True,elapsed_seconds=time.time()-tic)
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float));stream.write(json.dumps(row,allow_nan=False)+'\n')
            with csvpath.open('a',newline='') as f:
                if csv_fields is None:csv_fields=['branch',*plan_fields,*[k for k in row if k not in plan_fields and k!='branch']]
                w=csv.DictWriter(f,fieldnames=csv_fields)
                if step==1:w.writeheader()
                w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
            if step%50==0:stream.flush();os.fsync(stream.fileno());save(step)
            if step in cfg['checkpoint_steps']:validate(step)
            if step==1 or step%10==0:
                dump(br/'training_status.json',dict(status='RUNNING',step=step,target=5000,gpu=a.gpu,PID=os.getpid(),seconds_per_update=(time.time()-begin)/(step-start),updated_unix=time.time()));print('TRAIN',a.branch,step,flush=True)
    save(5000);validate(5000);unchanged()
    dump(br/'training_status.json',dict(status='PASS',step=5000,optimizer_updates=5000,gpu=a.gpu,PID=os.getpid()))
    dump(br/'training_integrity.json',dict(status='PASS',updates=5000,frozen_hashes=constant,teacher_not_deployed=True,**precision(im,pm,models,v4)))
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/failures'/f'train_{os.getpid()}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
