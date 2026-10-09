"""Resumable single-card branch. One clip/step for the full 15-P-frame budget."""
import argparse,fcntl,math,random,traceback
from io20 import *
from window_training import replay_rows,replay,make_runner
STEPS=(0,250,500,1000)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--branch',choices=BRANCHES,required=True);ap.add_argument('--initialize-only',action='store_true');a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);assert torch.cuda.device_count()==1;frozen();cfg=load(ROOT/'config.json');branch=ROOT/'branches'/a.branch;branch.mkdir(parents=True,exist_ok=True)
    if not a.initialize_only:
        assert load(ROOT/'audits/implementation_check.json')['status']=='PASS';assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS'
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]]);lock=(branch/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality];assert core==evalcfg()['compression_hash']
    params=[p for m in models.values() for p in m.parameters()];oc=cfg['optimizer'];opt=torch.optim.AdamW([dict(name=k,params=list(m.parameters()),lr=oc[k+'_lr']) for k,m in models.items()],weight_decay=oc['weight_decay']);assert not opt.state
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    code={n:sha(ROOT/n) for n in ('train.py','window_training.py','io20.py','gradient_tools.py')};plan=replay_rows(a.branch);planpath=ROOT/'plans/AB.json';assert len(plan)==1000
    latest=branch/'checkpoints/latest.pt';start=0;index=load(branch/'checkpoint_index.json') if (branch/'checkpoint_index.json').exists() else {}
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True);assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['plan_sha256']==sha(planpath) and s['source_checkpoint_sha256']==EXPECTED and s['branch']==a.branch
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);random.setstate(s['python_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state']);start=s['adaptation_step'];assert s['replay_cursor']==start;del s
    def save(step):
        assert core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
        state=dict(schema='v620',branch=a.branch,source_experiment='v618_B',source_step=1000,adaptation_step=step,optimizer_update_count=step,replay_cursor=step,supervised_P_frames=step*15,source_checkpoint_sha256=EXPECTED,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),python_rng_state=random.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),compression_hash=core,quality_hashes=qh,config_sha256=sha(ROOT/'config.json'),plan_sha256=sha(planpath),code_hashes=code)
        atomic_torch(latest,state)
        if step in STEPS:
            cp=branch/'checkpoints'/f'adaptation_step_{step:04d}.pt';export=ROOT/'evaluation/checkpoints'/f'{a.branch}_{step}.pt'
            if not cp.exists():atomic_torch(cp,state)
            if not export.exists():atomic_torch(export,{k:m.state_dict() for k,m in models.items()})
            hashes={k:v4.module_hash(m) for k,m in models.items()};index[str(step)]=dict(path=str(cp),sha256=sha(cp),source_step=1000,adaptation_step=step,module_hashes=hashes,compression_hash=core,inference=dict(path=str(export),sha256=sha(export),module_hashes=hashes));dump(branch/'checkpoint_index.json',index)
    dump(branch/'initialization_audit.json',dict(status='PASS',source_sha256=EXPECTED,module_hashes=initial,compression_hash=core,quality_hashes=qh,fresh_AdamW=True,source_optimizer_restored=False,branch_label='continuous16',i_frame_mode=a.branch))
    log=branch/'training_logs/train.jsonl';log.parent.mkdir(exist_ok=True);rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if any(r['adaptation_step']>start for r in rows):
        dump(branch/'logs'/f'pre_resume_rows_{time.time_ns()}.json',rows);rows=[r for r in rows if r['adaptation_step']<=start];temp=log.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(log)
    assert [r['adaptation_step'] for r in rows]==list(range(1,start+1))
    if start==0:save(0)
    if a.initialize_only:print('INITIALIZATION PASS',a.branch,flush=True);return
    run=make_runner(v4,im,pm,models,quality);tic=time.time();torch.cuda.reset_peak_memory_stats()
    from gradient_tools import stats
    with log.open('a',buffering=1) as stream:
        for step in range(start+1,1001):
            t=time.time();record=plan[step-1];assert record['adaptation_step']==step;frames=replay(record,a.branch,torch.device('cuda:0'));opt.zero_grad(set_to_none=True)
            sums,actual,trace=run(frames,record['external_qp'],a.branch,True);assert actual==record['actual_qps'] and len(trace)==15
            gs={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(g['finite'] and g['effective_parameters']>0 for g in gs.values())
            norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm) and norm>0 and all(math.isfinite(x) for x in sums.values());coefficient=min(1.,oc['grad_clip']/(norm+1e-6));opt.step();assert all(bool(torch.isfinite(p).all()) for p in params)
            row=dict(record,branch=a.branch,source_step=1000,optimizer_update_count=step,**sums,PBG_gradients=gs,global_preclip_norm=norm,global_clip_coefficient=coefficient,supervision_frames=15,reference_resets=[0],frame_trace=trace,learning_rates={g['name']:g['lr'] for g in opt.param_groups},gpu=a.gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.time()-t,all_finite=True,input_crop_hashes_match=True,actual_qps_match=True,exceptions=[])
            stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row);stream.flush();os.fsync(stream.fileno());save(step)
            if step%25==0:write(log.with_suffix('.csv'),rows)
            if step==1 or step%10==0:
                dump(branch/'training_status.json',dict(status='RUNNING',PID=os.getpid(),gpu=a.gpu,adaptation_step=step,target=1000,supervised_P_frames=15*step,seconds_per_update=(time.time()-tic)/(step-start),peak_memory_MiB=row['peak_memory_MiB'],updated_unix=time.time()));print('TRAIN',a.branch,step,'/1000',flush=True)
    assert len(rows)==1000 and sum(r['supervision_frames'] for r in rows)==15000;write(log.with_suffix('.csv'),rows)
    dump(branch/'training_integrity.json',dict(status='PASS',branch=a.branch,updates=1000,supervised_P_frames=15000,source_step=1000,exact_plan=True,fresh_AdamW=True,frozen_compression_unchanged=True,quality_frozen_unchanged=True,compression_hash=core,quality_hashes=qh,checkpoint_steps=STEPS,missing=[]));dump(branch/'training_status.json',dict(status='PASS',adaptation_step=1000,target=1000,supervised_P_frames=15000,PID=os.getpid()));print('TRAINING PASS',a.branch,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'failure_train_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
