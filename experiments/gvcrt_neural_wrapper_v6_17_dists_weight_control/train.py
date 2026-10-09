"""One-factor DISTS weight 0.5 control: exact V6.16 replay records."""
import argparse,fcntl,math,random,traceback
from io17 import *
STEPS=(0,250,500,1000)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--initialize-only',action='store_true');a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);frozen();cfg=load(ROOT/'config.json');gate=load(ROOT/'audits/weight_check.json')
    assert gate['status']=='PASS' and gate['optimizer_updates']==0
    if not a.initialize_only:assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS'
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    lock=(ROOT/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'))
    initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    assert core==evalcfg()['compression_hash']
    params=[p for m in models.values() for p in m.parameters()];oc=cfg['optimizer']
    opt=torch.optim.AdamW([dict(name=k,params=list(m.parameters()),lr=oc[k+'_lr']) for k,m in models.items()],weight_decay=oc['weight_decay']);assert not opt.state
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    code={n:sha(ROOT/n) for n in ('train.py','io17.py','gradient_tools.py')};plan=replay_rows();assert len(plan)==1000
    latest=ROOT/'checkpoints/latest.pt';start=0;index=load(ROOT/'checkpoint_index.json') if (ROOT/'checkpoint_index.json').exists() else {}
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True)
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['replay_plan_sha256']==sha(ROOT/'replay_plan.json') and s['source_checkpoint_sha256']==EXPECTED
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);random.setstate(s['python_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state'])
        start=s['adaptation_step'];assert s['replay_cursor']==start;del s
    def save(step):
        assert core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
        state=dict(schema='v617',source_step=1000,adaptation_step=step,optimizer_update_count=step,replay_cursor=step,source_checkpoint_sha256=EXPECTED,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),python_rng_state=random.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),compression_hash=core,quality_hashes=qh,config_sha256=sha(ROOT/'config.json'),replay_plan_sha256=sha(ROOT/'replay_plan.json'),code_hashes=code)
        atomic_torch(latest,state)
        if step in STEPS:
            cp=cp_path(step);export=ROOT/'evaluation/checkpoints'/f'dists_w05_{step}.pt'
            if not cp.exists():atomic_torch(cp,state)
            if not export.exists():atomic_torch(export,{k:m.state_dict() for k,m in models.items()})
            hashes={k:v4.module_hash(m) for k,m in models.items()}
            index[str(step)]=dict(path=str(cp),sha256=sha(cp),source_step=1000,adaptation_step=step,module_hashes=hashes,compression_hash=core,inference=dict(path=str(export),sha256=sha(export),module_hashes=hashes))
            dump(ROOT/'checkpoint_index.json',index)
    dump(ROOT/'audits/initialization.json',dict(status='PASS',source_step=1000,adaptation_step=0,source_sha256=EXPECTED,module_hashes=initial,compression_hash=core,fresh_AdamW=True,source_optimizer_restored=False,quality_frozen_eval=True,quality_hashes=qh,trainable_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},only_change='DISTS weight 1.0 to 0.5; require_grad=True unchanged'))
    log=ROOT/'training_logs/dists_w05.jsonl';log.parent.mkdir(exist_ok=True)
    rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if any(r['adaptation_step']>start for r in rows):
        dump(ROOT/'logs'/f'pre_resume_rows_{time.time_ns()}.json',rows);rows=[r for r in rows if r['adaptation_step']<=start]
        temp=log.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(log)
    assert [r['adaptation_step'] for r in rows]==list(range(1,start+1))
    if rows:write(log.with_suffix('.csv'),rows)
    dump(ROOT/'audits/resume_state.json',dict(status='PASS',start_step=start,optimizer_restored=start>0,RNG_restored=start>0,replay_cursor=start,random_sampling=False))
    if start==0:save(0)
    if a.initialize_only:print('INITIALIZATION PASS',flush=True);return
    run=objective(v4,im,pm,models,quality,True);tic=time.time()
    with log.open('a',buffering=1) as stream:
        for step in range(start+1,1001):
            t=time.time();record=plan[step-1];assert record['adaptation_step']==step
            frames,sample=replay(record,torch.device('cuda:0'));opt.zero_grad(set_to_none=True);q=record['external_qp'];sums,actual=run(frames,q)
            assert actual==record['actual_qps']
            from gradient_tools import stats
            gs={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()}
            assert all(g['finite'] and g['effective_parameters']>0 for g in gs.values())
            norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm) and norm>0 and all(math.isfinite(x) for x in sums.values())
            sums['weighted_DISTS']=cfg['lambda_dists']*sums['DISTS']
            clip_coefficient=min(1.0,oc['grad_clip']/(norm+1e-6))
            opt.step();assert all(bool(torch.isfinite(p).all()) for p in params)
            row=dict(branch='dists_w05',source_step=1000,adaptation_step=step,optimizer_update_count=step,external_qp=q,actual_qps=actual,**sample,**sums,PBG_gradients=gs,gradient_norm=norm,global_preclip_norm=norm,global_clip_coefficient=clip_coefficient,lambda_dists=cfg['lambda_dists'],learning_rates={g['name']:g['lr'] for g in opt.param_groups},elapsed_seconds=time.time()-t,all_finite=True,input_crop_hashes_match=True,actual_qps_match=True)
            stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            if step%25==0 or step==1000:
                stream.flush();os.fsync(stream.fileno());save(step);write(log.with_suffix('.csv'),rows)
            if step==1 or step%10==0:
                dump(ROOT/'training_status.json',dict(status='RUNNING',PID=os.getpid(),gpu=a.gpu,source_step=1000,adaptation_step=step,target=1000,seconds_per_update=(time.time()-tic)/(step-start),updated_unix=time.time()));print('TRAIN',step,'/1000',flush=True)
    assert len(rows)==1000
    for old,new in zip(plan,rows):
        assert all(old[k]==new[k] for k in ('domain','video','source_frame_indices','canonical_indices','crop_x','crop_y','crop_size','cropped_RGB_sha256','external_qp','actual_qps'))
    dump(ROOT/'training_integrity.json',dict(status='PASS',updates=1000,source_step=1000,exact_replay=True,fresh_AdamW=True,frozen_compression_unchanged=True,quality_frozen_unchanged=True,compression_hash=core,quality_hashes=qh,checkpoint_steps=STEPS,missing=[]))
    dump(ROOT/'training_status.json',dict(status='PASS',adaptation_step=1000,target=1000,source_step=1000,PID=os.getpid()))
    dump(ROOT/'frozen_checkpoint_index.json',dict(status='PASS',checkpoints=index,checkpoint_index_sha256=sha(ROOT/'checkpoint_index.json'),fixed_evaluation_step=1000,no_checkpoint_selection=True))
    print('TRAINING PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'logs'/f'failure_train_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
