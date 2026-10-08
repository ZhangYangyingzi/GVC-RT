"""5000 UVG-only updates using the exact historical nested run_clip AST."""
import argparse,ast,math,random,fcntl,textwrap,traceback
from v614_io import *
def objective(v4,im,pm,models,quality,cfg):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(V62));common=module('v614_fullqp_common',V62/'fullqp_common.py')
    tree=ast.parse((V62/'fullqp_train.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='run_clip')
    # Only redirect the old QP audit read; every objective/STE/DPB operation is unchanged.
    def historical_load(p):
        if Path(p).name=='qp_train_eval_semantics_audit.json':return load(V62/'qp_train_eval_semantics_audit.json')
        return load(p)
    ns=dict(torch=torch,F=F,math=math,v4=v4,im=im,pm=pm,wrapper=models['wrapper'],quality=quality,cfg=cfg,args=argparse.Namespace(branch='schedule_s1p0'),beta_q=common.beta_q,lambda_q=common.lambda_q,frame_qp=common.frame_qp,load=historical_load,ROOT=ROOT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(V62/'fullqp_train.py'),'exec'),ns)
    dump(ROOT/'audits/objective_source.json',dict(status='PASS',source=str(V62/'fullqp_train.py'),sha256=sha(V62/'fullqp_train.py'),nested_function='main.run_clip',AST_unmodified=True,extra_loss=False,rate_gradient_fix=False))
    return ns['run_clip']
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--initialize-only',action='store_true');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    lock=(ROOT/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.initialize_only:assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS'
    v4=module('v614_v4_train',V4/'train.py');device=torch.device('cuda:0');assert torch.cuda.device_count()==1
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12);im.requires_grad_(False);pm.requires_grad_(False)
    models=dict(wrapper=wrapper,bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True),generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True))
    cp=cfg['source_checkpoint'];assert sha(cp['path'])==EXPECTED;state=torch.load(cp['path'],map_location='cpu',weights_only=True);assert state['step']==1000
    for k,m in models.items():m.load_state_dict(state[k],strict=True)
    initial={k:v4.module_hash(m) for k,m in models.items()};assert initial==cp['module_hashes'];del state
    compression=v4.compression_hash(im,pm);params=[p for m in models.values() for p in m.parameters()];ids={id(p) for p in params}
    assert len(ids)==len(params) and all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    oc=cfg['optimizer'];opt=torch.optim.AdamW([dict(name=k,params=list(m.parameters()),lr=oc[k+'_lr']) for k,m in models.items()],weight_decay=oc['weight_decay']);assert not opt.state
    quality=v4.quality_models(device);quality_hashes=[v4.module_hash(m) for m in quality]
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);random.seed(cfg['seed']);rng=random.Random(cfg['seed'])
    code={p.name:sha(p) for p in ROOT.glob('*.py')}
    dump(ROOT/'audits/initialization.json',dict(status='PASS',source_step=1000,adaptation_step=0,checkpoint=cp,module_hashes=initial,compression_hash=compression,fresh_optimizer=True,source_optimizer_restored=False,initial_optimizer_entries=0,trainable_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},frozen_other_compression_parameters=True))
    index=load(ROOT/'checkpoint_index.json') if (ROOT/'checkpoint_index.json').exists() else {};latest=ROOT/'checkpoints/latest.pt';start=0
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True);assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['source_checkpoint_sha256']==EXPECTED
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);rng.setstate(s['sampling_rng_state']);random.setstate(s['python_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state']);start=s['adaptation_step'];del s
    def save(step):
        assert v4.compression_hash(im,pm)==compression and [v4.module_hash(m) for m in quality]==quality_hashes
        s=dict(schema='v614',branch='uvg_only',source_step=1000,adaptation_step=step,optimizer_update_count=step,source_checkpoint_sha256=EXPECTED,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),sampling_rng_state=rng.getstate(),python_rng_state=random.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),compression_hash=compression,quality_hashes=quality_hashes,config_sha256=sha(ROOT/'config.json'),code_hashes=code)
        atomic_torch(latest,s)
        if step in STEPS:
            path=cp_path(step)
            if not path.exists():atomic_torch(path,s)
            export=ROOT/'evaluation/checkpoints'/f'{name(step)}.pt'
            if not export.exists():atomic_torch(export,{k:m.state_dict() for k,m in models.items()})
            hashes={k:v4.module_hash(m) for k,m in models.items()}
            index[str(step)]=dict(path=str(path),sha256=sha(path),source_step=1000,adaptation_step=step,module_hashes=hashes,compression_hash=compression,inference=dict(path=str(export),sha256=sha(export),module_hashes=hashes))
            dump(ROOT/'checkpoint_index.json',index)
    log=ROOT/'training_logs/uvg_only.jsonl';csvpath=log.with_suffix('.csv');rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if any(r['adaptation_step']>start for r in rows):
        dump(ROOT/'logs'/f'pre_resume_rows_{time.time_ns()}.json',rows);rows=[r for r in rows if r['adaptation_step']<=start];tmp=log.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(log)
    assert [r['adaptation_step'] for r in rows]==list(range(1,start+1))
    if rows:write(csvpath,rows)
    if start==0:save(0)
    if a.initialize_only:print('INITIALIZATION PASS',flush=True);return
    run_clip=objective(v4,im,pm,models,quality,cfg)
    sample=module('v614_training_data',ROOT/'data.py').sample
    tic=time.time()
    with log.open('a',buffering=1) as stream:
        for step in range(start+1,5001):
            t=time.time();frames,plan=sample(rng,device);q=rng.randrange(10);opt.zero_grad(set_to_none=True);sums,actual=run_clip(frames,q)
            assert actual==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:4]
            gradients={k:v4.grad_norm(list(m.parameters())) for k,m in models.items()}
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(norm) and all(math.isfinite(x) for x in sums.values())
            opt.step();assert all(bool(torch.isfinite(p).all()) for p in params)
            row=dict(branch='uvg_only',source_step=1000,adaptation_step=step,optimizer_update_count=step,external_qp=q,actual_qps=actual,**plan,**sums,gradient_norm=norm,gradient_norms=gradients,learning_rates={g['name']:g['lr'] for g in opt.param_groups},elapsed_seconds=time.time()-t,all_finite=True)
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            with csvpath.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if step==1:w.writeheader()
                w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
            if step%50==0:stream.flush();os.fsync(stream.fileno());save(step)
            if step==1 or step%10==0:
                dump(ROOT/'training_status.json',dict(status='RUNNING',PID=os.getpid(),gpu=a.gpu,source_step=1000,adaptation_step=step,target=5000,seconds_per_update=(time.time()-tic)/(step-start),updated_unix=time.time()));print('TRAIN',step,'/5000',flush=True)
    save(5000)
    dump(ROOT/'training_integrity.json',dict(status='PASS',source_step=1000,updates=5000,compression_hash=compression,frozen_compression_unchanged=True,quality_unchanged=True,fresh_source_optimizer=True))
    dump(ROOT/'training_status.json',dict(status='PASS',PID=os.getpid(),gpu=a.gpu,source_step=1000,adaptation_step=5000,target=5000))
    dump(ROOT/'frozen_checkpoint_index.json',dict(status='PASS',checkpoints=index,checkpoint_index_sha256=sha(ROOT/'checkpoint_index.json'),frozen_unix=time.time(),holdout_evaluation_not_started=True))
    print('TRAINING PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'failure_train_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
