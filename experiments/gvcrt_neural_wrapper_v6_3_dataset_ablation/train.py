"""Three dataset branches with identical initialization, objective and QP sequence."""
import argparse,collections,fcntl,math,random,shutil,traceback
from v63_io import *
from loader import sample_clip
from objective import bind
def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    assert torch.cuda.device_count()==1
    br=ROOT/'branches'/a.branch;lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpulock=(ROOT/'parts'/f'gpu{a.gpu}.lock').open('a');fcntl.flock(gpulock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v4=module('v63_training_v4',V4/'train.py');device=torch.device('cuda:0')
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    wrapper=v4.NeuralWrapper().to(device).float().train()
    im,pm=v4.load_models(device,force_zero_thres=cfg['force_zero_thres']);im.requires_grad_(False);pm.requires_grad_(False)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    state=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True);assert state['step']==20000
    for k,m in models.items():m.load_state_dict(state[k],strict=True)
    initial={k:v4.module_hash(m) for k,m in models.items()};del state
    expected=load(V62B/'config.json')['checkpoints']['v41']['module_hashes'];assert initial==expected
    compression=v4.compression_hash(im,pm);assert compression==load(V62B/'final_integrity.json')['shared_compression_hash']
    params=[p for m in models.values() for p in m.parameters()];ids={id(p) for p in params}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    oc=cfg['optimizer'];optimizer=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
    assert len(optimizer.state)==0
    init=dict(status='PASS',branch=a.branch,module_hashes=initial,compression_hash=compression,fresh_optimizer=True,optimizer_entries=0,source_checkpoint_sha256=cfg['source_checkpoint_sha256'],gpu=a.gpu)
    if (br/'initialization_audit.json').exists():assert load(br/'initialization_audit.json')['module_hashes']==initial
    else:dump(br/'initialization_audit.json',init)
    quality=v4.quality_models(device);run_clip=bind(v4,im,pm,wrapper,quality,cfg)
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    if a.smoke:
        rows=[]
        for source in (('ulong','vimeo') if a.branch=='mixed_50_50' else (source_for(a.branch,1),)):
            frames,plan=sample_clip(source,random.Random(cfg['sample_seed']+1),device,v4)
            for q in (0,9):
                optimizer.zero_grad(set_to_none=True);sums,actual=run_clip(frames,q)
                norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm) and norm>0
                assert all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
                rows.append(dict(**plan,external_qp=q,actual_qps=actual,**sums,gradient_norm=norm,shape=[4,1,3,256,256],minimum=min(float(f.min()) for f in frames),maximum=max(float(f.max()) for f in frames)))
        assert {k:v4.module_hash(m) for k,m in models.items()}==initial and len(optimizer.state)==0
        optimizer.step();assert len(optimizer.state)>0
        assert all(bool(torch.isfinite(p).all()) for p in params)
        assert v4.compression_hash(im,pm)==compression
        dump(br/'smoke_audit.json',dict(status='PASS',rows=rows,disposable_optimizer_step=True,smoke_weights_discarded=True,formal_updates=0,initialization=init))
        print('SMOKE PASS',a.branch,flush=True);return
    assert load(ROOT/'smoke_audit.json')['status']=='PASS'
    qps=load(ROOT/'qp_sequence_audit.json')['sequence'];rng=random.Random(cfg['sample_seed']);start=0
    prior=br/'checkpoint_hashes.json';cp_hashes=load(prior) if prior.exists() else {}
    logfile=ROOT/'training_logs'/f'{a.branch}.jsonl';counts=collections.Counter()
    code_hashes={p.name:sha(p) for p in ROOT.glob('*.py')}
    if cp_hashes:
        start=max(map(int,cp_hashes));cp=checkpoint(a.branch,start);assert sha(cp)==cp_hashes[str(start)]['sha256']
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['step']==start and state['branch']==a.branch and state['config_sha256']==sha(ROOT/'config.json') and state['code_hashes']==code_hashes
        assert state['initialization_hashes']==initial and state['compression_hash']==compression
        for k,m in models.items():m.load_state_dict(state[k],strict=True)
        optimizer.load_state_dict(state['optimizer']);rng.setstate(state['sample_rng_state']);torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state(state['cuda_rng_state'])
        counts.update(state['source_counts']);del state
    else:assert not list((br/'checkpoints').glob('step_*.pt')),'Unindexed checkpoint requires audit'
    if logfile.exists():
        rows=[json.loads(s) for s in logfile.read_text().splitlines() if s.strip()];retained=[r for r in rows if r['update']<=start]
        assert [r['update'] for r in retained]==list(range(1,start+1))
        if len(rows)!=len(retained):
            shutil.copy2(logfile,br/'logs'/f'pre_resume_{time.time_ns()}.jsonl')
            tmp=logfile.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in retained));tmp.replace(logfile)
    else:assert start==0
    def save(step):
        assert v4.compression_hash(im,pm)==compression
        path=checkpoint(a.branch,step);assert not path.exists()
        state=dict(schema='v63_dataset_ablation_v1',branch=a.branch,step=step,wrapper=wrapper.state_dict(),bridge=bridge.state_dict(),generator=generator.state_dict(),optimizer=optimizer.state_dict(),
            initialization_hashes=initial,compression_hash=compression,sample_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
            qp_sequence_sha256=sha(ROOT/'qp_sequence_audit.json'),source_counts=dict(counts),config_sha256=sha(ROOT/'config.json'),code_hashes=code_hashes,
            training_log_sha256=sha(logfile) if logfile.exists() else None)
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
        cp_hashes[str(step)]=dict(path=str(path),sha256=sha(path),step=step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},compression_hash=compression)
        dump(prior,cp_hashes)
    if not cp_hashes:save(0)
    begin=time.time()
    with logfile.open('a',buffering=1) as log:
        for step in range(start+1,1001):
            source=source_for(a.branch,step);frames,plan=sample_clip(source,rng,device,v4);q=qps[step-1]
            optimizer.zero_grad(set_to_none=True);sums,actual=run_clip(frames,q)
            norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm) and norm>0
            optimizer.step();counts[source]+=1
            row=dict(branch=a.branch,update=step,external_qp=q,actual_qps=actual,**plan,**sums,gradient_norm=norm,all_finite=True)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step in STEPS:log.flush();os.fsync(log.fileno());save(step)
            if step==start+1 or step%10==0:
                dump(br/'training_status.json',dict(status='RUNNING',update=step,target=1000,gpu=a.gpu,pid=os.getpid(),source_counts=dict(counts),seconds_per_update=(time.time()-begin)/(step-start),checkpoints=sorted(map(int,cp_hashes)),updated_unix=time.time()))
                print('TRAIN',a.branch,step,'/ 1000',flush=True)
    expected_counts={'ulong':500,'vimeo':500} if a.branch=='mixed_50_50' else {source_for(a.branch,1):1000}
    assert dict(counts)==expected_counts
    frozen();dump(br/'training_status.json',dict(status='PASS',update=1000,target=1000,gpu=a.gpu,pid=os.getpid(),source_counts=dict(counts),checkpoints=list(STEPS),updated_unix=time.time()))
    print('TRAIN PASS',a.branch,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--branch');a,_=p.parse_known_args()
        dump(ROOT/'branches'/str(a.branch)/'failure.json',dict(status='FAIL',traceback=traceback.format_exc(),unix=time.time()));raise
