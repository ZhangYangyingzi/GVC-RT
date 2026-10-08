"""Five objective branches, identical sampling, isolated resumable checkpoints."""
import argparse
import fcntl
import math
import random
import shutil
import traceback
from fullqp_common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--branch',required=True,choices=list(BRANCHES))
    parser.add_argument('--gpu',required=True,type=int,choices=(4,5,6,7))
    parser.add_argument('--limit',type=int,default=3000)
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    cfg=check_gate(); assert 1<=args.limit<=cfg['updates']; assert_old_unchanged()
    br=ROOT/'branches'/args.branch
    lock=(br/'parts'/'fullqp_train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v4=module('fullqp_train_v4',V4/'train.py'); device=torch.device('cuda:0')
    assert torch.cuda.device_count()==1
    wrapper=v4.NeuralWrapper().to(device).float().train()
    im,pm=v4.load_models(device,force_zero_thres=cfg['force_zero_thres']);im.requires_grad_(False);pm.requires_grad_(False)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True)
    assert source['step']==20000
    for k,m in models.items():m.load_state_dict(source[k],strict=True)
    initial={k:v4.module_hash(m) for k,m in models.items()}
    assert initial==model_hashes(source)
    del source
    frozen_hash=v4.compression_hash(im,pm)
    params=[p for m in models.values() for p in m.parameters()]
    train_ids={id(p) for p in params}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in train_ids)
    oc=cfg['optimizer']
    optimizer=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
    assert len(optimizer.state)==0
    code_hashes={p.name:sha(p) for p in (ROOT/'fullqp_train.py',ROOT/'fullqp_common.py')}
    init_audit=dict(status='PASS',revision=REVISION,branch=args.branch,source_checkpoint_sha256=cfg['source_checkpoint_sha256'],
        model_hashes=initial,compression_hash=frozen_hash,fresh_optimizer=True,source_optimizer_restored=False,initial_optimizer_state_entries=0,
        trainable_parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},gpu=args.gpu,code_hashes=code_hashes)
    if (br/'initialization_audit.json').exists():assert load(br/'initialization_audit.json')['model_hashes']==initial
    else:dump(br/'initialization_audit.json',init_audit)
    quality=v4.quality_models(device)
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    rng=random.Random(cfg['seed']);records=load(cfg['train_manifest'])['videos'];hist={q:0 for q in range(10)}
    checkpoint_hashes={};start=0
    prior=br/'checkpoint_hashes.json'
    if prior.exists():
        checkpoint_hashes=load(prior);start=max(map(int,checkpoint_hashes))
        cp=cp_path(args.branch,start);assert sha(cp)==checkpoint_hashes[str(start)]['sha256']
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['revision']==REVISION and state['branch']==args.branch and state['code_hashes']==code_hashes
        assert state['config_sha256']==sha(br/'config.json') and state['initialization_hashes']==initial
        for k,m in models.items():m.load_state_dict(state[k],strict=True)
        optimizer.load_state_dict(state['optimizer']);rng.setstate(state['sampling_rng_state'])
        torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state(state['cuda_rng_state'])
        hist={int(k):v for k,v in state['qp_histogram'].items()};del state
    else:assert not list((br/'checkpoints').glob('step_*.pt')),'Unindexed checkpoint requires audit'
    logpath=br/'training_log.jsonl'
    if logpath.exists():
        rows=[json.loads(line) for line in logpath.read_text().splitlines() if line.strip()]
        retained=[r for r in rows if r['update']<=start]
        assert [r['update'] for r in retained]==list(range(1,start+1))
        if len(retained)!=len(rows):
            backup=br/'logs'/f'pre_resume_{time.time_ns()}.jsonl';shutil.copy2(logpath,backup)
            tmp=logpath.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in retained));tmp.replace(logpath)
    else:assert start==0
    def save(step):
        assert v4.compression_hash(im,pm)==frozen_hash
        path=cp_path(args.branch,step);assert not path.exists()
        state=dict(revision=REVISION,branch=args.branch,step=step,scale=BRANCHES[args.branch],
            wrapper=wrapper.state_dict(),bridge=bridge.state_dict(),generator=generator.state_dict(),optimizer=optimizer.state_dict(),
            initialization_hashes=initial,compression_hash=frozen_hash,sampling_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(),qp_histogram=hist,code_hashes=code_hashes,config_sha256=sha(br/'config.json'),
            training_log_sha256=sha(logpath) if logpath.exists() else None)
        temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
        checkpoint_hashes[str(step)]=dict(path=str(path),sha256=sha(path),step=step,module_hashes={k:v4.module_hash(m) for k,m in models.items()},compression_hash=frozen_hash)
        dump(prior,checkpoint_hashes)
    def run_clip(frames,q,diagnostic=False):
        beta=beta_q(args.branch,q); pm.clear_dpb();pm.set_curr_poc(0)
        actual=[frame_qp(pm,q,i) for i in range(4)]
        assert actual==load(ROOT/'qp_train_eval_semantics_audit.json')['rows'][q]['evaluation']
        with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(wrapper(frames[0])),actual[0])['x_hat'])
        capt=[];hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
        sums={k:0.0 for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','lambda_proxy_proxy_L1')}
        wp=list(wrapper.parameters());gd=[torch.zeros_like(p) for p in wp] if diagnostic else None;gr=[torch.zeros_like(p) for p in wp] if diagnostic else None
        try:
            for i,frame in enumerate(frames[1:],1):
                proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,actual[i])
                dist,lp,ds=v4.perceptual(rec,frame,quality);rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame)
                components=dict(loss=dist+beta*rb+cfg['lambda_proxy']*pl,LPIPS=lp,DISTS=ds,D=dist,rate_bpp=rb,beta_rate=beta*rb,proxy_L1=pl,lambda_proxy_proxy_L1=cfg['lambda_proxy']*pl)
                assert all(bool(torch.isfinite(x).all()) for x in components.values())
                if diagnostic:
                    d=torch.autograd.grad(dist/3,wp,retain_graph=True,allow_unused=True)
                    r=torch.autograd.grad(rb/3,wp,allow_unused=True)
                    for j in range(len(wp)):
                        if d[j] is not None:gd[j].add_(d[j].detach())
                        if r[j] is not None:gr[j].add_(r[j].detach())
                else:(components['loss']/3).backward()
                for k,x in components.items():sums[k]+=float(x.detach())/3
                with torch.no_grad():pm.add_ref_frame(capt[-1].detach(),(rec.detach()*2-1).half())
                capt.clear()
        finally:hook.remove()
        sums['beta_q']=beta;sums['lambda_q']=lambda_q(q);sums['rate_to_distortion_ratio']=sums['beta_rate']/sums['D']
        if diagnostic:
            nd=math.sqrt(sum(float(g.double().square().sum()) for g in gd));nr=math.sqrt(sum(float(g.double().square().sum()) for g in gr))
            dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gd,gr));cos=dot/(nd*nr) if nd*nr else None
            assert nd>0 and nr>0 and all(math.isfinite(x) for x in (nd,nr,cos))
            return dict(**sums,grad_D_norm=nd,grad_R_norm=nr,grad_beta_R_norm=beta*nr,cos_grad_D_R=cos,actual_qps=actual)
        return sums,actual
    diagnostic_plan=None
    def diagnostic(step):
        nonlocal diagnostic_plan
        target=br/'parts'/f'gradient_{step:04d}.json'
        if target.exists():return
        optimizer.zero_grad(set_to_none=True)
        before={k:v4.module_hash(m) for k,m in models.items()}
        with torch.random.fork_rng(devices=[0]):
            frames,plan=v4.sample_clip(records,random.Random(cfg['seed']+1),device)
            diagnostic_plan=plan;rows=[]
            for q in (0,9):
                rows.append(dict(branch=args.branch,checkpoint_update=step,external_qp=q,module='wrapper',**plan,**run_clip(frames,q,True)))
        optimizer.zero_grad(set_to_none=True)
        assert before=={k:v4.module_hash(m) for k,m in models.items()}
        assert v4.compression_hash(im,pm)==frozen_hash
        dump(target,dict(status='PASS',rows=rows,model_unchanged=True,optimizer_update=False))
        merged=[]
        for p in sorted((br/'parts').glob('gradient_*.json')):merged.extend(load(p)['rows'])
        write(br/'gradient_balance_diagnostics.csv',merged)
    if not prior.exists():save(0)
    if start in STEPS:diagnostic(start)
    dump(br/'runtime_qp_smoke_audit.json',dict(status='PASS',endpoint_external_qps=[0,9],diagnostic_gradients_finite=True,source_audit_sha256=sha(ROOT/'qp_train_eval_semantics_audit.json')))
    print('INITIALIZED',args.branch,'start',start,'gpu',args.gpu,flush=True)
    begin=time.time()
    with logpath.open('a',buffering=1) as log:
        for step in range(start+1,args.limit+1):
            frames,plan=v4.sample_clip(records,rng,device);q=rng.randrange(10);hist[q]+=1
            optimizer.zero_grad(set_to_none=True);sums,actual=run_clip(frames,q)
            norm=float(torch.nn.utils.clip_grad_norm_(params,oc['grad_clip']));assert math.isfinite(norm) and norm>0
            optimizer.step()
            row=dict(branch=args.branch,update=step,external_qp=q,actual_qps=actual,**plan,**sums,gradient_norm=norm,all_finite=True)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step==start+1 or step%25==0:
                dump(br/'training_status.json',dict(status='RUNNING',revision=REVISION,update=step,target=3000,gpu=args.gpu,pid=os.getpid(),updated_unix=time.time(),seconds_per_update=(time.time()-begin)/(step-start)))
                write(br/'training_qp_histogram.csv',[dict(external_qp=q,count=hist[q],fraction=hist[q]/step) for q in range(10)])
                print(args.branch,step,'/ 3000',flush=True)
            if step in STEPS:
                log.flush();os.fsync(log.fileno());save(step);diagnostic(step)
    assert_old_unchanged()
    dump(br/'training_status.json',dict(status='PASS' if args.limit==3000 else 'PAUSED',revision=REVISION,update=args.limit,target=3000,gpu=args.gpu,updated_unix=time.time()))
    print('TRAIN COMPLETE',args.branch,args.limit,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--branch');a,_=p.parse_known_args()
        if a.branch in BRANCHES:dump(ROOT/'branches'/a.branch/'training_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time_unix=time.time()))
        raise
