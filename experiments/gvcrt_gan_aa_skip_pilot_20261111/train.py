"""One independent generator/D pair per group, resumable after every update pair."""
from common_s2 import *
import argparse,traceback


def run(group,until):
    env();verify_sources()
    import torch
    from experiment_model import construct,restore,trainable_state
    from model_context import frozen_items,assert_reset
    from cache_codec import cache_paths
    from gan import discriminator,g_mode,d_mode,d_loss
    from loss import backward_window,norm,dot
    assert int(os.environ['CUDA_VISIBLE_DEVICES'])==GROUPS.index(group)
    assert load(ROOT/'audits/preflight.json')['status']=='PASS'
    coef=load(ROOT/'gradient_calibration.json');assert coef['status']=='PASS'
    root=ROOT/group/'training';root.mkdir(parents=True,exist_ok=True)
    c=construct(group,True,0);seed(TRAIN_SEED)
    params=c['params'];initial={n:p.detach().cpu().clone() for n,p in c['named']}
    assert tensor_hash(c['named'])==load(ROOT/'audits/preflight.json')['common_trainable_SHA256']
    frozen=tensor_hash(frozen_items(c));d=dopt=d_rng=None
    if group in ['J1','J3']:
        d,dopt=discriminator(TRAIN_SEED+1,c['device']);a=torch.load(ROOT/'checkpoints/common_D_initial.pt',map_location='cpu',weights_only=True)
        d.load_state_dict(a['D']);dopt.load_state_dict(a['D_optimizer']);d_rng=a['D_RNG'];g_mode(d)
        assert tensor_hash(d.state_dict().items())==coef['D_post_warmup_SHA256']
    latest=root/'latest.pt';step=0;rows=plan_rows()
    if latest.exists():
        a=torch.load(latest,map_location='cpu',weights_only=True)
        assert a['protocol_SHA256']==sha(ROOT/'protocol.json') and a['plan_SHA256']==sha(ROOT/'plans/train.json.gz') and a['calibration_SHA256']==sha(ROOT/'gradient_calibration.json')
        restore(c,a);restore_rng(a['RNG']);step=a['committed_updates']
        archive=root/f'checkpoints/step_{step:04d}.pt'
        if step%25==0 and not archive.exists():atomic_torch(archive,a)
        assert a['sampler_cursor']==step*4 and a['GAN_ramp_progress']==min(step,100)
        if d is not None:d.load_state_dict(a['D']);dopt.load_state_dict(a['D_optimizer']);d_rng=a['D_RNG']
        dump(root/'progress.json',dict(status='RUNNING',committed_updates=step,next_cursor=4*step,seconds=a.get('last_receipt',{}).get('seconds',0),time=time.time()))
    def save(s,receipt):
        a=dict(experiment_group=group,B300_SHA256=B300_SHA,committed_updates=s,sampler_cursor=s*4,GAN_ramp_progress=min(s,100),trainable_state=trainable_state(c),optimizer=c['optimizer'].state_dict(),D=None if d is None else d.state_dict(),D_optimizer=None if dopt is None else dopt.state_dict(),D_RNG=d_rng,RNG=rng_state(),protocol_SHA256=sha(ROOT/'protocol.json'),plan_SHA256=sha(ROOT/'plans/train.json.gz'),calibration_SHA256=sha(ROOT/'gradient_calibration.json'),last_receipt=receipt,provenance=c['provenance'])
        atomic_torch(latest,a)
        if s%25==0:atomic_torch(root/f'checkpoints/step_{s:04d}.pt',a)
    if not latest.exists():save(0,{})
    for update in range(step+1,min(until,500)+1):
        if (ROOT/'runner_stop_request.json').exists():return 7
        if (ROOT/group/'yield_for_cache.json').exists():return 6
        batch=rows[4*(update-1):4*update];paths=[]
        for row in batch:
            _,path,receipt=cache_paths(row,row['external_qp'],True)
            if not path.exists() or not receipt.exists():
                prior=load(root/'progress.json') if (root/'progress.json').exists() else {}
                dump(root/'progress.json',dict(status='WAITING_FOR_CACHE_GPU_RELEASED',committed_updates=update-1,next_cursor=4*(update-1),seconds=prior.get('seconds'),time=time.time()))
                return 6
            paths.append((path,receipt))
        begin=time.time();c['optimizer'].zero_grad(set_to_none=True);traces=[];fakes=[];gradients={name:None for name in ['spatial','temporal','adversarial']}
        weight=coef['lambda_adv']*min(update/100,1) if d is not None else 0.
        d_before=None if d is None else tensor_hash(d.state_dict().items())
        for row,(path,rp) in zip(batch,paths):
            r=load(rp);assert r['status']=='PASS' and sha(path)==r['cache_sha256']
            data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert data['info']['source_row']==row and data['info']['actual_qps']==row['actual_frame_qps']
            trace,pairs,diags=backward_window(c,data,batch=4,d=d,adv_weight=weight,diagnostics=update%25==0)
            traces.append(trace);fakes+=pairs
            for name,values in diags.items():
                if values is None:continue
                if gradients[name] is None:gradients[name]=values
                else:
                    for k,v in enumerate(values):
                        if v is not None:
                            if gradients[name][k] is None:gradients[name][k]=v
                            else:gradients[name][k].add_(v)
            del data
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        unclipped=norm([p.grad for p in params]);assert unclipped>0 and math.isfinite(unclipped)
        if d is not None:assert tensor_hash(d.state_dict().items())==d_before,'G_STEP_CHANGED_D_OR_SPECTRAL_BUFFERS'
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);c['optimizer'].step()
        assert all(torch.isfinite(p).all() for p in params)
        dloss=real_logit=fake_logit=None
        if d is not None:
            g_rng=rng_state();restore_rng(d_rng);d_mode(d);dopt.zero_grad(set_to_none=True);dloss=real_logit=fake_logit=0.
            for fake,real,scale in fakes:
                value,a,b=d_loss(d,real.to(c['device']),fake.to(c['device']));(value*scale).backward()
                dloss+=float(value.detach())*scale;real_logit+=float(a)*scale;fake_logit+=float(b)*scale
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
            dopt.step();d_rng=rng_state();restore_rng(g_rng);g_mode(d)
            assert all(torch.isfinite(p).all() for p in d.parameters())
        diagnostics={}
        if update%25==0:
            assert tensor_hash(frozen_items(c))==frozen
            for name,values in gradients.items():diagnostics[name+'_weighted_gradient_norm']=norm(values) if values is not None else 0.
            for a,b in [('spatial','temporal'),('spatial','adversarial'),('temporal','adversarial')]:
                na=diagnostics[a+'_weighted_gradient_norm'];nb=diagnostics[b+'_weighted_gradient_norm']
                diagnostics[a+'_'+b+'_cosine']=dot(gradients[a],gradients[b])/(na*nb) if na*nb else None
            diagnostics['trainable_drift_L2']=norm([p.detach().cpu()-initial[n] for n,p in c['named']])
            diagnostics['Skip_and_main_RMS']=c.get('last_RMS')
        receipt=dict(group=group,committed_updates=update,effective_batch=4,sample_keys=[r['sample_key'] for r in batch],domains=[r['domain'] for r in batch],QP=[r['external_qp'] for r in batch],actual_frame_QP=[r['actual_frame_qps'] for r in batch],next_cursor=4*update,G_loss=sum(t['objective_contribution'] for t in traces),G_preclip_norm=unclipped,G_adv_weight=weight,D_loss=dloss,D_real_logit=real_logit,D_fake_logit=fake_logit,diagnostics=diagnostics,windows=traces,frozen_SHA256=frozen,seconds=time.time()-begin,PID=os.getpid(),physical_GPU=physical_gpu(0),process_GPU=0,time=time.time())
        save(update,receipt)
        dump(root/f'receipts/{update:04d}.json',receipt)
        dump(root/'progress.json',dict(status='COMPLETE_500_UPDATES' if update==500 else 'RUNNING',committed_updates=update,next_cursor=4*update,seconds=receipt['seconds'],ETA_seconds=(500-update)*receipt['seconds'],time=time.time()))
        event('G_D_pair_committed',group=group,step=update,seconds=receipt['seconds'],loss=receipt['G_loss'])
        del traces,fakes,gradients,receipt
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--until',type=int,default=500);a=p.parse_args();env()
    try:
        with lock(f'locks/train_{a.group}.lock'):sys.exit(run(a.group,a.until))
    except Exception as error:
        message=repr(error)
        dump(ROOT/a.group/'training/failure.json',dict(status='FAILED',error=message,OOM='out of memory' in message.lower(),traceback=traceback.format_exc(),time=time.time()))
        raise
