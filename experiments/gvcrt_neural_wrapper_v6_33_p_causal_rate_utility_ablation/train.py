"""Fixed 2x2 experiment; historical B1 objective and exact shared frame plan."""
import argparse
import os
import time
import traceback
from runtime import *

def main(branch,gpu):
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
    import torch
    assert load(ROOT/'data_split_audit.json')['status']=='PASS'
    assert load(ROOT/'sanity/identity.json')['status']=='PASS'
    ctx=setup(branch);v4,im,pm,models,quality=ctx;initial=hashes(ctx)
    assert initial['compression']==load(V32/'evaluation_config.json')['compression_hash']
    cfg=load(ROOT/'protocol.json');pdir=ROOT/'branches'/branch;pdir.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(pdir/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    params=[p for m in models.values() for p in m.parameters() if p.requires_grad]
    groups=[dict(name=k,params=[p for p in m.parameters() if p.requires_grad],lr=cfg['optimizer'][k+'_lr']) for k,m in models.items() if any(p.requires_grad for p in m.parameters())]
    opt=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    plan=load(V32/'plans/shared.json')['updates'];run=runner(ctx);rows=[];start=0
    latest=pdir/'checkpoints/latest.pt'
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True)
        assert s['protocol_sha256']==sha(ROOT/'protocol.json')
        for k,m in models.items():m.load_state_dict(s[k])
        opt.load_state_dict(s['optimizer']);start=s['step'];torch.set_rng_state(s['rng']);torch.cuda.set_rng_state(s['cuda_rng'])
        rows=load(pdir/'training_rows.json')[:start]
    def save(step):
        h=hashes(ctx)
        assert h['compression']==initial['compression'] and h['I']==initial['I'] and h['quality']==initial['quality']
        if branch==BRANCHES[2]:
            assert all(h['modules'][k]==initial['modules'][k] for k in ('bridge','generator'))
        if branch==BRANCHES[1]:assert not list(models['wrapper'].parameters())
        if step>0:assert all(h['modules'][k]!=initial['modules'][k] for k,m in models.items() if any(p.requires_grad for p in m.parameters()))
        s=dict(branch=branch,step=step,protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=sha(V32/'plans/shared.json'),initial=initial,hashes=h,
               **{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        atomic_save(latest,s)
        if step in (0,100,250,1000):
            p=pdir/'checkpoints'/f'adaptation_step_{step:04d}.pt';atomic_save(p,s)
            dump(f'branches/{branch}/integrity_step_{step}.json',dict(status='PASS',step=step,initial=initial,final=h,checkpoint=str(p),sha256=sha(p)))
    if not start:save(0)
    for step in range(start+1,1001):
        t=time.time();row=plan[step-1];f=frames(row);opt.zero_grad(set_to_none=True)
        try:values,qps,trace,warmup,_=run(f,row['external_qp'],'B1_reconstruction',rec_weight=cfg['lambda_rec'])
        finally:f.close()
        assert qps==row['actual_qps']
        grad={k:dict(norm=sum(float(p.grad.double().square().sum()) for p in m.parameters() if p.grad is not None)**.5,
                     count=sum(p.grad is not None and bool(p.grad.any()) for p in m.parameters())) for k,m in models.items()}
        assert all(grad[k]['norm']>0 for k,m in models.items() if any(p.requires_grad for p in m.parameters()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert norm>0 and torch.isfinite(torch.tensor(norm))
        opt.step()
        rows.append(dict(step=step,video=row['video'],domain=row['domain'],external_qp=row['external_qp'],actual_qps=qps,crop_cache_used=getattr(f,'cache_used',False),
                         losses=values,gradients=grad,gradient_norm=norm,seconds=time.time()-t,peak_MiB=torch.cuda.max_memory_reserved()/2**20))
        dump(f'branches/{branch}/training_rows.json',rows);save(step)
        if step==1 or step%25==0:print(branch,step,rows[-1],flush=True)
    dump(f'branches/{branch}/training_integrity.json',dict(status='PASS',steps=1000,initial=initial,final=hashes(ctx),exact_shared_plan=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES[1:],required=True);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    try:main(a.branch,a.gpu)
    except Exception:
        dump(f'logs/error_{a.branch}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
