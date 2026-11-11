"""Independent GPU0/GPU1 trainers with atomic optimizer/RNG/cursor commits."""
from common_s2 import *
import argparse,traceback


def main(group):
    env();import torch
    from loss import load_training_context,window,norm
    from cache_h import ready,paths
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu==GROUPS.index(group)
    assert load(ROOT/'audits/preflight.json')['status']=='PASS'
    coeff=load(ROOT/'gradient_calibration.json')['lambda_temp']
    c=load_training_context(torch.device('cuda',0));branch=c['adapter']
    optimizer=torch.optim.AdamW(branch.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    root=ROOT/group/'training';root.mkdir(parents=True,exist_ok=True);rows=plan_rows();step=0
    latest=root/'latest.pt';frozen=tensor_hash(c['head'].state_dict().items())
    qhash=[tensor_hash(m.state_dict().items()) for m in c['quality']]
    seed(TRAIN_SEED)
    if latest.exists():
        a=torch.load(latest,map_location='cpu',weights_only=True)
        assert a['group']==group and a['B300_SHA256']==B300_SHA and a['protocol_SHA256']==sha(ROOT/'protocol.json')
        assert a['plan_SHA256']==sha(ROOT/'plans/index.json') and a['lambda_temp']==coeff
        branch.load_state_dict(a['adapter'],strict=True);optimizer.load_state_dict(a['optimizer']);restore_rng(a['RNG'])
        step=a['committed_updates'];assert a['next_cursor']==4*step
        if step:dump(root/f'receipts/{step:04d}.json',a['receipt'])
    def save(s,receipt):
        a=dict(group=group,B300_SHA256=B300_SHA,common_step0_SHA256=sha(ROOT/'checkpoints/common_adapter_step0.pt'),
            adapter={n:t.detach().cpu().clone() for n,t in branch.state_dict().items()},optimizer=optimizer.state_dict(),
            RNG=rng_state(),committed_updates=s,next_cursor=4*s,receipt=receipt,lambda_temp=coeff,
            protocol_SHA256=sha(ROOT/'protocol.json'),plan_SHA256=sha(ROOT/'plans/index.json'),
            calibration_SHA256=sha(ROOT/'gradient_calibration.json'),frozen_head_SHA256=frozen,
            common_seed=TRAIN_SEED,optimizer_parameters=[n for n,p in branch.named_parameters()])
        atomic_torch(latest,a)
        if s in (0,25,50,75,100):atomic_torch(root/f'checkpoints/step_{s:04d}.pt',a)
        dump(root/'progress.json',dict(status='COMPLETE' if s==100 else 'RUNNING',committed_updates=s,next_cursor=4*s,
            PID=os.getpid(),physical_GPU=gpu,time=time.time()))
    if not latest.exists():save(0,None)
    def monitor(s):
        rec=[]
        for r in load(ROOT/'manifests/monitor_windows.json'):
            while not ready(r):time.sleep(2)
            p,rp=paths(r);d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            with torch.no_grad():trace,_=window(c,d,lambda_temp=coeff if group==GROUPS[1] else 0.,backward=False)
            rec.append(dict(sample_key=r['sample_key'],QP=r['external_qp'],**trace))
        dump(ROOT/f'training_logs/monitor_{group}_step{s:04d}.json',dict(step=s,group=group,training_windows=True,records=rec))
    if step==0:monitor(0)
    for update in range(step+1,101):
        batch=rows[4*(update-1):4*update];start=time.time()
        while not all(ready(r) for r in batch):time.sleep(2)
        wait=time.time()-start;optimizer.zero_grad(set_to_none=True);traces=[];torch.cuda.reset_peak_memory_stats()
        for row in batch:
            p,rp=paths(row);assert sha(p)==load(rp)['SHA256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            tr,_=window(c,d,batch=4,lambda_temp=coeff if group==GROUPS[1] else 0.);traces.append(tr)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        grad=norm([p.grad for p in branch.parameters()]);assert grad>0 and math.isfinite(grad)
        upstream=norm([p.grad for m in [branch.A,branch.temporal] for p in m.parameters()])
        if update>=2:assert upstream>0,'UPSTREAM_ADAPTER_GRADIENT_MISSING_AFTER_B_UPDATE'
        torch.nn.utils.clip_grad_norm_(branch.parameters(),1,error_if_nonfinite=True);optimizer.step()
        assert all(int(v['step'])==update for v in optimizer.state.values())
        assert all(torch.isfinite(p).all() for p in branch.parameters())
        assert tensor_hash(c['head'].state_dict().items())==frozen
        assert all(p.grad is None and not p.requires_grad for p in c['head'].parameters())
        receipt=dict(group=group,committed_updates=update,next_cursor=4*update,effective_batch=4,
            sample_keys=[r['sample_key'] for r in batch],loss=sum(t['objective'] for t in traces),
            gradient_norm=grad,upstream_gradient_norm=upstream,windows=traces,cache_wait_seconds=wait,
            seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),physical_GPU=gpu,PID=os.getpid(),frozen_head_unchanged=True,time=time.time())
        save(update,receipt);dump(root/f'receipts/{update:04d}.json',receipt)
        event('optimizer_update_committed',group=group,step=update,loss=receipt['loss'],seconds=receipt['seconds'])
        if update in (50,100):monitor(update)
    assert [tensor_hash(m.state_dict().items()) for m in c['quality']]==qhash
    dump(root/'completion.json',dict(status='PASS',committed_updates=100,consumed_windows=400,frozen_head_unchanged=True,
        frozen_perceptual_unchanged=True,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);a=p.parse_args()
    try:
        with lock('locks/train_'+a.group+'.lock'):main(a.group)
    except Exception as e:
        env();dump(ROOT/f'audits/train_failure_{a.group}_{os.getpid()}.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
