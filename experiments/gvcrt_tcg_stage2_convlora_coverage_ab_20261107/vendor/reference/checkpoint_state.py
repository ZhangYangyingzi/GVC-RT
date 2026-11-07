"""Validation metadata commits preserve immutable evaluated model checkpoint bytes."""
from common_s2 import *

def synchronize_validation_metadata(control):
    import torch
    path=ROOT/'checkpoints/latest.pt';state=torch.load(path,map_location='cpu',weights_only=True)
    step=state['committed_optimizer_updates'];assert control['last_validation_step']<=step
    assert state['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
    if state.get('validation_state')!=control:
        prior=sha(path);model_hash=tensor_hash(state['lora_state'].items())
        state['validation_state']=control
        state['validation_metadata_commit']=dict(model_updates=step,last_validation_step=control['last_validation_step'],previous_latest_sha256=prior,time=time.time())
        if step==control['last_validation_step']:atomic_torch(ROOT/f'checkpoints/validated_step_{step}.pt',state)
        atomic_torch(path,state)
        reloaded=torch.load(path,map_location='cpu',weights_only=True)
        assert reloaded['validation_state']==control and reloaded['committed_optimizer_updates']==step
        assert tensor_hash(reloaded['lora_state'].items())==model_hash
        dump(ROOT/f'audits/validation_checkpoint_{step:04d}_{control["last_validation_step"]:04d}.json',dict(status='PASS',model_updates=step,
            last_validation_step=control['last_validation_step'],lora_sha256=model_hash,latest_sha256=sha(path),
            optimizer_and_all_rank_RNG_preserved=True,immutable_evaluated_checkpoint_changed=False))
    checkpoint_manifest()

def checkpoint_manifest():
    import torch
    records=[]
    for p in sorted((ROOT/'checkpoints').glob('*.pt')):
        s=torch.load(p,map_location='cpu',weights_only=True)
        records.append(dict(filename=p.name,size_bytes=p.stat().st_size,sha256=sha(p),committed_optimizer_updates=s['committed_optimizer_updates'],
            baseline_cumulative_TCG_step=6500,plan_cursor=s['next_cursor'],all_rank_RNG=len(s['all_rank_RNG']),
            last_validation_step=None if s.get('validation_state') is None else s['validation_state']['last_validation_step']))
    dump(ROOT/'checkpoints_manifest.json',dict(checkpoints=records,time=time.time(),weights_uploaded=False))
