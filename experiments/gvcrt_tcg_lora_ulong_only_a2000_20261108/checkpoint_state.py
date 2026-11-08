"""Validation bookkeeping without modifying immutable evaluated checkpoint files."""
from common_s2 import *

def synchronize_validation_metadata(group,control):
    import torch
    home=group_dir(group);path=home/'checkpoints/latest.pt';s=torch.load(path,map_location='cpu',weights_only=True)
    step=s['committed_optimizer_updates'];assert control['last_validation_step']<=step
    assert s['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
    if s.get('validation_state')!=control:
        before=tensor_hash(s['lora_state'].items());s['validation_state']=control
        s['validation_metadata_commit']=dict(model_updates=step,last_validation_step=control['last_validation_step'],time=time.time())
        atomic_torch(home/f'checkpoints/validated_step_{step}.pt',s);atomic_torch(path,s)
        r=torch.load(path,map_location='cpu',weights_only=True)
        assert r['validation_state']==control and tensor_hash(r['lora_state'].items())==before and r['committed_optimizer_updates']==step
    checkpoint_manifest()

def checkpoint_manifest():
    import torch
    records=[]
    for group in GROUPS:
        for p in sorted((group_dir(group)/'checkpoints').glob('*.pt')):
            s=torch.load(p,map_location='cpu',weights_only=True)
            records.append(dict(group=group,filename=p.name,path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p),
                committed_optimizer_updates=s['committed_optimizer_updates'],next_cursor=s['next_cursor'],logical_RNG_states=len(s['all_logical_rank_RNG']),
                physical_world_size=s['physical_world_size'],TCG_cumulative_step_frozen_at=6500))
    dump(ROOT/'checkpoints_manifest.json',dict(records=records,weights_uploaded=False,time=time.time()))
