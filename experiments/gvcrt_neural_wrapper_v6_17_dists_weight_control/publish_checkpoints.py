from io17 import *
def publish():
    import torch
    cfg=evalcfg();index=load(ROOT/'checkpoint_index.json') if (ROOT/'checkpoint_index.json').exists() else {};changed=False
    for step,r in index.items():
        if int(step) not in (0,1000):continue
        m=f'dists_w05_{int(step)}'
        if m in cfg['checkpoints']:continue
        cp=r['inference'];assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes']
        cfg['checkpoints'][m]=cp;cfg['deployment_receiver_hashes'][m]={k:tensor_hash({n:t.half() for n,t in s[k].items()}) for k in ('bridge','generator')};changed=True
    if changed:dump(ROOT/'evaluation/config.json',cfg)
    return cfg
if __name__=='__main__':publish()
