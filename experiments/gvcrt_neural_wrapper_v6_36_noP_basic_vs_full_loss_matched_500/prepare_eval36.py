import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
idx=json.load(open(R/'branches/B_basicLoss_noP_500/checkpoint_index.json'));ix=idx['500'];assert ix['adaptation_step']==500
inf=ix['inference'];assert sha(inf['path'])==inf['sha256']
# Persist the deployment hashes required by the inherited evaluator.
import torch
s=torch.load(inf['path'],map_location='cpu',weights_only=True)
def tensor_hash(sd):
 h=hashlib.sha256()
 for k,t in sd.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
dep={k:tensor_hash({n:t.half() for n,t in s[k].items()}) for k in ('bridge','generator')}
ix['deployment_receiver_hashes']=dep
idx['500']=ix
json.dump(idx,open(R/'branches/B_basicLoss_noP_500/checkpoint_index.json','w'),indent=2);open(R/'branches/B_basicLoss_noP_500/checkpoint_index.json','a').write('\n')
cfg=json.load(open(R/'evaluation_config.json'))
cfg['methods']['B_basicLoss_noP_500']=[None,'B_basicLoss_noP_500']
cfg['checkpoints']['B_basicLoss_noP_500']=dict(path=inf['path'],sha256=inf['sha256'],module_hashes=ix['module_hashes'])
cfg['deployment_receiver_hashes']['B_basicLoss_noP_500']=dep
json.dump(cfg,open(R/'evaluation_config.json','w'),indent=2);open(R/'evaluation_config.json','a').write('\n')
print('evaluation config ready',inf['sha256'],dep)
