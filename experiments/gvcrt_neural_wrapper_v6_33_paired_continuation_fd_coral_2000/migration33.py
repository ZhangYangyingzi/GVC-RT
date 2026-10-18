"""Explicit verified old-to-new namespace mapping; never freshens optimizer/EMA."""
from io32 import *

def state_digest(x):
 import torch
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,torch.Tensor):
   h.update(('tensor:'+str(v.dtype)+str(tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k in v:visit(k);visit(v[k])
  elif isinstance(v,(tuple,list)):
   h.update(type(v).__name__.encode())
   for e in v:visit(e)
  else:h.update((type(v).__name__+':'+repr(v)).encode())
 visit(x);return h.hexdigest()

def load_state(ctx,opt,branch):
 import torch
 from training32 import restore_rng,restore_buffers,buffer_state,check_frozen
 b=ROOT/'branches'/branch;cfg=load(ROOT/'config.json');latest=b/'checkpoints/latest.pt';fresh_namespace=not latest.exists()
 src=cfg['branch_sources'][branch];p=Path(src['path']) if fresh_namespace else latest
 if fresh_namespace:
  assert sha(p)==src['sha256'];parent=p.parents[3];oldio=module('source_io_'+branch,parent/'io32.py')
  expected=oldio.signature('train');assert expected==src['old_signature']
 else:expected=signature('train')
 s=torch.load(p,map_location='cpu',weights_only=True)
 assert s['signature']==expected and s['branch']==branch
 if fresh_namespace:
  assert s['adaptation_step']==1000 and s['replay_cursor']==1000 and s['plan_sha256']==sha(ROOT/'plans/shared.json')
  assert s['config_sha256']==sha(parent/'config.json') and s['protocol_sha256']==sha(parent/'protocol.json')
 else:
  assert 1000<=s['adaptation_step']<=2000 and s['continuation_sha256']==sha(ROOT/'plans/continuation.json')
  assert s['calibration_sha256']==sha(ROOT/'calibration/final.json') and s['seed_index_sha256']==sha(ROOT/'reference_stats/index.json')
  assert s['parent_training_checkpoint_sha256']==src['sha256']
 models=ctx[3]
 for k,m in models.items():
  m.load_state_dict(s[k],strict=True)
  assert tensor_hash(m.state_dict())==tensor_hash(s[k])
  if k in ('bridge','generator'):assert all(p.dtype==torch.float32 for p in m.parameters())
 if 'buffers' in s:restore_buffers(models,s['buffers'])
 else:
  # Historical B stores all persistent module buffers in state_dict, not a second map.
  assert branch==BRANCHES[0]
  for k,m in models.items():
   assert all(n in s[k] for n,_ in m.named_buffers()),'Nonpersistent historical B buffer is not recoverable'
 opt.load_state_dict(s['optimizer'])
 assert state_digest(opt.state_dict())==state_digest(s['optimizer'])
 for g,lr in zip(opt.param_groups,(6e-6,1e-6)):
  assert g['lr']==lr and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==1e-4
 assert {int(v['step']) for v in opt.state.values()}=={s['adaptation_step']}
 restore_rng(s['crop_rng']);check_frozen(ctx);assert ctx[-1]==s['frozen_hashes']
 if branch!=BRANCHES[0]:
  assert s['calibration_sha256']==sha(ROOT/'calibration/final.json') and s['seed_index_sha256']==sha(ROOT/'reference_stats/index.json')
  assert len(s['EMA'])==20 and sum(v['counter'] for v in s['EMA'].values())==s['adaptation_step']
  assert all(s['EMA'][k]['counter']==s['visits'][k] for k in s['EMA'])
  if fresh_namespace:assert set(v['counter'] for v in s['EMA'].values())=={50}
 elif fresh_namespace:assert not s.get('EMA')
 receipt=dict(status='PASS',branch=branch,old_checkpoint=str(p),old_checkpoint_sha256=sha(p),parent_training_checkpoint=src['path'],parent_training_checkpoint_sha256=src['sha256'],old_signature=expected,new_signature=signature('train'),source_plan_sha256=sha(ROOT/'plans/shared.json'),continuation_sha256=sha(ROOT/'plans/continuation.json'),restored_step=s['adaptation_step'],model_state_hashes={k:tensor_hash(m.state_dict()) for k,m in models.items()},optimizer_state_sha256=state_digest(opt.state_dict()),crop_rng_sha256=state_digest(s['crop_rng']),global_rng_sha256=state_digest(s['global_rng']),buffers_sha256=state_digest(buffer_state(models)),buffer_source='explicit buffer map plus state_dict' if 'buffers' in s else 'all named buffers strictly recovered from historical model state_dict',EMA_sha256=state_digest(s.get('EMA')) if branch!=BRANCHES[0] else None,EMA_status='RESTORED' if branch!=BRANCHES[0] else 'N/A',source_optimizer_internal_steps=sorted({int(v['step']) for v in opt.state.values()}),fresh_optimizer=False,optimizer_moments_preserved=True,RNG_preserved=True,optimizer_updates=0,EMA_commits=0,new_namespace=fresh_namespace)
 dump(b/('migration_receipt.json' if fresh_namespace else 'restore_receipt.json'),receipt)
 return s,receipt
