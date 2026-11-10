"""Complete audited B300; full Bridge, three TCG and five existing LoRA branches."""
from common_s2 import *
import torch
from b300_loader import construct as load_b300,detach_history
from model_context import assert_reset

def category(n):
 if n.startswith('recon_generation_net.mlp.'):return 'Bridge'
 if n.startswith(tuple('recon_generation_net.decoder.'+p for p in TCG_PREFIXES)):return 'TCG'
 if n.startswith('recon_generation_net.decoder.') and '.lora.' in n:return 'LoRA'

def construct(group=GROUPS[0],train=True,rank=0):
 assert group in [*GROUPS,'B300','A2000'];c=load_b300('A2000' if group=='A2000' else 'B300',train,rank);c['experiment_group']=group;c['optimizer']=None;c['ctx']=None;c['initial_trainable']={}
 for p in c['pm'].parameters():p.requires_grad_(False)
 named=[];sets={k:[] for k in ['Bridge','TCG','LoRA']};seen=set()
 for n,p in c['pm'].named_parameters():
  k=category(n)
  if k:
   assert id(p) not in seen;seen.add(id(p));p.requires_grad_(train);named.append((n,p));sets[k].append(p)
 assert {k:sum(p.numel() for p in ps) for k,ps in sets.items()}==dict(Bridge=801124,TCG=137760,LoRA=59392)
 assert len([m for m in c['models']['generator'].modules() if m.__class__.__name__=='ParallelConvLoRA'])==5
 rates=dict(Bridge=2e-6,TCG=1e-5,LoRA=5e-5)
 if train:c['optimizer']=torch.optim.AdamW([dict(params=ps,lr=rates[k],group_name=k) for k,ps in sets.items()],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 c.update(named=named,params=[p for n,p in named],parameter_sets=sets,trainable_parameter_names={n for n,p in named},alignment_enabled=False)
 c['provenance'].update(experiment_group=group,fresh_AdamW=True,optimizer_restored=False,trainable_parameter_counts={k:sum(p.numel() for p in ps) for k,ps in sets.items()},s_train=.5,s_eval=.5,LoRA_removed_or_merged=False,Teacher_present=False)
 return c

def bridge(c,feature):
 m=c['models']['bridge'];x=torch.nn.functional.pixel_unshuffle(feature.detach().to(c['device'],next(m.parameters()).dtype),2);x=m[2](m[1](m[0](x)));return m[3](x*torch.sigmoid(x))

def reconstruct(c,feature,qp,retain=False,latent=False):
 gen=c['models']['generator'];gen.set_temporal_cascade_grad(retain)
 for n in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,n).alignment=None
 z=bridge(c,feature);q=c['pm'].q_scale_recon[qp:qp+1].to(z.dtype);out=gen(z,q,use_tcg=True,update_buffer=True)
 return (out,z) if latent else out

def auxiliary(c,feature,qp):
 z=bridge(c,feature);q=c['pm'].q_scale_recon[qp:qp+1].to(z.dtype)
 return (c['models']['generator'](z,q,use_tcg=False,update_buffer=False).float().clamp(-1,1)+1)/2

def cache_snapshot(c):
 return {n:(getattr(c['models']['generator'],n)._previous,None if getattr(c['models']['generator'],n)._previous is None else getattr(c['models']['generator'],n)._previous.detach().clone()) for n in ['tcg_low','tcg_mid','tcg_out']}

def assert_cache(c,s):
 for n,(obj,value) in s.items():
  now=getattr(c['models']['generator'],n)._previous;assert now is obj
  if now is not None:assert torch.equal(now.detach(),value)

def trainable_state(c):return {n:p.detach().cpu().clone() for n,p in c['named']}

def restore(c,a,optimizer=True):
 assert a['B300_SHA256']==B300_SHA and a['experiment_group']==c['experiment_group'] and set(a['trainable_state'])==c['trainable_parameter_names']
 full=c['pm'].state_dict();full.update(a['trainable_state']);c['pm'].load_state_dict(full,strict=True)
 if optimizer:
  names={id(p):n for n,p in c['named']};assert a['optimizer_name_groups']=={g['group_name']:[names[id(p)] for p in g['params']] for g in c['optimizer'].param_groups};c['optimizer'].load_state_dict(a['optimizer'])
 assert_reset(c)
