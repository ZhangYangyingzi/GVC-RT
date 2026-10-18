"""Independent CPU verification of migrated snapshots; no forward or optimizer call."""
from io32 import *
from migration33 import state_digest

def main():
 import torch
 checks=[]
 for b in BRANCHES:
  source=load(ROOT/'config.json')['branch_sources'][b];assert sha(source['path'])==source['sha256'];old=torch.load(source['path'],map_location='cpu',weights_only=True);bp=ROOT/'branches'/b
  entry=load(bp/'checkpoint_index.json')['1000'];assert sha(entry['path'])==entry['sha256'];new=torch.load(entry['path'],map_location='cpu',weights_only=True)
  assert new['signature']==signature('train') and new['adaptation_step']==1000 and new['additional_step']==0 and new['replay_cursor']==0
  fields=['wrapper','bridge','generator','optimizer','crop_rng','global_rng','visits','frozen_hashes'];same={k:state_digest(old[k])==state_digest(new[k]) for k in fields};assert all(same.values()),(b,same)
  if b!=BRANCHES[0]:assert state_digest(old['EMA'])==state_digest(new['EMA']) and state_digest(old['buffers'])==state_digest(new['buffers'])
  else:
   assert new['EMA'] is None
   for k,buf in new['buffers'].items():
    for n,t in buf.items():assert n in old[k] and torch.equal(t,old[k][n])
  first=load(ROOT/'audits'/f'first_committed_step_{b}.json');assert first['adaptation_step']==1001 and first['additional_step']==1 and first['source_plan_step']==1 and first['optimizer_internal_step']==1001
  assert first['gradient_formula_error']<=1e-6 and first['postclip_scaling_error']<=1e-6 and all(x['norm']>0 and x['finite'] for x in first['parameter_delta'].values())
  assert first['total_EMA_commits']==(None if b==BRANCHES[0] else 1001)
  checks.append(dict(branch=b,status='PASS',source_sha256=source['sha256'],new_snapshot_sha256=entry['sha256'],exact_state_fields=same,buffers_exact=True,EMA_exact=True if b!=BRANCHES[0] else 'N/A',first_new_step=1001,first_new_optimizer_internal_step=1001,first_preclip_merge_error=first['gradient_formula_error'],first_postclip_scaling_error=first['postclip_scaling_error'],first_parameter_delta=first['parameter_delta']));del old,new
  print('MIGRATION AND FIRST COMMITTED STEP PASS',b,flush=True)
 dump(ROOT/'audits/migration_and_first_step.json',dict(status='PASS',optimizer_updates_by_this_audit=0,EMA_commits_by_this_audit=0,checks=checks))
if __name__=='__main__':main()
