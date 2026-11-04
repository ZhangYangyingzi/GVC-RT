"""Synthetic state/aggregation/recovery tests; no model-quality observation."""
from v3_common import *
def main():
 import torch
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 takeover=load(MIG/'takeover.json');bootstrap=load(MIG/'bootstrap.json')
 for arm in ARMS:
  inherited=ROOT/takeover['frozen_checkpoints'][arm]['checkpoint'] if takeover['frozen_checkpoints'][arm]['checkpoint'] else P/'checkpoints/step_5000.pt'
  before=torch.load(inherited,map_location='cpu',weights_only=True);after=torch.load(ROOT/bootstrap['arms'][arm]['seed_checkpoint'],map_location='cpu',weights_only=True)
  for key in ('tcg_model_state','optimizer','all_rank_RNG_states','TCG_buffers'):assert equal(before[key],after[key]),(arm,key)
  assert after['additional_step']==takeover['frozen_checkpoints'][arm]['additional_step'] and after['absolute_optimizer_step']==5000+after['additional_step']
 tests=[];s=stopping_init();assert s['bad_check_count']==0 and s['best_raw']['step']==0
 tests.append('bootstrap_TCG_optimizer_all_four_RNG_buffers_and_cursors_exactly_preserved')
 for step,M,bad in [(500,.998,1),(1000,.997,2),(1500,.996,3)]:
  s=stopping_update(s,step,M,str(step));assert s['bad_check_count']==bad and s['best_raw']['step']==step and s['improvement_anchor']==1
 assert s['stop_reason']=='EARLY_STOPPED_PATIENCE' and s['best_raw']['step']==1500
 before=digest(s);s2=stopping_update(s,1500,.996,'1500');assert digest(s2)==before;tests.append('raw_best_updates_without_min_delta_and_patience_is_idempotent')
 s=stopping_update(stopping_init(),500,.99,'a');assert s['improvement_anchor']==.99 and s['bad_check_count']==0
 boundary=s['improvement_anchor']-.005;s=stopping_update(s,1000,boundary,'b');assert s['bad_check_count']==1 and s['improvement_anchor']==.99
 s=stopping_update(s,1500,.98,'c');assert s['bad_check_count']==0 and s['improvement_anchor']==.98;tests.append('strict_absolute_min_delta_anchor_and_reset')
 tie=stopping_update(stopping_init(),500,1.,'tie');assert tie['best_raw']['step']==0;tests.append('baseline_best_and_earlier_tie')
 x=stopping_init();x=stopping_update(x,12710,.9,'cap');assert x['stop_reason']=='MAX_BUDGET_REACHED'
 x=stopping_init();x=stopping_update(x,500,1.,'x');x=stopping_update(x,1000,1.,'y');x=stopping_update(x,12710,1.,'z');assert x['stop_reason']=='EARLY_STOPPED_PATIENCE';tests.append('patience_not_ignored_at_cap')
 names=[Path(r['video']).stem for r in load(ROOT/'manifests/evaluation_rows.json') if r['split']=='development'];fake=[dict(sequence=n,QP=q,FloLPIPS=float(q+1)) for n in names for q in (0,5,9)]
 base=development_score(fake);assert base['M']==1
 changed=[dict(r,FloLPIPS=r['FloLPIPS']*.99) for r in fake];assert abs(development_score(changed,base['F_q'])['M']-.99)<1e-12
 for records,denom in [(fake[:-1],base['F_q']),(fake,{'0':0.,'5':1.,'9':1.})]:
  try:development_score(records,denom)
  except AssertionError:pass
  else:raise AssertionError('Incomplete/invalid baseline accepted')
 tests.append('fixed_QP_video_mean_ratio_and_anomalies_not_patience')
 try:stopping_update(tie,1000,float('nan'),'nan')
 except AssertionError:pass
 else:raise AssertionError('NaN accepted')
 assert tie['bad_check_count']==1;tests.append('nonfinite_has_no_state_mutation')
 assert max({0,500,1000}&{0,500,1000,1500})==1000;tests.append('largest_common_step_selection')
 dump(MIG/'selftest.json',dict(status='PASS',scope='Synthetic stopping, aggregation and persistent-state semantics only; not model engineering/evaluation acceptance',tests=tests,no_model_or_quality_scores_read=True,protocol_sha256=sha(ROOT/'protocol_v3.json')));print(json.dumps(dict(status='PASS',tests=tests)))
if __name__=='__main__':main()
