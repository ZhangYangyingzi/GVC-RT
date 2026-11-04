"""Independent stopping/selection, logical-sample, RNG, stream and dynamic-budget audit."""
from v3_common import *
from v3_evaluate import audit_batch
def main():
 import torch
 verify_sources();p=protocol();freeze=load(MIG/'protocol_freeze.json');assert freeze['protocol_sha256']==sha(ROOT/'protocol_v3.json') and freeze['source_manifest_v3_sha256']==sha(ROOT/'source_manifest_v3.json')
 assert p['max_additional_updates']==12710 and p['stopping']['patience']==3 and p['stopping']['min_delta']==.005 and p['min_required_coverage_passes']==0
 selected=load(ROOT/'selected_models_v3.json');mode=load(MIG/'runtime_selection.json');bootstrap=load(MIG/'bootstrap.json');base=load(ROOT/'development_baseline_v3.json');assert base['M']==1 and all(x>0 and math.isfinite(x) for x in base['F_q'].values())
 parent=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True);assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 original_A0=torch.load(ROOT/'A_clean_old_pool/checkpoints/step_0.pt',map_location='cpu',weights_only=True)
 B0=torch.load(ROOT/'B_expanded_ulong/checkpoints_v3/step_0.pt',map_location='cpu',weights_only=True)
 for key in ('tcg_model_state','optimizer','all_rank_RNG_states'):assert equal(original_A0[key],parent[key]) and equal(B0[key],parent[key])
 arm_audits={};common_sets=[]
 for arm in ARMS:
  state=early_state(arm);n=committed(arm);assert state['stop_reason'] in ('EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED') and 0<n<=12710
  cp=torch.load(latest_path(arm),map_location='cpu',weights_only=True);assert cp['protocol_sha256']==sha(ROOT/'protocol_v3.json') and cp['source_hashes_v3']==load(ROOT/'source_manifest_v3.json')['code']
  assert cp['absolute_optimizer_step']==5000+n and cp['rank_sample_cursor']==4*n and cp['logical_world_size']==4 and len(cp['all_rank_RNG_states'])==4
  assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={5000+n} and all(v is None for b in cp['TCG_buffers'] for v in b.values())
  plan=rows(arm);prefix=bootstrap['arms'][arm]['legacy_prefix_additional_step']
  for step in range(1,n+1):
   receipt=load(ROOT/arm/('receipts' if step<=prefix else 'receipts_v3')/f'{step:06d}.json.gz');assert receipt['additional_step']==step and receipt['absolute_optimizer_step']==5000+step and receipt['rank_sample_cursor']==4*step
   assert receipt['rank_sync']=='PASS' and receipt['finite_loss'] and receipt['finite_nonzero_gradient'] and receipt['only_TCG_changed'] and receipt['buffer_reset']
   audits=receipt['rank_audits'];assert len(audits)==4 and {a['rank'] for a in audits}==set(range(4)) and len({a['TCG_hash'] for a in audits})==1
   for a in audits:
    row=plan[4*(step-1)+a['rank']];assert a['sample_sha256']==digest(row) and a['source_sha256']==row['source_sha256'] and a['external_qp']==row['external_qp'] and a['actual_qps']==row['actual_frame_qps']
    assert a['frozen_hash']==parent['frozen_state_hash'] and a['frozen_check']=='EXACT_DEVICE_BYTES_PASS' and a['non_TCG_delta_norm']==0 and a['TCG_delta_norm']>0 and a['buffer_reset']
    assert all(math.isfinite(v) for v in a['losses'].values()) and a['local_gradient_norm']>0 and math.isfinite(a['local_gradient_norm'])
    if step>prefix:
     cards=mode['physical_GPUs'][arm];world=len(cards);assert a['logical_rank']==a['rank'] and a['physical_rank']==a['rank']%world and a['physical_GPU']==cards[a['physical_rank']]
     assert a['auxiliary_GT_flow']['GPU_mapping']['physical_GPU']==a['physical_GPU']
     assert receipt['logical_world_size']==receipt['effective_batch_size']==4 and receipt['physical_world_size']==world
   assert collections.Counter(a['domain'] for a in audits)=={'uvg':2,'ulong':2}
  assert len(list((ROOT/arm/'receipts_v3').glob('*.json.gz')))==n-prefix
  replay=stopping_init()
  for check in state['checks']:
   metrics=ROOT/check['metrics_file'];assert sha(metrics)==check['metrics_sha256']
   with metrics.open() as f:records=[r for r in csv.DictReader(f) if r['model_state']==f'{arm}_add{check["step"]}']
   score=development_score(records,base['F_q']);assert score['M']==check['M'] and score['F_q']==check['F_q']
   assert sha(check['checkpoint_path'])==check['checkpoint_sha256'] and 0<check['step']<=n and (check['step']%500==0 or check['step']==12710)
   replay=stopping_update(replay,check['step'],check['M'],check['evaluation_sha256']);assert replay['checks'][-1]['bad_check_count']==check['bad_check_count'] and replay['best_raw']==check['best_raw'] and replay['improvement_anchor']==check['improvement_anchor']
  for key in ('best_raw','improvement_anchor','bad_check_count','stop_reason','theoretical_stop_step','last_validation_step'):assert replay[key]==state[key]
  last=state['theoretical_stop_step'];required=list(range(500,last+1,500))+([12710] if last==12710 else []);assert [c['step'] for c in state['checks']]==required
  assert selected['per_arm'][arm]['early_stop_state_sha256']==sha(ROOT/arm/'early_stop_v3.json')
  assert selected['per_arm'][arm]['best_dev_step']==state['best_raw']['step'] and selected['per_arm'][arm]['actual_additional_updates']==n
  common_sets.append({0,*[c['step'] for c in state['checks']]})
  arm_audits[arm]=dict(status='PASS',stop_reason=state['stop_reason'],actual_additional_updates=n,absolute_optimizer_step=5000+n,logical_samples=4*n,legacy_prefix=prefix,best_dev_step=state['best_raw']['step'],patience_count=state['bad_check_count'])
 assert selected['K_common']==max(common_sets[0]&common_sets[1])
 from ab_model_context import tensor_hash
 byid={m['id']:m for m in selected['states']}
 expected_roles={'parent0':P/'checkpoints/step_0.pt','parent5000':P/'checkpoints/step_5000.pt'}
 for arm,label in zip(ARMS,('A','B')):expected_roles.update({label+'_common':checkpoint(arm,selected['K_common']),label+'_best_dev':checkpoint(arm,early_state(arm)['best_raw']['step'])})
 for role,path in expected_roles.items():
  target=torch.load(path,map_location='cpu',weights_only=True);signature=digest([target['frozen_state_hash'],tensor_hash(sorted(target['tcg_model_state'].items()))])
  assert byid[selected['role_to_unique_state'][role]]['model_signature']==signature
 assert load(ROOT/'stream_comparison_v3.json')['status']=='PASS'
 expected_models=len(selected['states']);assert selected['expected_final_points']==45*3*expected_models
 all_batches=[];final_count=0;dev_count=0
 for manifest in sorted((ROOT/'evaluation_v3/batches').glob('*.json')):
  batch=load(manifest);directory=ROOT/'evaluation_v3'/batch['id'];done=directory/'run_status.json'
  if not done.exists() or load(done)['status']!='PASS':raise AssertionError(('INCOMPLETE_EVALUATION_BATCH',batch['id']))
  result=audit_batch(batch,directory);assert result['status']=='PASS'
  if batch['kind']=='final':
   assert batch['selection_sha256']==sha(ROOT/'selected_models_v3.json');final_count+=result['points']
   for point in (directory/'points').glob('*.json'):
    r=load(point);assert r['selection_sha256']==sha(ROOT/'selected_models_v3.json') and r['evaluated_unix']>=selected['frozen_unix']
  else:dev_count+=result['points']
  all_batches.append(dict(batch=batch['id'],kind=batch['kind'],points=result['points']))
 assert final_count==selected['expected_final_points']
 split=load(ROOT/'split_audit.json');assert split['status']=='PASS' and not(split['train_dev_test_group_overlap'] or split['new_dev_test_historical_group_overlap'])
 from integrity import validate_groups
 group_audit=validate_groups({arm:rows(arm) for arm in ARMS},load(ROOT/'protocol.json'));assert group_audit['status']=='PASS'
 assert load(MIG/'selftest.json')['status']=='PASS'
 proof=load(MIG/'gradient_equivalence.json')
 if mode['mode']=='two_gpu_parallel_arms':assert proof['status']=='PASS'
 else:assert mode['engineering_audit_sha256']==sha(MIG/'gradient_equivalence.json') and mode['reason']
 assert load(ROOT/'evaluation/parent_gtflow/reuse_manifest.json')['status']=='PASS'
 dump(ROOT/'final_integrity_v3.json',dict(status='PASS',protocol_revision=V3,protocol_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_sha256=sha(ROOT/'source_manifest_v3.json'),arms=arm_audits,K_common=selected['K_common'],largest_common_step_not_quality_selected=True,independent_patience_replay=True,parent_included_raw_best_and_tie_rules=True,four_logical_RNG_and_batch=True,actual_sample_order=True,frozen_state_checks=True,dynamic_final_state_deduplication=True,final_test_selection_locked=True,source_group_isolation=True,no_test_selection=True,development_points=dev_count,final_points=final_count,evaluation_batches=all_batches,legal_early_stop_is_not_convergence_claim=True))
if __name__=='__main__':main()
