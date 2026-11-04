"""Read-only objective task status; atomic latest controls committed update counts."""
from ab_common import *
def optional(path,default=None):
 p=ROOT/path
 return load(p) if p.exists() else default
def main():
 state=optional('run_status.json',dict(status='NOT_STARTED',stage=None))
 protocol=optional('protocol.json',{});frozen=optional('audits/plans_frozen.json',{}).get('status')=='PASS'
 publication=optional('publication_status.json',{})
 result=dict(status=state['status'],experiment_directory=str(ROOT),protocol_revision=REVISION,
  quarantined_source_groups=protocol.get('quarantined_source_groups') if frozen else None,
  removed_old_training_identities=protocol.get('removed_old_training_identities') if frozen else None,
  N_A=protocol.get('N_A') if frozen else None,N_B=protocol.get('N_B') if frozen else None,
  S=protocol.get('additional_global_updates') if frozen else None)
 result['data_plan_frozen']=frozen
 result['counts_scope']='Frozen protocol and accepted immutable plans' if frozen else 'Source pools/plan under preparation; no immutable-plan acceptance'
 if not frozen and all((ROOT/f'manifests/{a}.json').exists() for a in ('A_clean_old_pool','B_expanded_ulong')):
  result['N_A']=len(load(ROOT/'manifests/A_clean_old_pool.json'));result['N_B']=len(load(ROOT/'manifests/B_expanded_ulong.json'));result['S']=10*((3*result['N_B']+19)//20)
  grouping=optional('graph/final_summary.json',{});result['quarantined_source_groups']=grouping.get('quarantined_groups')
  result['removed_old_training_identities']=optional('historical_exposure.json',{}).get('removed')
 checkpoints=[]
 for label,arm in [('A','A_clean_old_pool'),('B','B_expanded_ulong')]:
  path=ROOT/arm/'checkpoints/latest.pt'
  if path.exists():
   import torch
   committed=torch.load(path,map_location='cpu',weights_only=True)
   assert committed['protocol_sha256']==sha(ROOT/'protocol.json') and committed['parent_checkpoint_sha256']==PARENT_SHA
   additional=committed['additional_step'];absolute=committed['absolute_optimizer_step']
   assert absolute==5000+additional and committed['plan_cursor']==additional
  else:additional=0;absolute=5000
  result[label]=dict(committed_additional_updates=additional,absolute_optimizer_step=absolute)
  checkpoints.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/arm/'checkpoints').glob('*.pt')))
 result['new_training_checkpoints']=checkpoints
 def points(kind):
  return sum(load(p).get('status')=='PASS' for p in (ROOT/f'evaluation/{kind}/points').glob('*.json'))
 reused=optional('evaluation/parent_gtflow/reuse_manifest.json',{})
 result['evaluation']=dict(parent_GTFlow_reused_points=len(reused.get('points',[])) if reused.get('status')=='PASS' else 0,development_completed_points=points('development'),final_test_completed_points=points('final'))
 result.update(report_path=str(ROOT/'REPORT.md'),metrics_paths=[str(p) for p in [ROOT/'evaluation/parent_gtflow/metrics.csv',ROOT/'evaluation/development/metrics.csv',ROOT/'evaluation/final/metrics.csv',ROOT/'metrics.csv',ROOT/'paired_deltas.csv',ROOT/'warping_error.csv'] if p.exists()],audit_paths=[str(p) for p in (ROOT/'audits',ROOT/'graph',ROOT/'split_audit.json') if p.exists()],integrity_path=str(ROOT/'final_integrity.json'),github=dict(branch='codex/tcg-stage1-data-expansion-ab-20261103',commit_url=publication.get('commit_url'),remote_verified=publication.get('remote_verified',False),core_blobs_verified=publication.get('core_blobs_verified',False)),failure_stage=state.get('stage') if state['status'] in ('FAILED','BLOCKED') else None,resume_command=f'env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py',resume_precondition='If a same-task runner is active, it retains execution and a repeated invocation launches no additional work. Otherwise resolve any recorded failure without changing frozen plans/protocol and run the resume command with host CUDA/network access.',current_stage=state.get('stage'),state_updated_unix=state.get('updated'))
 if state.get('completed_sources') is not None:result['source_screen']=dict(completed=state['completed_sources'],expected=state['expected_sources'])
 if state.get('stage')=='native_plan_window_verification':result['plan_window_verification']=dict(completed=state['completed'],expected=state['expected'],trusted_reused=state['trusted_reused'])
 print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
