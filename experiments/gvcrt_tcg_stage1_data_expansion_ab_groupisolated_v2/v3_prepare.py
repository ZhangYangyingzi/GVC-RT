"""Freeze this mid-training protocol revision without changing any immutable data plan."""
from v3_common import *
import ast,shutil
def main():
 freeze=MIG/'protocol_freeze.json'
 if freeze.exists():
  r=load(freeze);assert sha(ROOT/'protocol_v3.json')==r['protocol_sha256'] and sha(ROOT/'source_manifest_v3.json')==r['source_manifest_v3_sha256'];verify_sources();print('Existing v3 protocol/code freeze verified; no data work repeated.');return
 assert not list((ROOT/'evaluation_v3').glob('*/points/*.json'))
 old=load(ROOT/'protocol.json');base=load(ROOT/'source_manifest.json');data=load(ROOT/'audits/plans_frozen.json');takeover=load(MIG/'takeover.json')
 assert sha(ROOT/'protocol.json')==data['protocol_sha256'] and (old['N_A'],old['N_B'],old['additional_global_updates'])==(923,8470,12710)
 assert not takeover['old_processes_remaining_after_SIGTERM']
 for name,h in base['code'].items():assert sha(ROOT/name)==h
 for arm in ARMS:
  assert sha(ROOT/f'plans/{arm}/index.json')==old['plan_index_sha256'][arm]
  for shard in load(ROOT/f'plans/{arm}/index.json')['shards']:assert sha(ROOT/shard['file'])==shard['sha256']
 assert sha(ROOT/'manifests/evaluation_rows.json')==old['evaluation_rows_sha256']
 p=dict(old)
 for key in ('additional_global_updates','absolute_final_optimizer_step','A_then_B_serial','world_size','checkpoint_additional_steps','development_steps','primary_checkpoint','no_checkpoint_selection','coverage_passes','padding_ulong_samples'):p.pop(key,None)
 p.update(protocol_revision=V3,predecessor_protocol_file='protocol.json',predecessor_protocol_sha256=sha(ROOT/'protocol.json'),max_additional_updates=12710,absolute_max_optimizer_step=17710,cap_is_not_required_budget=True,min_required_coverage_passes=0,planned_coverage_passes_at_cap=3,padding_ulong_samples_at_cap=10,logical_world_size=4,effective_batch_size=4,preferred_physical_world_size_per_arm=2,preferred_GPU_assignment={ARMS[0]:[4,5],ARMS[1]:[6,7]},logical_to_physical_rank='logical_rank % physical_world_size',gradient_reduction='sum of four logical gradients /4; one global clip1 and optimizer.step; no extra world-size division',logical_RNG_streams=4,prefer_parallel_arms=True,fallback='four_gpu_alternating_segments_after_each_complete_validation_node',numerical_acceptance_relative_L2=.001,bitwise_equivalence_not_assumed=True,validation_interval_additional_updates=500,validation_safety_cap_endpoint=12710,development_split='existing frozen32 development videos, same64-frame windows and QP0/5/9',development_rows_sha256=old['evaluation_rows_sha256'],stopping=dict(metric='FloLPIPS',mode='min',aggregation='For each QP, equal-video mean over32 sequences; M=mean_q(F_q(checkpoint)/F_q(parent5000)); per-video time aggregation unchanged',min_delta=.005,min_delta_definition='absolute decrease in baseline-normalized M, not relative to current anchor',patience=3,baseline_counts_for_patience=False,baseline_M=1.,best_raw_includes_parent=True,best_raw_tie='earlier additional step',substantive_improvement='M < improvement_anchor - 0.005',independent_per_arm=True,invalid_or_incomplete_validation='exception; never a bad check'),no_checkpoint_selection=False,model_selection='independent best_raw development M, plus largest-common-planned-validation-step comparison',final_selection_frozen_before_test=True,final_test_not_used_for_stopping=True,final_unique_states='parent0,parent5000,A_common,B_common,A_best_dev,B_best_dev; deduplicate same model/checkpoint',final_point_count='45 videos *3 QPs *actual unique states',old_training_prefix={a:takeover['frozen_checkpoints'][a]['additional_step'] for a in ARMS},legacy_runtime_description='A four-GPU serial prefix before this mid-training revision; not v3 from step1',retroactive_validation_policy='Use saved passed-node checkpoints, record theoretical and actual stop; never rollback committed updates',rendezvous='Exit all training/RAFT workers before evaluation; retain surviving arm effective batch4',checkpoint_policy='atomic latest each update; retain0,2,reached500-node checkpoints,cap if reached,and migration prefix',pause_control='control_v3.json pause_requested; stop only after an atomic committed update',GPU_time_definition='Observed assigned-card wall seconds including process startup/I-O, not kernel utilization time')
 dump(ROOT/'protocol_v3.json',p)
 names=sorted(x.name for x in ROOT.glob('v3_*.py'))+['launch_v3.py']
 assert {'v3_common.py','v3_runtime.py','v3_train.py','v3_equivalence.py','v3_evaluate.py','v3_controller.py','v3_finalize.py','v3_integrity.py','v3_publish.py','v3_prepare.py','v3_selftest.py','v3_status.py','launch_v3.py'}<=set(names)
 for name in names:assert (ROOT/name).is_file(),name;ast.parse((ROOT/name).read_text())
 source=dict(protocol_revision=V3,protocol_sha256=sha(ROOT/'protocol_v3.json'),base_source_manifest_file='source_manifest.json',base_source_manifest_sha256=sha(ROOT/'source_manifest.json'),base_code=base['code'],code={n:sha(ROOT/n) for n in names},predecessor_data_plan_index_sha256=old['plan_index_sha256'],frozen_evaluation_rows_sha256=old['evaluation_rows_sha256'],native_geometry_v1_retained=True,takeover_sha256=sha(MIG/'takeover.json'))
 dump(ROOT/'source_manifest_v3.json',source)
 import torch
 parent=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True);assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
 bootstrap={}
 for arm in ARMS:
  record=takeover['frozen_checkpoints'][arm]
  if record['checkpoint']:
   path=ROOT/record['checkpoint'];assert sha(path)==record['sha256'];state=torch.load(path,map_location='cpu',weights_only=True);prefix=state['additional_step']
  else:path=P/'checkpoints/step_5000.pt';state=torch.load(path,map_location='cpu',weights_only=True);prefix=0
  inherited_protocol=state['protocol_sha256'];state.update(additional_step=prefix,committed_global_updates=5000+prefix,absolute_optimizer_step=5000+prefix,plan_cursor=prefix,rank_sample_cursor=4*prefix,arm=arm,active_protocol_revision=V3,protocol_sha256=sha(ROOT/'protocol_v3.json'),inherited_checkpoint_protocol_sha256=inherited_protocol,predecessor_protocol_sha256=sha(ROOT/'protocol.json'),source_hashes_v3=source['code'],base_source_hashes=base['code'],parent_checkpoint_sha256=PARENT_SHA,plan_index_sha256=old['plan_index_sha256'][arm],logical_world_size=4,physical_world_size=None)
  assert len(state['all_rank_RNG_states'])==4 and {int(v['step']) for v in state['optimizer']['state'].values()}=={5000+prefix}
  if prefix==0:state['last_committed_receipt']=dict(event='PARENT5000_INITIALIZATION',additional_step=0,no_optimizer_update=True)
  destination=ROOT/arm/f'checkpoints_v3/step_{prefix}.pt';assert not destination.exists() and not latest_path(arm).exists();atomic_torch(destination,state);atomic_torch(latest_path(arm),state)
  bootstrap[arm]=dict(legacy_prefix_additional_step=prefix,absolute_optimizer_step=5000+prefix,inherited_checkpoint_filename=path.name,inherited_checkpoint_sha256=sha(path),seed_checkpoint=str(destination.relative_to(ROOT)),seed_checkpoint_sha256=sha(destination),all_four_logical_RNG_preserved=True,optimizer_preserved=True,metadata_migration_only=True,formal_optimizer_updates_added=0)
 dump(MIG/'bootstrap.json',dict(protocol_revision=V3,arms=bootstrap))
 # Legacy allocated-card intervals are observed from actual coordinator logs.
 starts=[]
 for line in (ROOT/'logs/runner.log').open():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('event')=='STAGE_STARTED' and r.get('stage')=='train_A':starts.append(r['time'])
 intervals=[]
 for began in starts:
  failures=[int(f.name.split('_')[1]) for f in (ROOT/'audits').glob('failure_*_train_A.json') if int(f.name.split('_')[1])>=began]
  end=min(failures) if failures else takeover['stopped_unix'];assert end>=began;intervals.append(dict(started_unix=began,ended_unix=end,wall_seconds=end-began,physical_GPUs=[4,5,6,7]))
 dump(MIG/'legacy_gpu_time.json',{ARMS[0]:dict(phase_wall_seconds=sum(x['wall_seconds'] for x in intervals),allocated_GPU_seconds=4*sum(x['wall_seconds'] for x in intervals),intervals=intervals,basis='Observed coordinator train_A stage starts through recorded failure/control-stop, four assigned cards; not kernel-active time'),ARMS[1]:dict(phase_wall_seconds=0,allocated_GPU_seconds=0,intervals=[])})
 dump(ROOT/'control_v3.json',dict(pause_requested=False));dump(ROOT/'ACTIVE_PROTOCOL.json',dict(protocol_revision=V3,active_protocol='protocol_v3.json',source_manifest='source_manifest_v3.json',runner='v3_controller.py',launcher='launch_v3.py',legacy_protocol_read_only='protocol.json',active_checkpoint_layout='<arm>/checkpoints_v3/latest.pt'))
 dump(freeze,dict(status='FROZEN_BEFORE_ANY_NEW_DEVELOPMENT_SCORE',frozen_unix=time.time(),protocol_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_v3_sha256=sha(ROOT/'source_manifest_v3.json'),new_development_point_count_at_freeze=0,new_test_point_count_at_freeze=0,prior_plans_and_splits_unchanged=True,prior_native_geometry_chain_preserved=True))
 print(json.dumps(dict(status='PREPARED',protocol_revision=V3,bootstrap=bootstrap,protocol_sha256=sha(ROOT/'protocol_v3.json'))))
if __name__=='__main__':main()
