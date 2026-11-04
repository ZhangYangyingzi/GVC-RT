"""Online development-only stopping, rendezvous evaluation, and resumable A/B scheduling."""
from v3_common import *
from v3_equivalence import launch
import traceback,concurrent.futures
def model_spec(identifier,path,additional,absolute=None,roles=None):
 path=Path(path);return dict(id=identifier,checkpoint_path=str(path),checkpoint_filename=path.name,checkpoint_sha256=sha(path),additional_step=additional,absolute_step=5000+additional if absolute is None else absolute,roles=roles or [identifier])
def heartbeat(stage,**extra):
 update_status(stage,**extra);status=load(ROOT/'run_status.json');status['snapshot_UTC']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
 status['engineering_checks']={k:load(p)['status'] for k,p in [('gradient_equivalence',MIG/'gradient_equivalence.json'),('early_stop_selftest',MIG/'selftest.json')] if p.exists()}
 status['publication']=load(ROOT/'publication_status_v3.json') if (ROOT/'publication_status_v3.json').exists() else dict(status='PENDING')
 dump(ROOT/'logs/live_status_v3.json',status)
 if stage not in ('final_summary','final_integrity','result_publication','complete'):
  body=['# TCG online development early stopping v3 — execution snapshot','',f"Snapshot UTC: {status['snapshot_UTC']}",'This snapshot is not final acceptance. The live controller state is run_status.json and logs/live_status_v3.json.','', '```json',json.dumps(status,indent=2,ensure_ascii=False),'```','', 'Protocol: protocol_v3.json and PROTOCOL_REVISION_V3.md.12710 is a safety cap, not a required number of updates. Each optimizer update retains the original four logical samples/RNG streams,2 UVG/2 U-Long, LR1e-4 and unchanged loss/precision/freezing.','A has an observed legacy four-GPU prefix through additional595/absolute5595; the new scheduling protocol was introduced afterwards. Native_geometry_v1 and old source/protocol/checkpoint chains are retained.','Only complete fixed32-video/QP0/5/9 development evaluations advance patience. M is the mean of QP video-mean FloLPIPS ratios to the common parent5000 baseline; min_delta0.005,patience3. Final test is not used for selection.','Runtime selection, same-budget K_common and per-arm dev-best are separate records. Four logical samples on two physical ranks use SUM/4, one global clip and one optimizer update. A failed engineering probe selects four-GPU alternating segments, not a smaller batch.','No convergence, effectiveness or generalization conclusion is made. Full metrics/report and Git publication require subsequent final integrity.']
  (ROOT/'REPORT.md').write_text('\n'.join(body)+'\n')
def wait(processes,stage,**context):
 def observe_exit(item):
  p,h,original=item;r=dict(original);code=p.wait();ended=time.time();h.close();r.update(exit_code=code,ended_unix=ended,elapsed_seconds=ended-r['started_unix']);r['allocated_GPU_seconds']=len(r.get('physical_GPUs',[]))*r['elapsed_seconds'];return r
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
  futures=[pool.submit(observe_exit,item) for item in processes]
  while not all(f.done() for f in futures):
   heartbeat(stage,active_processes=[r for p,h,r in processes if p.poll() is None],**context);time.sleep(15)
  results=[f.result() for f in futures]
 return results
def command(script,logname,*args):
 handle=(ROOT/'logs'/logname).open('a');cmd=[PYTHON,'-B','-u',str(ROOT/script),*map(str,args)];p=subprocess.Popen(cmd,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
 return p,handle,dict(pid=p.pid,command=cmd,log='logs/'+logname,started_unix=time.time(),physical_GPUs=[4,5,6,7] if script in ('v3_equivalence.py','v3_evaluate.py') else [])
def evaluate(states,kind,splits):
 identity=digest(dict(kind=kind,states=states,splits=splits,protocol=sha(ROOT/'protocol_v3.json'),source=sha(ROOT/'source_manifest_v3.json')))[:20];batch_id=f'{kind}_{identity}'
 path=ROOT/f'evaluation_v3/batches/{batch_id}.json'
 batch=dict(id=batch_id,kind=kind,splits=splits,states=states,protocol_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_sha256=sha(ROOT/'source_manifest_v3.json'),evaluation_rows_sha256=load(ROOT/'protocol.json')['evaluation_rows_sha256'],QP=[0,5,9],fresh_encode=True,independent_decode=True,selection_sha256=sha(ROOT/'selected_models_v3.json') if kind=='final' else None)
 if path.exists():assert load(path)==batch
 else:dump(path,batch)
 out=ROOT/'evaluation_v3'/batch_id;done=out/'run_status.json'
 if not done.exists() or load(done)['status']!='PASS':
  process=command('v3_evaluate.py',f'v3_{batch_id}.log','--batch',path);result=wait([process],'development_evaluation' if kind=='development' else 'final_evaluation',evaluation_batch=batch_id,expected_points=sum(32 if s=='development' else 13 if s=='historical_reference' else 32 for s in splits)*3*len(states))[0]
  assert result['exit_code']==0 and load(done)['status']=='PASS','EVALUATION_EXCEPTION_NOT_PATIENCE'
 assert load(done)['metrics_sha256']==sha(out/'metrics.csv') and load(out/'integrity.json')['status']=='PASS'
 with (out/'metrics.csv').open() as f:records=list(csv.DictReader(f))
 return records,out
def baseline():
 destination=ROOT/'development_baseline_v3.json'
 if destination.exists():
  r=load(destination);assert r['protocol_sha256']==sha(ROOT/'protocol_v3.json') and r['source_manifest_v3_sha256']==sha(ROOT/'source_manifest_v3.json') and sha(ROOT/r['metrics_file'])==r['metrics_sha256'];return r
 spec=model_spec('parent5000',P/'checkpoints/step_5000.pt',0);records,out=evaluate([spec],'development',['development']);score=development_score(records)
 result=dict(**score,protocol_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_v3_sha256=sha(ROOT/'source_manifest_v3.json'),checkpoint_sha256=PARENT_SHA,metrics_file=str((out/'metrics.csv').relative_to(ROOT)),metrics_sha256=sha(out/'metrics.csv'),baseline_not_a_patience_check=True,expected_points=96)
 dump(destination,result)
 for arm in ARMS:
  if not(ROOT/arm/'early_stop_v3.json').exists():dump(ROOT/arm/'early_stop_v3.json',stopping_init())
 return result
def validate_nodes(nodes,base,retroactive=False):
 if not nodes:return
 specs=[model_spec(f'{arm}_add{step}',checkpoint(arm,step),step) for arm,step in nodes]
 records,out=evaluate(specs,'development',['development'])
 for (arm,step),spec in zip(nodes,specs):
  selected=[r for r in records if r['model_state']==spec['id']];score=development_score(selected,base['F_q']);state=early_state(arm)
  receipt=digest(dict(batch_status_sha256=sha(out/'run_status.json'),metrics_sha256=sha(out/'metrics.csv'),checkpoint_sha256=spec['checkpoint_sha256'],score=score))
  new=stopping_update(state,step,score['M'],receipt)
  if len(new['checks'])>len(state['checks']):new['checks'][-1].update(F_q=score['F_q'],checkpoint_sha256=spec['checkpoint_sha256'],checkpoint_path=spec['checkpoint_path'],batch_id=out.name,metrics_sha256=sha(out/'metrics.csv'),metrics_file=str((out/'metrics.csv').relative_to(ROOT)),retroactive=retroactive,actual_training_step_at_evaluation=committed(arm),recorded_unix=time.time())
  if new['stop_reason']:new['actual_stop_step']=committed(arm)
  dump(ROOT/arm/'early_stop_v3.json',new)
  event('V3_DEVELOPMENT_DECISION',arm=arm,additional_step=step,M=score['M'],improvement_anchor=new['improvement_anchor'],best_raw=new['best_raw'],bad_check_count=new['bad_check_count'],stop_reason=new['stop_reason'],retroactive=retroactive)
def train_segment(arm,target,cards):
 lease=(ROOT/arm/'v3_training.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (ROOT/arm/'v3_rank0_active.lock').open('a') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=committed(arm);p,h,r=launch(arm,latest_path(arm),target,cards,logname=f'v3_{arm}_to{target}.log')
 # The rank-zero lifetime lease also guards orphan workers; retain this
 # coordinator lease until the entire torchrun process has exited.
 r.update(arm=arm,start_additional_step=start,target_additional_step=target);dump(ROOT/arm/'active_launcher_v3.json',r)
 return (p,h,r),lease
def select_models():
 path=ROOT/'selected_models_v3.json'
 if path.exists():
  previous=load(path);assert previous['protocol_sha256']==sha(ROOT/'protocol_v3.json')
  for arm in ARMS:assert previous['per_arm'][arm]['early_stop_state_sha256']==sha(ROOT/arm/'early_stop_v3.json')
  for model in previous['states']:assert sha(model['checkpoint_path'])==model['checkpoint_sha256']
  return previous
 assert not list((ROOT/'evaluation_v3').glob('final_*/points/*.json')),'FINAL_TEST_ALREADY_STARTED_WITHOUT_FROZEN_SELECTION'
 states={a:early_state(a) for a in ARMS};assert all(s['stop_reason'] in ('EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED') for s in states.values())
 common=set([0]+[x['step'] for x in states[ARMS[0]]['checks']])&set([0]+[x['step'] for x in states[ARMS[1]]['checks']]);K=max(common)
 specs=[model_spec('parent0',P/'checkpoints/step_0.pt',None,absolute=0),model_spec('parent5000',P/'checkpoints/step_5000.pt',0)]
 for arm,label in zip(ARMS,('A','B')):
  specs += [model_spec(label+'_common',checkpoint(arm,K),K),model_spec(label+'_best_dev',checkpoint(arm,states[arm]['best_raw']['step']),states[arm]['best_raw']['step'])]
 import torch
 from ab_model_context import tensor_hash
 unique=[];signatures={};aliases={}
 for spec in specs:
  cp=torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=True);signature=digest([cp['frozen_state_hash'],tensor_hash(sorted(cp['tcg_model_state'].items()))])
  if signature not in signatures:signatures[signature]=spec['id'];unique.append(dict(spec,model_signature=signature,aliases=[spec['id']]))
  else:next(x for x in unique if x['id']==signatures[signature])['aliases'].append(spec['id'])
  aliases[spec['id']]=signatures[signature]
 result=dict(status='SELECTION_FROZEN_BEFORE_FINAL_TEST',frozen_unix=time.time(),protocol_sha256=sha(ROOT/'protocol_v3.json'),K_common=K,common_selection_rule='largest intersection of completed planned validation steps; independent of quality differences',per_arm={a:dict(stop_reason=s['stop_reason'],actual_additional_updates=committed(a),actual_absolute_optimizer_step=5000+committed(a),best_dev_step=s['best_raw']['step'],best_dev_M=s['best_raw']['M'],theoretical_stop_step=s['theoretical_stop_step'],early_stop_state_sha256=sha(ROOT/a/'early_stop_v3.json')) for a,s in states.items()},states=unique,role_to_unique_state=aliases,expected_final_points=45*3*len(unique),new_test_points=32*3*len(unique),historical_reference_points=13*3*len(unique),test_used_for_selection=False,roles_with_unequal_best_steps_are_not_same_budget=True)
 dump(path,result);return result
def main():
 with lock('runner_v3.lock'),lock('runner.lock'):
  verify_sources();assert load(MIG/'selftest.json')['status']=='PASS'
  # An orphaned same-task segment retains rank-zero's own lifetime lease.
  # Wait for it to reach its saved endpoint rather than launching duplicates.
  while True:
   active=[]
   for arm in ARMS:
    for name in ('v3_training.lock','v3_rank0_active.lock'):
     path=ROOT/arm/name
     if not path.exists():continue
     with path.open('a') as f:
      try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:active.append(str(path.relative_to(ROOT)))
   if not active:break
   heartbeat('tracking_existing_v3_segments',held_leases=active);time.sleep(15)
  if not(MIG/'gradient_equivalence.json').exists():
   result=wait([command('v3_equivalence.py','v3_gradient_equivalence.log')],'runtime_equivalence_v3')[0];assert result['exit_code']==0
  probe=load(MIG/'gradient_equivalence.json')
  selection_path=MIG/'runtime_selection.json'
  selection=dict(mode='two_gpu_parallel_arms' if probe['status']=='PASS' else 'four_gpu_alternating_segments',reason=None if probe['status']=='PASS' else probe.get('fallback_reason',probe.get('reason')),engineering_audit_sha256=sha(MIG/'gradient_equivalence.json'),physical_GPUs={ARMS[0]:[4,5],ARMS[1]:[6,7]} if probe['status']=='PASS' else {a:[4,5,6,7] for a in ARMS},logical_batch_per_update=4,learning_rate=1e-4)
  if selection_path.exists():assert load(selection_path)==selection
  else:dump(selection_path,selection)
  publication=ROOT/'publication_status_v3.json'
  if not publication.exists() or not load(publication).get('code_commit_remote_verified'):
   r=wait([command('v3_publish.py','v3_publish_code.log','--phase','code')],'code_protocol_publication')[0]
   if r['exit_code']!=0:event('V3_CODE_PUBLICATION_FAILED_LOCAL_WORK_CONTINUES',log=r['log'])
  base=baseline()
  # Retrospective checks use already saved checkpoints; no prefix rollback.
  for arm in ARMS:
   already=committed(arm);past_nodes=list(range(500,already+1,500))+([12710] if already==12710 else [])
   for node in past_nodes:
    if early_state(arm)['stop_reason']:break
    if node>early_state(arm)['last_validation_step']:validate_nodes([(arm,node)],base,retroactive=True)
  while any(early_state(a)['stop_reason'] is None for a in ARMS):
   if load(ROOT/'control_v3.json').get('pause_requested'):dump(ROOT/'run_status.json',dict(status='CONTROL_PAUSED',stage='between_segments',protocol_revision=V3,updated=time.time()));return 2
   targets={a:min((committed(a)//500+1)*500,12710) for a in ARMS if early_state(a)['stop_reason'] is None}
   if selection['mode']=='two_gpu_parallel_arms':
    running=[];leases=[]
    try:
     for arm,target in targets.items():process,lease=train_segment(arm,target,selection['physical_GPUs'][arm]);running.append(process);leases.append(lease)
     results=wait(running,'parallel_training',runtime_selection=selection,segment_targets=targets)
    finally:
     for lease in leases:lease.close()
    history=load(ROOT/'training_segments_v3.json') if (ROOT/'training_segments_v3.json').exists() else [];history.extend(results);dump(ROOT/'training_segments_v3.json',history)
    assert all(r['exit_code']==0 for r in results),'TRAINING_SEGMENT_FAILED'
    for arm,target in targets.items():
     if committed(arm)!=target:dump(ROOT/'run_status.json',dict(status='CONTROL_PAUSED',stage='checkpoint_boundary',updated=time.time()));return 2
    validate_nodes(list(targets.items()),base)
   else:
    for arm,target in targets.items():
     process,lease=train_segment(arm,target,[4,5,6,7])
     try:results=wait([process],'alternating_training',runtime_selection=selection,segment_targets={arm:target})
     finally:lease.close()
     history=load(ROOT/'training_segments_v3.json') if (ROOT/'training_segments_v3.json').exists() else [];history.extend(results);dump(ROOT/'training_segments_v3.json',history)
     assert results[0]['exit_code']==0,'TRAINING_SEGMENT_FAILED'
     if committed(arm)!=target:dump(ROOT/'run_status.json',dict(status='CONTROL_PAUSED',stage='checkpoint_boundary',updated=time.time()));return 2
     validate_nodes([(arm,target)],base)
  selected=select_models();records,out=evaluate(selected['states'],'final',['historical_reference','new_holdout_test']);dump(ROOT/'final_evaluation_v3.json',dict(batch_id=out.name,metrics_file=str((out/'metrics.csv').relative_to(ROOT)),metrics_sha256=sha(out/'metrics.csv'),selection_sha256=sha(ROOT/'selected_models_v3.json'),completed_points=len(records),expected_points=selected['expected_final_points']))
  for script,stage in [('v3_finalize.py','final_summary'),('v3_integrity.py','final_integrity'),('v3_publish.py','result_publication')]:
   arguments=['--phase','results'] if script=='v3_publish.py' else []
   result=wait([command(script,f'{stage}_v3.log',*arguments)],stage)[0];assert result['exit_code']==0,(stage,'FAILED')
  pub=load(ROOT/'publication_status_v3.json');assert pub['results_remote_verified'] and pub['core_blobs_verified'] and load(ROOT/'final_integrity_v3.json')['status']=='PASS'
  dump(ROOT/'run_status.json',dict(status='PASS',stage='complete',protocol_revision=V3,updated=time.time(),verified_commit=pub['results_commit']));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BlockingIOError:print('Existing same-task controller/stage lease held; no duplicate started.',flush=True)
 except Exception:
  state=load(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {};dump(MIG/f'failure_{time.time_ns()}.json',dict(previous_state=state,error=traceback.format_exc()));state.update(status='FAILED',error=traceback.format_exc(),updated=time.time());dump(ROOT/'run_status.json',state);traceback.print_exc();sys.exit(1)
