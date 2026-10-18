"""Persist every trial; select independent global QPs using rate fields only."""
from protocol import *
import collections,struct,traceback

def validate_new():
 import numpy as np
 records=[];errors=[]
 for p in sorted((ROOT/'manifests/receipts').glob('*.json')):
  r=load(p)
  if r['status']!='PASS':continue
  try:
   j=r['job'];x=r['result'];a=r['rate'];data=Path(x['bitstream_path']).read_bytes()
   assert sha(x['bitstream_path'])==x['bitstream_sha256'] and len(data)*8==a['actual_total_bits']
   assert data[:8]==b'GROV1\0\0\0';n=struct.unpack('>H',data[8:10])[0];header=10+11*n
   decoded=[]
   for i in range(n):
    f,y,xx,h,w,level=struct.unpack('>5HB',data[10+11*i:21+11*i]);decoded.append(dict(frame=f,y=y,x=xx,h=h,w=w,factor=(1.,.5,2.)[level]))
   assert decoded==j['plan']
   native=Path(x['bitstream_path']).with_name('native.bin');assert native.read_bytes()==data[header:]
   assert sha(native)==x['native_bitstream_sha256']
   assert len(data)*8==sum(a[k] for k in ('RANS_payload_bits','native_SPS_IP_header_bits','control_header_bits'))
   assert a['control_header_bits']==header*8
   assert x['frames']==16 and x['num_transitions']==15 and x['bytes_consumed']==len(data)
   assert r['reference_state_exact_all_P_frames'] and r['independent_decode'] and len(r['frozen_weight_checks'])==3
   assert all(c['unchanged'] for c in r['frozen_weight_checks'])
   assert len(r['decoder_symbol_checks'])==4 and all(g['exact'] for f in r['decoder_symbol_checks'] for g in f['groups'])
   for frame,tensor in r['tensor_artifacts'].items():
    assert sha(tensor['path'])==tensor['sha256']
    with np.load(tensor['path']) as z:
     assert z['y'].shape==z['symbols'].shape==(1,128,68,120)
     expected=z['original_q'].copy()
     for b in j['plan']:
      if b['frame']==int(frame):
       ix=(slice(None),slice(None),slice(b['y'],b['y']+b['h']),slice(b['x'],b['x']+b['w']))
       expected[ix]=np.maximum(expected[ix]*b['factor'],.5)
     assert np.array_equal(expected,z['effective_q'])
   with (ROOT/'metrics/per_frame'/(j['id']+'_'+str(r['attempt'])+'.csv')).open() as h:frames=list(csv.DictReader(h))
   assert len(frames)==16 and sum(int(f['real_bits']) for f in frames)==len(data)*8
   for m in ('LPIPS','DISTS'):assert math.isclose(sum(float(f[m]) for f in frames)/16,x[m],rel_tol=1e-12)
   with (ROOT/'metrics/per_transition'/(j['id']+'_'+str(r['attempt'])+'.csv')).open() as h:transitions=list(csv.DictReader(h))
   assert len(transitions)==15 and math.isclose(sum(float(f['FloLPIPS']) for f in transitions)/15,x['FloLPIPS'],rel_tol=1e-12)
   if j['kind']=='repeat':assert all(r['reproducibility'].values())
   if j.get('standalone'):assert r['standalone']['reconstruction_sha256']==x['reconstruction_sha256']
   records.append(dict(id=j['id'],status='PASS',bytes=len(data),controls=n))
  except Exception:errors.append(dict(id=p.stem,traceback=traceback.format_exc()))
 changed=[]
 for p,v in load(ROOT/'manifests/weights.json').items():
  if sha(p)!=v['sha256']:changed.append(p)
 for d in load(ROOT/'manifests/reused_dependency_sources.json')['records']:
  if sha(d['original_path'])!=d['sha256']:changed.append(d['original_path'])
 for p,v in load(ROOT/'manifests/metric_weights.json').items():
  if sha(p)!=v['sha256']:changed.append(p)
 report=dict(status='PASS' if not errors and not changed else 'FAILED',validated_new_receipts=len(records),records=records,errors=errors,changed_frozen_sources_or_weights=changed)
 dump('manifests/artifact_validation.json',report)
 return report

def observation(ref,phase,d,q,strategy,seed=None,target=None):
 r=read_receipt(ref);j=r['job'];x=r['result'];a=r['rate'];native=strategy=='A_native_baseline'
 bits=a['native_codec_stream_bits'] if native else a['actual_total_bits']
 target=bits if target is None else target
 return dict(phase=phase,dataset=d,video=load(ROOT/'manifests/videos.json')[d][phase]['name'],base_QP=q,strategy=strategy,seed=seed,status='VALID_RATE_MATCH' if matched(bits,target) else 'INVALID_ACTUAL_RATE',encoding_QP=j['qp'],target_bits=target,actual_bits=bits,actual_bytes=bits//8,bpp=bits/(16*1920*1080),kbps=bits*30/(16*1000),delta_actual_bits=bits-target,relative_rate_deviation=(bits-target)/target,control_header_bits=0 if native else a['control_header_bits'],native_headers_bits=a['native_SPS_IP_header_bits'],RANS_payload_bits=a['RANS_payload_bits'],plan=json.dumps([] if native else j['plan'],sort_keys=True),LPIPS=x['LPIPS'],DISTS=x['DISTS'],FloLPIPS=x['FloLPIPS'],FID=None,FID_status=cfg()['FID'],encode_ms=r['timings']['encode_ms'],decode_ms=r['timings']['decode_ms'],wall_seconds=r['wall_seconds'],actual_rate_matched=matched(bits,target),exact_rate_match=bits==target,real_encoder=True,real_decoder=True,calibration_only=phase=='calibration',oracle_from_independent_calibration=phase=='test' and strategy=='D_oracle',rate_selection_uses_test_quality=False,receipt_path=ref['receipt_path'],stream_path=str(Path(x['bitstream_path']).with_name('native.bin')) if native else x['bitstream_path'],source=ref['source'])

def rate_only_choice(trials,target,base_q):
 # Deliberately accepts rates and QPs only: independent quality cannot influence this choice.
 pool=[t for t in trials if matched(t['bits'],target)] or trials
 return min(pool,key=lambda t:rate_key(t['bits'],target,t['qp'],base_q)) if pool else None

def main():
 validation=validate_new();mapping=refs();frozen_path=ROOT/'oracle/rate_allocation/frozen_policy.json'
 frozen=load(frozen_path) if frozen_path.exists() else dict(policies=[])
 rows=[];all_trials=[];failures=[]
 for phase in ('calibration','test'):
  for d in load(ROOT/'manifests/videos.json'):
   for q in cfg()['QPs']:
    ref=lookup(phase,d,q,[])
    if ref:rows.append(observation(ref,phase,d,q,'A_native_baseline'))
 for p in frozen['policies']:
  d,q=p['dataset'],p['base_qp'];bref=lookup('test',d,q,[])
  if p['status']!='FROZEN':
   for strategy,seed in [('D_oracle',None),('C_temporal_or_content_heuristic',None)]+[('B_random',s) for s in cfg()['random_seeds']]:
    rows.append(dict(phase='test',dataset=d,base_QP=q,strategy=strategy,seed=seed,status='no_feasible_candidate',reason=p['reason'],actual_rate_matched=False,exact_rate_match=False))
   continue
  c=p['calibration_choice'];cref=dict(receipt_path=c['receipt_path'],source='verified_calibration_pool')
  rows.append(observation(cref,'calibration',d,q,p['strategy'],p['seed'],p['target_bits']))
  if not bref:
   rows.append(dict(phase='test',dataset=d,base_QP=q,strategy=p['strategy'],seed=p['seed'],status='MISSING_NATIVE_BASELINE',actual_rate_matched=False,exact_rate_match=False));continue
  target=read_receipt(bref)['rate']['native_codec_stream_bits'];trials=[]
  for qq in p['test_global_qps']:
   ref=lookup('test',d,qq,p['frozen_plan'])
   if not ref:continue
   r=read_receipt(ref);trials.append(dict(bits=r['rate']['actual_total_bits'],qp=qq,ref=ref))
   item=observation(ref,'test',d,q,p['strategy'],p['seed'],target);item['trial']=True;all_trials.append(item)
  choice=rate_only_choice(trials,target,q)
  if choice:
   row=observation(choice['ref'],'test',d,q,p['strategy'],p['seed'],target)
   row.update(selected_by='actual byte gap only; then closest global QP',frozen_policy_sha256=sha(frozen_path))
   original=read_receipt(bref)['result']
   for m in ('LPIPS','DISTS','FloLPIPS'):row['delta_'+m.lower()]=row[m]-original[m]
   if not row['actual_rate_matched']:row['reason']='Frozen plan missed the fixed actual-rate band at all three legal QPs'
   rows.append(row)
  else:rows.append(dict(phase='test',dataset=d,base_QP=q,strategy=p['strategy'],seed=p['seed'],status='NO_SUCCESSFUL_ACTUAL_ENCODING',actual_rate_matched=False,exact_rate_match=False))
 write('oracle/rate_allocation/strategy_results.csv',rows)
 write('metrics/per_video_results.csv',rows)
 write('oracle/rate_allocation/all_independent_trials.csv',all_trials)
 write('baseline/baseline_summary.csv',[r for r in rows if r['strategy']=='A_native_baseline'])
 write('oracle/rate_allocation/random_all_seeds.csv',[r for r in rows if r['strategy']=='B_random'])
 predictions=ROOT/'oracle/rate_allocation/candidate_predictions.json'
 if predictions.exists():
  resolved=[]
  for r in load(predictions):
   ref=mapping.get(r['candidate_id']);row=dict(r)
   if ref:
    x=read_receipt(ref);row.update(actual_bits=x['rate']['actual_total_bits'],actual_delta_bits=x['rate']['actual_total_bits']-r['target_bits'],actual_rate_matched=matched(x['rate']['actual_total_bits'],r['target_bits']),actual_delta_lpips=x['result']['LPIPS']-baseline('calibration',r['dataset'],r['base_qp'])['result']['LPIPS'],LPIPS=x['result']['LPIPS'],DISTS=x['result']['DISTS'],FloLPIPS=x['result']['FloLPIPS'],receipt_path=ref['receipt_path'],failure_reason=None if matched(x['rate']['actual_total_bits'],r['target_bits']) else 'actual_rate_outside_predeclared_band',execution='reused' if ref['source'].startswith('v1') else 'new')
   else:
    p=ROOT/'manifests/receipts'/(r['candidate_id']+'.json')
    row.update(execution='failed' if p.exists() else 'not_scheduled_bounded_search',failure_reason=load(p).get('traceback',load(p).get('reason')) if p.exists() else r['selection_rejection'] or 'outside_fixed_validation_subset')
   resolved.append(row)
  dump('oracle/rate_allocation/candidate_outcomes.json',resolved)
  write('oracle/rate_allocation/candidate_outcomes.csv',[dict(r,plan=json.dumps(r['plan']),levels=json.dumps(r['levels'])) for r in resolved])
 tests=[r for r in rows if r['phase']=='test' and r['strategy']!='A_native_baseline']
 matched_tests=[r for r in tests if r['actual_rate_matched']]
 exact=[r for r in matched_tests if r['exact_rate_match']]
 pairs=[];dominance=[]
 for r in matched_tests:
  pairs.append(dict(r,comparison='paired with native baseline at same dataset/video/base QP',comparison_type='exact' if r['exact_rate_match'] else 'approximate_under_budget'))
  if r['strategy']=='D_oracle':
   rivals=[x for x in matched_tests if x['dataset']==r['dataset'] and x['base_QP']==r['base_QP'] and x['strategy']!='D_oracle']
   dominance.append(dict(dataset=r['dataset'],base_QP=r['base_QP'],oracle_no_more_bits_than_every_matched_comparator=all(r['actual_bits']<=x['actual_bits'] for x in rivals),matched_comparator_count=len(rivals),all_three_metrics_improve_vs_native=all(r['delta_'+m.lower()]<=-cfg()['stability_gate'][m] for m in ('LPIPS','DISTS','FloLPIPS'))))
 # Always create these tables, even when no comparison is eligible.
 empty_header='phase,dataset,base_QP,strategy,seed,status,actual_bits,LPIPS,DISTS,FloLPIPS\n'
 for name,table in [('matched_pair_results.csv',pairs),('exact_pair_results.csv',exact)]:
  if table:write('oracle/rate_allocation/'+name,table)
  else:(ROOT/'oracle/rate_allocation'/name).write_text(empty_header)
 oracle=[r for r in matched_tests if r['strategy']=='D_oracle']
 stable_subset=len({r['dataset'] for r in oracle})>=cfg()['stability_gate']['minimum_independent_datasets'] and all(all(r['delta_'+m.lower()]<=-cfg()['stability_gate'][m] for m in ('LPIPS','DISTS','FloLPIPS')) for r in oracle)
 stable_all=stable_subset and len(oracle)==8
 complete_sets=[]
 for d in load(ROOT/'manifests/videos.json'):
  for q in cfg()['QPs']:
   relevant=[r for r in matched_tests if r['dataset']==d and r['base_QP']==q]
   if len(relevant)==5:complete_sets.append(point(d,q))
 rate_status=dict(exact_experimental_test_pairs=len(exact),approximate_under_budget_test_pairs=len(matched_tests)-len(exact),complete_ABCD_with_three_random_seeds_points=complete_sets,all_eight_points_fairly_compared=len(complete_sets)==8,tolerance_fraction=.005,over_budget_results_excluded=True,empty_plans_excluded=True,eligible_comparison_file='oracle/rate_allocation/matched_pair_results.csv',exact_comparison_file='oracle/rate_allocation/exact_pair_results.csv',all_results_file='oracle/rate_allocation/strategy_results.csv',unmatched_and_skipped=[{k:r.get(k) for k in ('dataset','base_QP','strategy','seed','status','reason','actual_bits','target_bits')} for r in tests if not r['actual_rate_matched']])
 dump('oracle/rate_allocation/rate_matching_status.json',rate_status)
 dump('metrics/independent_stability_status.json',dict(protocol_thresholds=cfg()['stability_gate'],independent_matched_oracle_points=len(oracle),independent_matched_datasets=sorted({r['dataset'] for r in oracle}),stable_on_all_eight_preregistered_points=stable_all,stable_on_matched_subset=stable_subset,no_population_claim=True,per_point_quality_and_comparator_rate_guards=dominance,scope='Descriptive fixed-protocol eligibility and metric thresholds only; missing or unmatched points are not counted as improvements'))
 jobs=[load(p) for p in (ROOT/'manifests/jobs').glob('*.json')]
 receipts={p.stem:load(p) for p in (ROOT/'manifests/receipts').glob('*.json')}
 for j in jobs:
  if receipts.get(j['id'],{}).get('status')!='PASS':failures.append(dict(job=j,receipt=receipts.get(j['id'],dict(status='MISSING'))))
 dump('logs/failed_tasks.json',dict(final_failures=failures,failed_attempts=[load(p) for p in (ROOT/'manifests/attempts').glob('*.json') if load(p)['status']!='PASS']))
 all_exits=[]
 for p in (ROOT/'logs/stage_exits').glob('*.json'):all_exits.extend(load(p))
 exit_map={x['pid']:x['exit_code'] for x in all_exits}
 final_workers=[dict(id=r['job']['id'],pid=r.get('pid'),gpu=r.get('gpu'),status=r['status'],exit_code=exit_map.get(r.get('pid'))) for r in receipts.values()]
 driver_exit=load(ROOT/'logs/driver_exit.json')
 dump('logs/GPU_task_exit_status.json',dict(attempt_exits=all_exits,final_workers=final_workers,all_final_tasks_successful=not failures,all_final_worker_exit_codes_zero=all(x['exit_code']==0 for x in final_workers),driver_exit=driver_exit,GPU_tasks_running=0))
 sensitivity=list(csv.DictReader((ROOT/'oracle/local_sensitivity/block_intervention_results.csv').open()))
 for r in receipts.values():
  if r['status']=='PASS' and r['job']['kind']=='repeat':
   j=r['job'];original=load(j['repeat_source'])
   sensitivity.append(dict(id=j['id'],dataset=j['dataset'],QP=j['qp'],plan=json.dumps(j['plan']),source='v2_exact_repeat',all_metrics_and_streams_exact=all(r['reproducibility'].values()),source_receipt=j['repeat_source'],actual_bits=r['rate']['actual_total_bits'],LPIPS=r['result']['LPIPS'],DISTS=r['result']['DISTS'],FloLPIPS=r['result']['FloLPIPS'],real_encoder=True,real_decoder=True))
 write('oracle/local_sensitivity/block_intervention_results.csv',sensitivity)
 dump('manifests/binary_artifacts_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'artifacts').rglob('*')) if p.is_file()])
 passed=not failures and validation['status']=='PASS' and driver_exit['exit_code']==0 and all(x['exit_code']==0 for x in final_workers)
 manifest=load(ROOT/'manifests/experiment_manifest.json');manifest.update(status='PASS' if passed else 'PARTIAL',new_GPU_jobs=len(jobs),new_successful_jobs=sum(r['status']=='PASS' for r in receipts.values()),final_failed_or_missing_jobs=len(failures),rate_matching='oracle/rate_allocation/rate_matching_status.json',independent_stability='metrics/independent_stability_status.json',real_local_rate_control=True,exact_experimental_test_pairs=len(exact),approximate_under_budget_test_pairs=len(matched_tests)-len(exact),stable_on_all_preregistered_points=stable_all)
 dump('manifests/experiment_manifest.json',manifest);dump('experiment_manifest.json',manifest)
 state=dict(status=manifest['status'],stage='ALL_BOUNDED_TASKS_FINISHED',GPU_tasks_running=0,completed_GPU_jobs=manifest['new_successful_jobs'],failed_GPU_jobs=len(failures),stopping=load(ROOT/'manifests/stopping.json') if (ROOT/'manifests/stopping.json').exists() else None,finished=time.time(),publication='see publication_status.json; final push performed after this report')
 dump('manifests/run_status.json',state);dump('run_status.json',state)
 dump('manifests/implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')})
 with (ROOT/'README.md').open('a') as f:f.write('\nFinal bounded-run status: '+manifest['status']+'. New successful GPU jobs: '+str(manifest['new_successful_jobs'])+'. Exact independent experimental pairs: '+str(len(exact))+'. Approximate under-budget pairs: '+str(len(matched_tests)-len(exact))+'. Eligibility, skips and all raw metrics are saved in oracle/rate_allocation/strategy_results.csv and rate_matching_status.json. Driver and worker exit codes are in logs/GPU_task_exit_status.json.\n')
 print('FINALIZED',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
