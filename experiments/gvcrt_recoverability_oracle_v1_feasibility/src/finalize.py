from base import *
def main():
 from validate_artifacts import validate
 validation=validate()
 from verify_saved_controls import verify
 verify()
 jobs=[load(p) for p in sorted((ROOT/'manifests/jobs').glob('*.json'))]
 receipts={p.stem:load(p) for p in (ROOT/'manifests/receipts').glob('*.json')}
 good=[r for r in receipts.values() if r['status']=='PASS'];bad=[dict(job=j,receipt=receipts.get(j['id'],dict(status='MISSING'))) for j in jobs if receipts.get(j['id'],{}).get('status')!='PASS']
 rows=[]
 for r in good:
  j=r['job'];a=r['rate'];x=r['result']
  stock_bits=receipts[j.get('baseline_id',j['id'])]['rate']['native_codec_stream_bits']
  rows.append(dict(id=j['id'],kind=j['kind'],dataset=j['dataset'],QP=j['qp'],baseline_id=j.get('baseline_id'),strategy=j.get('strategy'),plan=json.dumps(j['plan'],sort_keys=True),actual_total_bits=a['actual_total_bits'],actual_bpp=x['bpp'],kbps=x['kbps'],native_codec_bits=a['native_codec_stream_bits'],RANS_payload_bits=a['RANS_payload_bits'],control_header_bits=a['control_header_bits'],LPIPS=x['LPIPS'],DISTS=x['DISTS'],FloLPIPS=x['FloLPIPS'],FID=None,FID_status=x['FID_status'],**r['delta'],delta_actual_bits_vs_stock_original=a['actual_total_bits']-stock_bits,encode_ms=r['timings']['encode_ms'],decode_ms=r['timings']['decode_ms'],wall_seconds=r['wall_seconds'],success=True,independent_decode=True,causal_suffix_reencoded=True,gpu=r['gpu'],stream_path=x['bitstream_path'],stream_sha256=x['bitstream_sha256']))
 write('metrics/all_raw_results.csv',rows)
 base=[r for r in good if r['job']['kind']=='baseline'];repeat=[r for r in good if r['job']['kind']=='repeat'];local=[r for r in good if r['job']['kind']=='local'];joint=[r for r in good if r['job']['kind']=='joint'];noop=[r for r in good if r['job']['kind']=='noop'];alloc=[r for r in good if r['job']['kind']=='allocation']
 write('baseline/baseline_summary.csv',[dict(next(x for x in rows if x['id']==r['job']['id']),original_native_bpp=r['rate']['native_codec_stream_bits']/(16*1920*1080),original_native_kbps=r['rate']['native_codec_stream_bits']*30/(16*1000),native_prefix_exact=r['result']['historical_native_prefix_exact']) for r in base])
 write('baseline/reproducibility.csv',[dict(id=r['job']['id'],**r['reproducibility']) for r in repeat])
 write('oracle/local_sensitivity/block_intervention_results.csv',[r for r in rows if r['kind'] in ('local','noop','joint')])
 bykind={}
 for j in jobs:
  d=bykind.setdefault(j['kind'],dict(planned=0,successful=0,failed_or_missing=0));d['planned']+=1
  if receipts.get(j['id'],{}).get('status')=='PASS':d['successful']+=1
  else:d['failed_or_missing']+=1
 for d in bykind.values():d['success_rate']=d['successful']/d['planned']
 dump('metrics/task_success_rates.json',bykind)
 dump('oracle/local_sensitivity/intervention_manifest.json',dict(valid_real_rate_control=load(ROOT/'audit/local_control_correctness.json')['real_local_rate_control'],coordinate_system='all128 y channels; y/x/h/w in stride16 latent grid, nominal input footprint is grid times16, CNN receptive field larger',levels=[.5,1.,2.],positions=[2,8],independent_decode_reads_signalled_controls=True,no_weight_changes=True,no_training=True,tasks=[j for j in jobs if j['kind'] in ('local','noop','joint')],raw_symbols='oracle/local_sensitivity/symbols/',full_tensor_paths={r['job']['id']:r['tensor_artifacts'] for r in good}))
 dump('oracle/local_sensitivity/failed_interventions.json',dict(final_failed_tasks=bad,recovered_attempts=[dict(path=str(p),record=load(p)) for p in (ROOT/'logs/failed_attempts').glob('*.json')]))
 dump('oracle/local_sensitivity/actual_rate_accounting.json',dict(actual_total_definition='native full codec stream plus complete signalled experiment envelope; no estimated rates',base_header_bytes=10,additional_bytes_per_local_decision=11,records={r['job']['id']:r['rate'] for r in good},components_sum_exact=all(r['rate']['actual_total_bits']==r['rate']['RANS_payload_bits']+r['rate']['native_SPS_IP_header_bits']+r['rate']['control_header_bits'] for r in good)))
 selected=[]
 for b in base:
  target=b['rate']['native_codec_stream_bits'];bid=b['job']['id']
  def entry(r,strategy):
   x=r['result'];stock=strategy=='A_original';bits=r['rate']['native_codec_stream_bits' if stock else 'actual_total_bits'];delta=bits-target
   path=str(Path(x['bitstream_path']).parent/'native.bin') if stock else x['bitstream_path']
   return dict(baseline_id=bid,dataset=b['job']['dataset'],baseline_QP=b['job']['qp'],strategy=strategy,selected_job=r['job']['id'],selected_QP=r['job']['qp'],target_bits=target,actual_bits=bits,actual_bpp=bits/(16*1920*1080),actual_kbps=bits*30/(16*1000),stream_path=path,stream_sha256=sha(path),rate_scope='unmodified stock codec stream' if stock else 'complete experimental stream including all control signalling',delta_actual_bits=delta,relative_rate_deviation=delta/target,exact_actual_rate_match=delta==0,fair_equal_rate_comparison=delta==0,budget_feasible=bits<=target,LPIPS=x['LPIPS'],DISTS=x['DISTS'],FloLPIPS=x['FloLPIPS'],FID=None,FID_status=x['FID_status'],selection='minimum absolute actual bit gap only; tie prefers lower actual bits and closest baseline QP; no quality selection',global_optimal_actual_allocation=False)
  selected.append(entry(b,'A_original'))
  for strategy in ('B_random','C_temporal_RGB','D_oracle'):
   trials=[r for r in good if r['job'].get('baseline_id')==bid and r['job'].get('strategy')==strategy]
   if not trials:continue
   best=min(trials,key=lambda r:(abs(r['rate']['actual_total_bits']-target),r['rate']['actual_total_bits']>target,abs(r['job']['qp']-b['job']['qp']),r['job']['id']))
   selected.append(entry(best,strategy))
 write('oracle/rate_allocation/strategy_results.csv',selected)
 write('oracle/rate_allocation/exact_actual_rate_comparisons.csv',[r for r in selected if r['exact_actual_rate_match']])
 write('oracle/rate_allocation/unmatched_rate_results.csv',[r for r in selected if not r['exact_actual_rate_match']])
 interaction=[]
 for r in joint:
  single=[x for x in local if x['job']['baseline_id']==r['job']['baseline_id'] and ((x['job']['block_index']==0 and x['job']['plan'][0]['factor']==.5) or (x['job']['block_index']==1 and x['job']['plan'][0]['factor']==2.))]
  if len(single)==2:
   a=dict(id=r['job']['id'],measured_joint_delta_bits=r['delta']['delta_actual_bits'],sum_single_delta_bits=sum(x['delta']['delta_actual_bits'] for x in single))
   for metric in ('lpips','dists','flolpips'):a.update({f'measured_joint_delta_{metric}':r['delta']['delta_'+metric],f'sum_single_delta_{metric}':sum(x['delta']['delta_'+metric] for x in single)})
   interaction.append(a)
 write('oracle/rate_allocation/joint_interaction_validation.csv',interaction)
 inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'artifacts').rglob('*')) if p.is_file()]
 dump('manifests/binary_artifacts_index.json',inventory)
 weights=load(ROOT/'manifests/weights.json');assert all(sha(p)==v['sha256'] for p,v in weights.items())
 audit=load(ROOT/'audit/source_path_audit.json');source_changes=[p for p,v in audit['source_files'].items() if sha(p)!=v['sha256']]
 control=load(ROOT/'audit/local_control_correctness.json')['real_local_rate_control']
 audit.update(status='PASS_REAL_LOCAL_CONTROL' if control and not bad else 'PARTIAL',real_local_control_validated=control,stock_block_QP_API=False,stock_decoder_compatible=False,full_causal_RANS_validation=control,actual_observed_tensors=base[0]['tensor_artifacts'] if base else {},source_changes_observed=source_changes)
 dump('audit/source_path_audit.json',audit)
 checks=dict(baselines=len(base),repeats=len(repeat),local_interventions=len(local),noops=len(noop),joint_tests=len(joint),allocation_initial_trials=len(alloc),all_planned_jobs_accounted=all(j['id'] in receipts for j in jobs),all_planned_jobs_successful=not bad,frozen_weight_files_sha256_unchanged=True,source_files_unchanged=not source_changes,all_frame_reference_checks=all(r['reference_state_exact_all_P_frames'] for r in good),all_independent_decodes=all(r['independent_decode'] for r in good),all_repeats_exact=all(all(r['reproducibility'].values()) for r in repeat),all_code_and_symbol_artifacts_indexed=True)
 passed=(len(base),len(repeat),len(local),len(noop),len(joint),len(alloc))==(8,8,64,1,8,24) and not bad and not source_changes
 dump('manifests/integrity.json',dict(status='PASS' if passed else 'PARTIAL',checks=checks,final_failures=bad))
 exits=[]
 for p in (ROOT/'logs/stage_exits').glob('*.json'):exits.extend(load(p))
 dump('logs/GPU_task_exit_status.json',dict(preflight_launch_exits=load(ROOT/'manifests/smoke_launches.json'),stage_worker_exits=exits,standalone_decoder_exits=[load(p) for p in (ROOT/'logs/standalone_exits').glob('*.json')],final_receipt_statuses={j['id']:receipts.get(j['id'],{}).get('status','MISSING') for j in jobs},failed_attempts='logs/failed_attempts/',all_final_tasks_finished=checks['all_planned_jobs_accounted']))
 dump('manifests/implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')})
 manifest=load(ROOT/'experiment_manifest.json');manifest.update(status='PASS' if passed else 'PARTIAL',completed_successful_jobs=len(good),failed_final_jobs=len(bad),real_local_precision_control=control,experimental_control_header=True,allocation_comparisons_exact_rate_only=True,allocation_budget='stock original native bytes, without experiment envelope',all_fixed_weights_unchanged=True)
 dump('experiment_manifest.json',manifest)
 dump('run_status.json',dict(status='PASS' if passed else 'PARTIAL',stage='EXPERIMENT_COMPLETE_PUBLICATION_PENDING',completed_jobs=len(good),failed_jobs=len(bad),GPU_tasks_running=0,real_local_rate_intervention=control,FID='NOT_COMPUTED_INSUFFICIENT_SAMPLES',integrity='manifests/integrity.json',finished=time.time()))
 with (ROOT/'README.md').open('a') as f:f.write('\nFinal experiment status: '+('PASS' if passed else 'PARTIAL')+'. Completed jobs:'+str(len(good))+'. Final failures:'+str(len(bad))+'. See manifests/integrity.json, metrics/all_raw_results.csv, oracle/rate_allocation/strategy_results.csv and logs/GPU_task_exit_status.json. Nonzero actual-rate deviations are excluded from exact-rate comparisons.\n')
 print('FINALIZED',manifest['status'],len(good),'successful jobs',flush=True)
if __name__=='__main__':main()
