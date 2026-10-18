"""Read-only v1 audit; lock protocol before any new GPU job or test metric."""
from protocol import *
import itertools,shutil,collections

def prepare():
 assert not (ROOT/'configs/experiment.json').exists(),'Protocol already registered; resume driver instead'
 vcfg=load(V1/'configs/experiment.json')
 config=dict(name=ROOT.name,frames=16,width=1920,height=1080,fps=30,QPs=[2,7],blocks=vcfg['blocks'],levels=[1.,.5,2.],q_floor=.5,force_zero_thres=.12,seed=20261018,random_seeds=[20261018,20261019,20261020],rate_tolerance_fraction=.005,rate_band='[0.995*target,target], all control bytes charged; no over-budget comparison',exact_priority=True,maximum_base_q_new_plans_per_point=3,global_q_offsets=[-1,0,1],maximum_new_neighbor_q_plans_per_point=2,maximum_new_GPU_jobs=100,minimum_free_MiB=7000,resource_wait_limit_seconds=1800,GPU_ids=[4,5,6,7],max_workers_per_GPU=1,FID='NOT_COMPUTED_INSUFFICIENT_SAMPLES',training=False,weights_modified=False,objective='calibration sequence LPIPS only',stability_gate=dict(minimum_independent_datasets=2,LPIPS=1e-4,DISTS=1e-4,FloLPIPS=1e-4),approximate_comparison_claim='under-budget paired measurements only; Oracle superiority over any other method additionally requires no more actual bits than that method',selection='bounded joint search plus measured single candidates; no empty fallback; identical actual feasibility pool for B/C/D',stops=['no validated calibration candidate at a point: skip its independent strategies','no independent matched candidate after fixed QP trials: report invalid, do not relax tolerance','after fixed independent tests: stop regardless of quality; no further search','100 new GPU jobs or 1800 seconds no available authorized GPU: explicit failure'])
 dump('configs/experiment.json',config)
 videos={};ev=load(V32/'eval_manifest.json')
 for d in ('uvg_holdout','hevc_b','mcljcv','ulong'):
  assert len(ev[d])>=2
  videos[d]=dict(calibration=ev[d][0],test=ev[d][1])
  assert videos[d]['calibration']['input_dir']!=videos[d]['test']['input_dir']
  for v in videos[d].values():assert Path(v['input_dir']).is_dir()
 dump('manifests/videos.json',videos)
 for f in ('weights.json','metric_weights.json','environment.json'):
  shutil.copy2(V1/'manifests'/f,ROOT/'manifests'/f)
 for f in ('source_path_audit.json','metric_protocol_audit.json','local_control_correctness.json'):
  shutil.copy2(V1/'audit'/f,ROOT/'audit'/('v1_'+f))
 for p,v in load(ROOT/'manifests/weights.json').items():assert sha(p)==v['sha256']
 dep=load(V1/'manifests/dependency_sources.json')
 assert all(sha(x['original_path'])==x['sha256'] for x in dep['records'])
 dump('manifests/reused_dependency_sources.json',dict(v1_manifest=str(V1/'manifests/dependency_sources.json'),sha256=sha(V1/'manifests/dependency_sources.json'),records=dep['records']))
 assert load(V1/'manifests/artifact_validation.json')['validated_receipts']==161
 mapping={};records=[];sensitivity=[];floor=[]
 for p in sorted((V1/'manifests/receipts').glob('*.json')):
  r=load(p);assert usable(r)
  j=r['job'];x=r['result'];a=r['rate']
  assert sha(x['bitstream_path'])==x['bitstream_sha256']
  assert Path(x['bitstream_path']).stat().st_size*8==a['actual_total_bits']
  for info in r['tensor_artifacts'].values():assert sha(info['path'])==info['sha256']
  if j['kind']=='repeat':continue
  key=signature('calibration',j['dataset'],j['qp'],j['plan'])
  mapping.setdefault(key,dict(receipt_path=str(p),source='v1_checksum_verified'))
  records.append(dict(id=j['id'],receipt_path=str(p),receipt_sha256=sha(p),phase='calibration',dataset=j['dataset'],qp=j['qp'],kind=j['kind'],plan=j['plan'],native_bits=a['native_codec_stream_bits'],experimental_bits=a['actual_total_bits'],decoded=True))
  if j['kind']=='local':
   sy=load(V1/'oracle/local_sensitivity/symbols'/(j['id']+'.json'))[0]
   b=load(V1/'manifests/receipts'/(j['baseline_id']+'.json'))
   item=dict(id=j['id'],dataset=j['dataset'],QP=j['qp'],block_index=j['block_index'],plan=json.dumps(j['plan']),q_changed_fraction=sy['effective_q_changed_fraction'],ROI_symbol_changed_fraction=sy['symbol_changed_fraction'],delta_actual_bits_vs_stock=a['actual_total_bits']-b['rate']['native_codec_stream_bits'],**r['delta'],source_receipt=str(p),source='v1_reused',real_encoder=True,real_decoder=True)
   sensitivity.append(item);floor.append(item)
 dump('manifests/encoding_refs.json',mapping)
 write('oracle/local_sensitivity/block_intervention_results.csv',sensitivity)
 # Actual first-16-frame native rates at adjacent QPs, from verified historical streams.
 # They are rate priors only; no 64-frame quality is presented as 16-frame quality.
 native_rates={}
 for d,v in videos.items():
  vv=v['calibration'];native_rates[d]={}
  for q in (1,2,3,6,7,8):
   p=V33/'evaluation/points'/f'{d}_{vv["video_index"]}_A_noP_frozenReceiver_{q}.json'
   if not p.exists():continue
   r=load(p);assert r['status']=='PASS';x=r['result']
   fp=Path(x['frame_metrics_path']);assert sha(fp)==x['frame_metrics_sha256']
   with fp.open() as f:rows=list(csv.DictReader(f))[:16]
   assert len(rows)==16
   bits=sum(int(float(row['real_bits'])) for row in rows)
   native_rates[d][str(q)]=dict(bits=bits,frame_metrics_path=str(fp),frame_metrics_sha256=sha(fp),historical_receipt=str(p),historical_receipt_sha256=sha(p),scope='first 16 causal frames, rate prior only; no transferred quality')
  for q in (2,7):assert native_rates[d][str(q)]['bits']==baseline('calibration',d,q)['rate']['native_codec_stream_bits']
 dump('manifests/native_rate_priors.json',native_rates)
 selected=load(V1/'oracle/rate_allocation/selection_manifest.json')
 v1rates=load(V1/'oracle/rate_allocation/rate_matching_status.json')
 reuse=dict(status='PASS',v1_directory=str(V1),v1_manifest_sha256=sha(V1/'experiment_manifest.json'),v1_integrity=load(V1/'manifests/integrity.json'),records=records,valid_single_interventions=len(sensitivity),empty_strategy_count=sum(not x['plan'] for x in selected),empty_strategies=[dict(baseline_id=x['baseline_id'],strategy=x['strategy'],fallback=x.get('no_surrogate_budget_feasible_state',False)) for x in selected if not x['plan']],v1_rate_matching=v1rates,v1_unmatched_results=str(V1/'oracle/rate_allocation/unmatched_rate_results.csv'),test_videos='second fixed inherited identity per dataset, never consumed for calibration selection',q_floor_observations=floor,invalid_v1_results_not_claimed_as_fair=True)
 dump('audit/v1_result_reuse.json',reuse)
 dump('audit/rate_accounting_audit.json',dict(native_block_QP=False,local_control='unchanged v1 dual-end prior-step override',native_global_control='integer frame QP with inherited P shifts; no continuous external rate knob',fixed_control_bits=80,per_decision_bits=88,fixed_cost_constraint='v1 cost+80 is physically correct, not removed',delta_formula='native_target_bits + 80 + sum(single_delta_vs_empty_envelope); each single delta already contains its own 88 decision bits',joint_costs_not_assumed_additive=True,q_floor=.5,legal_levels=[.5,1.,2.],no_bitstream_edits=True,no_padding=True,tolerance=config['rate_band'],tolerance_fraction=.005,tolerance_basis='predeclared tight 0.5% maximum shortfall in actual bytes; no over-budget quality advantage is admitted, discrete RANS bytes and integer QPs prevent arbitrary exact tuning',exact_results_separate=True,transfer_to_neighbor_q='additive rate surrogate only; each actual candidate re-encoded and must pass actual budget',all_fields_charged=True))
 manifest=dict(name=ROOT.name,status='PREPARED',config='configs/experiment.json',config_sha256=sha(ROOT/'configs/experiment.json'),videos='manifests/videos.json',videos_sha256=sha(ROOT/'manifests/videos.json'),calibration='v1 first16 frames of first inherited video in each dataset',test='first16 frames of second inherited video in each dataset',QP=[2,7],frozen_before_new_GPU_jobs=True,independent_test=True,training_steps=0,v1_read_only=True,maximum_new_GPU_jobs=100,FID=config['FID'])
 dump('manifests/experiment_manifest.json',manifest);dump('experiment_manifest.json',manifest)
 dump('manifests/run_status.json',dict(status='PREPARED',stage='AUDIT_COMPLETE',time=time.time()));dump('run_status.json',load(ROOT/'manifests/run_status.json'))
 (ROOT/'audit/v2_design.md').write_text('''# Fixed validation protocol

The old additive constraint charged the right 80 fixed bits; its empty-plan fallback was invalid. v2 never selects an empty B/C/D plan. The codec and stream envelope are copied byte-for-byte from v1. Official source, P module parameters, Bridge and all weights remain frozen. No training is run.

All four first-videos at QP2/7 are calibration points. Independent tests use the second fixed inherited video identity in each dataset, selected before new GPU jobs. All clips have 16 native-resolution frames. v1 baseline, local and joint results are reused only with stream and tensor checksum verification. Adjacent-QP native rate priors use the first16 frame byte counts, never 64-frame quality.

Enumerate all 80 nonempty assignments of four blocks and levels 1/0.5/2. Reject actual no-ops, retain weak changes with measured fractions, and deduplicate complete encoding identities. For each point, validate at most three previously unmeasured base-QP plans: the best two predicted LPIPS plans under the band (or closest predicted rate if none), plus the lowest predicted-rate plan. Validate one closest-predicted-rate plan at each adjacent legal global QP. These are fixed bounded checks of interactions and global control, not an exhaustive actual-codec upper bound. Every single-candidate prediction and rejection is saved. Already measured actual candidates at the same allowed QPs join the same pool. The feasible pool is determined by actual complete bytes, not by the additive surrogate.

The predeclared approximate band is [99.5%,100%] of the original native-stream target. All 80 fixed bits and 88 bits per local decision remain charged. Exact matches take priority when available. No padding, post-encoding edits, over-budget result or tolerance relaxation is allowed. Approximate and exact comparisons are separate. For a directional superiority statement relative to any comparator, the Oracle must also use no more actual bits than that comparator; otherwise the rate confound is unresolved.

D chooses the lowest actual calibration LPIPS from the bounded verified pool. C uses fixed temporal RGB-difference importance in the existing input and a signed precision score. B samples the identical pool with each of three saved seeds; all seeds are retained, including duplicate choices. Each strategy freezes one plan on calibration. If there is no actual feasible nonempty plan, record no_feasible_candidate and skip that point's independent strategies. No empty fallback or random-seed cherry-picking is permitted.

On the independent video, encode the frozen plan at the base QP and both predeclared adjacent QPs. Select using only actual byte gap (then distance from base QP), never test quality. Save all trials and all failures. A changed baseline QP is never substituted for the target baseline. Baseline and every changed stream run the full causal suffix and fresh independent decode. Representative local controls are repeated; correctness includes a separate decoder process with only the persisted stream and fixed weights before expanding the schedule.

Stop after this bounded plan. Points without feasible calibration candidates or actual test rate matches are explicit skips. Do not expand after unfavorable test quality. Stop on a 100-job cap, a resource wait timeout, or correctness failure. A descriptive stability gate requires matched D/A tests on at least two independent datasets and improvement above 0.0001 for each of LPIPS, DISTS and FloLPIPS at every included point. Comparator rate/quality guards are saved separately; no population or statistical significance claim is made. FID is not computed for this sample size.
''')
 (ROOT/'README.md').write_text('''# GVC-RT oracle v2 final validation

Frozen codec and weights; no training. Four datasets, calibration QP2/7, and preselected independent video identities. See audit/v2_design.md and configs/experiment.json for the protocol registered before new GPU work.

Run `PYTHON -B src/driver.py` with the gvc-rt environment after `src/prepare.py` has completed. The driver is resumable and serializes scheduler ownership with a file lock. Only GPUs 4-7 may run workers, one worker per GPU with at least 7000 MiB free. Per-attempt stdout/stderr, GPU snapshots and actual exit codes are saved under logs/. The 0.5% one-sided rate band includes every control byte; no empty strategy is a valid candidate. No rate- or quality-based tolerance relaxation is performed.

Reused calibration records remain at their immutable v1 server paths with SHA256 indexes. New streams, symbols, features and representative lossless reconstruction videos are under artifacts/ (server-only; indexed for reproduction). Raw per-video, per-strategy metrics and all three random seeds are exported under metrics/ and oracle/. See manifests/run_status.json for actual state. Dataset and model files are excluded from Git. No research discussion is included.
''')
 print('PROTOCOL REGISTERED',manifest['config_sha256'],flush=True)

if __name__=='__main__':prepare()
