"""Objective dynamic-state tables, coverage, time ledgers and measured curves."""
from v3_common import *
from gtflow import METRICS
import gzip
def read(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def main():
 verify_sources();selection=load(ROOT/'selected_models_v3.json');final_info=load(ROOT/'final_evaluation_v3.json');final=read(ROOT/final_info['metrics_file']);assert len(final)==selection['expected_final_points']
 dev=[]
 for p in sorted((ROOT/'evaluation_v3').glob('development_*/metrics.csv')):dev.extend(read(p))
 write_csv(ROOT/'metrics_v3.csv',final);write_csv(ROOT/'development_metrics_v3.csv',dev)
 lookup={(r['split'],r['domain'],r['sequence'],int(r['QP']),r['model_state']):r for r in final};pairs=[];alias=selection['role_to_unique_state']
 comparisons=[('same_budget','B_common','A_common'),('independent_dev_best','B_best_dev','A_best_dev')]
 comparisons += [('versus_parent5000',role,'parent5000') for role in ('A_common','B_common','A_best_dev','B_best_dev')]
 comparisons += [('versus_parent0',role,'parent0') for role in ('A_common','B_common','A_best_dev','B_best_dev')]
 for kind,target,reference in comparisons:
  for key,a in lookup.items():
   if key[-1]!=alias[reference]:continue
   b=lookup[(*key[:-1],alias[target])]
   for metric in METRICS:
    av,bv=float(a[metric]),float(b[metric]);delta=bv-av;direction=1 if metric in ('PSNR','MS_SSIM') else -1
    pairs.append(dict(comparison_type=kind,target_role=target,reference_role=reference,split=key[0],domain=key[1],sequence=key[2],QP=key[3],metric=metric,reference_value=av,target_value=bv,absolute_delta=delta,relative_delta=delta/abs(av) if av else None,numerical_improvement=direction*delta>0,denominator='abs(reference_value); undefined for zero',same_budget=kind=='same_budget'))
 write_csv(ROOT/'paired_deltas_v3.csv',pairs)
 summary=[]
 groups=collections.defaultdict(list)
 for r in pairs:groups[(r['comparison_type'],r['target_role'],r['reference_role'],r['split'],r['domain'],r['QP'],r['metric'])].append(r)
 for key,rs in sorted(groups.items()):
  relative=[r['relative_delta'] for r in rs if r['relative_delta'] is not None]
  summary.append(dict(comparison_type=key[0],target_role=key[1],reference_role=key[2],split=key[3],domain=key[4],QP=key[5],metric=key[6],points=len(rs),unique_videos=len({r['sequence'] for r in rs}),mean_absolute_delta=sum(r['absolute_delta'] for r in rs)/len(rs),mean_point_relative_delta=sum(relative)/len(relative) if relative else None,relative_point_denominator=len(relative),numerical_improvement_points=sum(r['numerical_improvement'] for r in rs),weighting='equal video weight at fixed QP; QPs are not independent videos'))
 write_csv(ROOT/'paired_summary_v3.csv',summary)
 traces=[];coverage={};arms={};checkpoints={};segments=load(ROOT/'training_segments_v3.json') if (ROOT/'training_segments_v3.json').exists() else []
 for arm in ARMS:
  n=committed(arm);plan=rows(arm)[:4*n];stop=early_state(arm);assert stop['stop_reason'] in ('EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED')
  ul=[r for r in plan if r['domain']=='ulong'];pool=protocol()['N_A' if arm==ARMS[0] else 'N_B'];identities={Path(r['video']).stem for r in ul}
  coverage[arm]=dict(actual_rank_samples=len(plan),ULong_samples=len(ul),UVG_samples=len(plan)-len(ul),unique_ULong_identities=len(identities),qualified_pool_identities=pool,ULong_identity_coverage_ratio=len(identities)/pool,unique_ULong_windows=len({digest([r['source_sha256'],r['source_frame_indices']]) for r in ul}),unique_ULong_window_crop_combinations=len({digest([r['source_sha256'],r['source_frame_indices'],r['crop_x'],r['crop_y']]) for r in ul}),complete_coverage_passes=sum(len({Path(r['video']).stem for r in ul if r.get('coverage_pass')==p})==pool for p in (1,2,3)) if arm==ARMS[1] else None,full_expanded_pool_seen=len(identities)==pool if arm==ARMS[1] else None)
  boot=load(MIG/'bootstrap.json')['arms'][arm];prefix=boot['legacy_prefix_additional_step'];all_receipts=[]
  for step in range(1,n+1):all_receipts.append(load(ROOT/arm/('receipts' if step<=prefix else 'receipts_v3')/f'{step:06d}.json.gz'))
  for receipt in all_receipts:
   traces.append(dict(arm=arm,additional_step=receipt['additional_step'],absolute_optimizer_step=receipt['absolute_optimizer_step'],physical_world_size=receipt.get('physical_world_size',4),gradient_norm=receipt['rank_audits'][0]['reduced_preclip_gradient_norm'],**receipt['mean_losses']))
  active=[r for r in segments if r['arm']==arm];allocated=sum(r['allocated_GPU_seconds'] for r in active);elapsed=sum(r['elapsed_seconds'] for r in active)
  legacy=load(MIG/'legacy_gpu_time.json').get(arm,{});allocated+=legacy.get('allocated_GPU_seconds',0);elapsed+=legacy.get('phase_wall_seconds',0)
  arms[arm]=dict(stop_reason=stop['stop_reason'],actual_additional_updates=n,absolute_optimizer_step=5000+n,best_dev_step=stop['best_raw']['step'],best_dev_M=stop['best_raw']['M'],improvement_anchor=stop['improvement_anchor'],bad_check_count=stop['bad_check_count'],theoretical_stop_step=stop['theoretical_stop_step'],actual_stop_step=n,training_phase_wall_seconds=elapsed,training_allocated_GPU_seconds=allocated,GPU_time_definition='Observed physical-card allocation wall seconds, including startup/I-O; not CUDA kernel utilization time',legacy_prefix_additional_updates=prefix)
  checkpoints[arm]=[dict(filename=str(p.relative_to(ROOT/arm)),size_bytes=p.stat().st_size,sha256=sha(p)) for folder in ('checkpoints','checkpoints_v3') for p in sorted((ROOT/arm/folder).glob('*.pt'))]
 write_csv(ROOT/'training_loss_v3.csv',traces);dump(ROOT/'data_coverage_v3.json',coverage);dump(ROOT/'checkpoints_manifest_v3.json',dict(arms=checkpoints,large_files_local_only=True))
 decisions=[]
 for arm in ARMS:
  for row in early_state(arm)['checks']:decisions.append(dict(arm=arm,**row))
 write_csv(ROOT/'development_scores_v3.csv',decisions)
 write_csv(ROOT/'warping_error_v3.csv',[{k:r[k] for k in ('split','domain','sequence','QP','model_state','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')} for r in final])
 aggregates=[]
 for rows_,kind in ((final,'final'),(dev,'development')):
  grouped=collections.defaultdict(list)
  for r in rows_:grouped[(r['split'],r['domain'],int(r['QP']),r['model_state'])].append(r)
  for key,rs in sorted(grouped.items()):aggregates.append(dict(kind=kind,split=key[0],domain=key[1],QP=key[2],model_state=key[3],video_count=len(rs),weighting='equal video weight within QP',**{m:sum(float(r[m]) for r in rs)/len(rs) for m in METRICS}))
 write_csv(ROOT/'summary_by_domain_QP_v3.csv',aggregates)
 stream_audit(final,ROOT/'evaluation_v3'/final_info['batch_id'])
 plot(final,dev,traces,aggregates,decisions)
 info=dict(status='NUMERICAL_RESULTS_READY_PENDING_FINAL_INTEGRITY',protocol_revision=V3,arms=arms,K_common=selection['K_common'],common_effective_samples_per_arm=4*selection['K_common'],coverage=coverage,unique_final_models=len(selection['states']),final_completed_points=len(final),final_expected_points=selection['expected_final_points'],development_completed_points=len(dev),parent_GTFlow_reused_points=156,test_selection_manifest_sha256=sha(ROOT/'selected_models_v3.json'),scientific_interpretation='NOT_PERFORMED')
 dump(ROOT/'training_summary_v3.json',info)
 lines=['# TCG A/B online development early stopping v3','', 'Numerical execution record; final acceptance: final_integrity_v3.json. Publication: publication_status_v3.json.','', '```json',json.dumps(info,indent=2,ensure_ascii=False),'```','', 'The protocol was revised during A training at additional595/absolute5595. Old protocols, native_geometry_v1, checkpoint595 and source hashes are retained under migrations/ and repairs/. Existing committed updates were not restarted.','The development stopping statistic is the mean over QP0/5/9 of equal32-video FloLPIPS means divided by the same-QP parent5000 means. Mode min, absolute normalized min_delta0.005,patience3. Baseline does not count as a patience check. best_raw and improvement_anchor are distinct. Exact ties retain the earlier checkpoint.','K_common is the largest intersection of completed planned validation steps and is independent of performance gaps. A/B dev-best comparisons may use unequal training budgets and are labeled separately. The final model manifest was frozen before new test evaluation; identical model states are deduplicated.','FloLPIPS dev-best does not assert optimality on other metrics. Legal early stopping does not establish convergence or effectiveness. No scientific interpretation, BD-rate extrapolation or new hyperparameter search is provided.','Training prefix, actual visited identities/windows/ULong sample counts and coverage fractions are reported above; incomplete expanded-pool coverage is not called full-data training. Allocation time is observed wall time multiplied by assigned physical cards, not kernel-active time.','PWC sampler remains pixel-center grid/align_corners=False/W−1/H−1 flow normalization/invalid-zero/full-image denominator; width11 dx1 means1.1 sampling pixels. No exact S2VC implementation reproduction claim.','GT flow is offline-only; raw independent decoder tensors produce neural metrics, local JPEGs are viewing previews only. Each unique model/video/QP point has its own fresh RANS stream and independent decoder. Actual rates are retained; no equal-rate claim is inferred from equal QP.','Historical13 references are separate from new_holdout_test32. Parent156 GT-flow reused scores are preserved separately and were not used as the new development baseline.','Directory initialization and native-geometry metadata repair history: ENGINEERING_REPAIR.md. The original source/data assertions and1e-3 numerical criteria were retained. Old source/protocol snapshots are not relabeled as v3 passes.','', '| Split | Sequence | QP | Unique model | '+' | '.join(METRICS)+' |','|---|---|---:|---|'+'|'.join(['---:']*len(METRICS))+'|']
 for r in final:lines.append('| '+' | '.join(str(r[k]) for k in ('split','sequence','QP','model_state'))+' | '+' | '.join(str(r[m]) for m in METRICS)+' |')
 lines += ['','Complete development scores/patience ledger: development_scores_v3.csv and per-arm early_stop_v3.json.','Full point/frame/transition and real-stream receipts: evaluation_v3/.','Same-budget and dev-best paired results: paired_deltas_v3.csv and paired_summary_v3.csv.','Original and v3 failure records remain in audits/, migrations/, and bounded log summaries.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def stream_audit(final,directory):
 grouped=collections.defaultdict(list);records=[];diagnoses=[]
 for r in final:grouped[(r['split'],r['domain'],r['sequence'],int(r['QP']))].append(r)
 for key,rs in sorted(grouped.items()):
  reference=sorted(rs,key=lambda r:r['model_state'])[0];a=read(directory/'encoder_reference'/(Path(reference['bitstream_file']).stem+'.csv'))
  records.append(dict(split=key[0],domain=key[1],sequence=key[2],QP=key[3],models=[r['model_state'] for r in rs],stream_sha256=[r['bitstream_sha256'] for r in rs],real_bytes=[int(r['real_bytes']) for r in rs],all_stream_bytes_identical=len({r['bitstream_sha256'] for r in rs})==1))
  for r in rs:
   if r['bitstream_sha256']==reference['bitstream_sha256']:continue
   b=read(directory/'encoder_reference'/(Path(r['bitstream_file']).stem+'.csv'));assert len(a)==len(b)==64 and all(x['actual_qp']==y['actual_qp'] for x,y in zip(a,b))
   different=[i for i,(x,y) in enumerate(zip(a,b)) if x['payload_sha256']!=y['payload_sha256']];assert different and different[0]>0,'UNEXPECTED_I_FRAME_STREAM_DIFFERENCE'
   first=different[0];changed=[field for field in ('reference_frame_sha256','reference_feature_sha256') if a[first-1][field]!=b[first-1][field]]
   diagnoses.append(dict(sequence=key[2],QP=key[3],reference=reference['model_state'],target=r['model_state'],first_payload_difference_frame=first,preceding_reference_fields_different=changed,resolved=bool(changed),record='Actual encoded payload differs after differing reconstructed reference state' if changed else 'Unresolved difference requires engineering inspection'))
 dump(ROOT/'stream_comparison_v3.json',dict(records=records,no_rate_overwrite=True,engineering_comparisons=diagnoses,status='PASS' if all(r['resolved'] for r in diagnoses) else 'FAILED'))
 assert all(r['resolved'] for r in diagnoses),'UNRESOLVED_BITSTREAM_DIFFERENCE'
def plot(final,dev,traces,aggregates,decisions):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 quality=[m for m in METRICS if m not in ('bpp','kbps','WarpingError')]
 grouped=collections.defaultdict(list)
 for r in final:grouped[(r['split'],r['domain'],r['sequence'])].append(r)
 for keys,rs in grouped.items():
  for metric in quality:
   fig,ax=plt.subplots()
   for label in sorted({r['model_state'] for r in rs}):
    pp=sorted((r for r in rs if r['model_state']==label),key=lambda r:float(r['bpp']));ax.plot([float(r['bpp']) for r in pp],[float(r[metric]) for r in pp],marker='o',label=label)
   ax.set(xlabel='Measured bpp',ylabel=metric,title=' / '.join(keys));ax.legend();fig.tight_layout();fig.savefig(out/f'v3_{keys[-1]}_{metric}.png',dpi=120);plt.close(fig)
 domains=collections.defaultdict(list)
 for r in aggregates:
  if r['kind']=='final':domains[(r['split'],r['domain'])].append(r)
 for key,rs in domains.items():
  for metric in quality:
   fig,ax=plt.subplots()
   for label in sorted({r['model_state'] for r in rs}):
    points=sorted((r for r in rs if r['model_state']==label),key=lambda r:r['bpp']);ax.plot([r['bpp'] for r in points],[r[metric] for r in points],marker='o',label=label)
   ax.set(xlabel='Equal-video mean measured bpp',ylabel=metric,title=' / '.join(key));ax.legend();fig.tight_layout();fig.savefig(out/f'v3_domain_{key[0]}_{key[1]}_{metric}.png',dpi=120);plt.close(fig)
 for metric in ('total_loss','frame_loss','motion_loss','gradient_norm'):
  fig,ax=plt.subplots()
  for arm in ARMS:
   pp=[r for r in traces if r['arm']==arm];ax.plot([r['additional_step'] for r in pp],[r[metric] for r in pp],label=arm)
  ax.set(xlabel='Committed additional updates',ylabel=metric);ax.legend();fig.tight_layout();fig.savefig(out/f'v3_training_{metric}.png',dpi=120);plt.close(fig)
 for metric in quality:
  fig,ax=plt.subplots()
  for arm in ARMS:
   steps=early_state(arm)['checks'];values=[]
   for item in steps:
    rr=[r for r in dev if r['model_state']==f'{arm}_add{item["step"]}'];assert len(rr)==96;values.append(sum(float(r[metric]) for r in rr)/96)
   ax.plot([r['step'] for r in steps],values,marker='o',label=arm)
  ax.set(xlabel='Additional updates',ylabel='Fixed development mean '+metric);ax.legend();fig.tight_layout();fig.savefig(out/f'v3_development_{metric}.png',dpi=120);plt.close(fig)
 fig,ax=plt.subplots()
 for arm in ARMS:
  pp=early_state(arm)['checks'];ax.plot([0]+[r['step'] for r in pp],[1]+[r['M'] for r in pp],marker='o',label=arm)
 ax.set(xlabel='Additional updates',ylabel='Normalized development FloLPIPS M');ax.legend();fig.tight_layout();fig.savefig(out/'v3_development_monitor_M.png',dpi=120);plt.close(fig)
if __name__=='__main__':main()
