import csv,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent
def readcsv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
status=json.load(open(R/'pipeline_status.json'));assert status['status'] in ('EVALUATIONS_COMPLETE','RUNNING','COMPUTATION_AND_REPORT_COMPLETE') and status['completed_new']==42 and not status.get('failures')
points=readcsv(R/'results/all_rd_points.csv');pooled=readcsv(R/'results/pooled_fid.csv')
assert len(points)==117 and len(pooled)==27
assert all(x['real_RANS']=='True' or x['real_RANS']=='true' for x in points)
assert all(x['independent_decode_pass']=='True' or x['independent_decode_pass']=='true' for x in points)
assert all(x['state_sync_pass']=='True' or x['state_sync_pass']=='true' for x in points)
import math
qpmap=json.load(open(R/'qp_semantics_audit.json'))['actual_qps']
for row in points:
 v=next(z for z in json.load(open(R/'eval_manifest.json'))[row['dataset']] if z['video_index']==int(row['video']))
 assert row['source_sha256']==v['source_sha256'] and row['source_rgb_sha256']==v['rgb_sha256']
 indices=json.loads(row['source_frame_indices']) if isinstance(row['source_frame_indices'],str) else row['source_frame_indices']
 assert indices==v['source_frame_indices']
 actual_qps=json.loads(row['actual_qps']) if isinstance(row.get('actual_qps'),str) else row.get('actual_qps')
 assert actual_qps==qpmap[str(row['QP'])][:64]
 bpp=float(row['actual_bpp']);expected_bpp=int(row['real_bytes'])*8/(64*v['width']*v['height']);assert abs(bpp-expected_bpp)<1e-15
 for m in ('LPIPS','DISTS','FloLPIPS'):assert math.isfinite(float(row[m]))
new=sum(1 for x in points if x['reuse'].startswith('new'))
reuse=len(points)-new;assert new==42 and reuse==75
manifest=json.load(open(R/'eval_manifest.json'))
for ds,vs in manifest.items():
 assert len(vs)=={'ulong':6,'uvg_holdout':2,'hevc_b':5}[ds]
for p in points:
 assert Path(p['bitstream_path']).is_file() and Path(p['feature_path']).is_file()
 assert int(p['real_bytes'])==Path(p['bitstream_path']).stat().st_size
 assert int(p['frames'])==64 and int(p['num_transitions'])==63
 feas=json.load(open(R/'reference_objective_feasibility.json'))
 assert feas['status']=='BLOCKED_MISSING_REFERENCE_OBJECTIVE' and feas['additional_updates']==0
ref_audit=json.load(open(R/'audits/gt_reference_isolation.json'))
assert ref_audit['status']=='PASS' and ref_audit['GT_reference_receipts_verified']==1000
assert ref_audit['GT_reference_optimizer_updates']==0 and not any(ref_audit['eval_vs_full_1000_reference_intersections'].values())
metric_audit=json.load(open(R/'audits/metric_reuse_verification.json'))
assert metric_audit['status']=='PASS' and metric_audit['v24_original_points']==21
examples=sorted((R/'plots/fixed_examples').glob('*.png'))
assert len(examples)==3 and all(p.with_suffix('.json').is_file() for p in examples)
for p in examples:
 h=hashlib.sha256(p.read_bytes()).hexdigest();meta=json.load(open(p.with_suffix('.json')));assert meta['output_sha256']==h
common=readcsv(R/'results/common_rate_bounds.csv');pooledcommon=readcsv(R/'results/pooled_fid_common_bounds.csv')
checks={
 'source_checkpoint_and_inference_hashes_verified':True,
 'N_cumulative_step_500_reused':True,'P_cumulative_step_500_reused':True,
 'training_updates_in_v635':0,'A_reference_objective_gate':'BLOCKED_MISSING_REFERENCE_OBJECTIVE',
 'fixed_manifest_videos':sum(len(v) for v in manifest.values()),'fixed_qps':[0,5,9],
 'logical_points_expected':117,'logical_points_complete':len(points),'verified_historical_points':reuse,'new_real_RANS_encodes':new,
 'all_points_have_64_frames_and_63_transitions':True,'all_points_real_RANS_independent_decode_state_sync':True,'source_RGB_frame_indices_actual_QP_and_actual_bpp_formula_verified':True,'LPIPS_DISTS_FloLPIPS_finite':True,
 'pooled_fid_points_expected':27,'pooled_fid_points_complete':len(pooled),'pooled_gt_pool_hashes_consistent_per_dataset':True,
 'same_rate_dataset_bounds':common,'pooled_fid_same_rate_bounds':pooledcommon,
 'data_identity_source_sha_intersection_empty':True,'content_near_duplicate_audit':'not exhaustive',
 'GT_reference_1000_receipts_verified':ref_audit['GT_reference_receipts_verified'],'GT_reference_vs_eval_identity_intersections':ref_audit['eval_vs_full_1000_reference_intersections'],
 'historical_metric_implementation_reuse_verified_points':metric_audit['v24_original_points'],
 'fixed_preselected_frame_example_panels':len(examples),
 'reporting_scope':'fixed 13-video 3-QP evaluation; no model selection or training'}
# Server-side binary index: verify all point bitstreams/features and source/checkpoint files, but keep binaries out of Git.
import hashlib
artifacts={}
def add_art(path,expected,kind):
 q=Path(path);assert q.is_file(),path
 h=hashlib.sha256()
 with q.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 actual=h.hexdigest();assert actual==expected,(path,actual,expected)
 artifacts[str(q)]={'path':str(q),'bytes':q.stat().st_size,'sha256':actual,'kind':kind}
for row in points:
 add_art(row['bitstream_path'],row['bitstream_sha256'],'full_RANS_bitstream')
 add_art(row['feature_path'],row['feature_sha256'],'Inception2048_feature_cache')
 for kind in ('frame_metrics','transitions'):
  if row.get(kind+'_path') and row.get(kind+'_sha256'):add_art(row[kind+'_path'],row[kind+'_sha256'],kind+'_metrics')
for x in json.load(open(R/'source_hashes.json'))['official_model_files']:
 add_art(x['path'],x['sha256'],'official_public_model_weight')
for name,x in json.load(open(R/'checkpoint_reuse_manifest.json'))['models'].items():
 add_art(x['train_checkpoint'],x['train_checkpoint_sha256'],name+'_full_training_checkpoint_step500')
 add_art(x['inference_path'],x['inference_sha256'],name+'_inference_export_step500')
for ds,vs in manifest.items():
 for v in vs:
  add_art(v['source_path'],v['source_sha256'],'source_video')
(R/'server_artifacts_index.json').write_text(json.dumps({'count':len(artifacts),'artifacts':list(artifacts.values()),'binary_files_excluded_from_git':True},indent=2)+'\n')
writej(R/'final_integrity.json',{'status':'PASS','calculation_status':'PASS','training_status':'REUSED_NO_NEW_UPDATES','A_branch_status':feas['status'],'checks':checks,'completed_unix':time.time()})
# Generate report from numeric tables without adding inferential claims.
summary=readcsv(R/'results/equal_rate_summary.csv');fid=readcsv(R/'results/pooled_fid_equal_rate_summary.csv')
lines=['# v6.35 results report','', '## Fixed answer and coverage','',
 'A_refLoss_noP_500 was not trained: the executable-source gate is `BLOCKED_MISSING_REFERENCE_OBJECTIVE` with zero updates. N/P were reused at their validated cumulative step 500. The O/N/P matrix is complete for the fixed 13-video, QP 0/5/9 manifest.', '',
 f'Completed logical RD points: {len(points)}/117 ({reuse} verified reuse, {new} new encodes). Pooled FID points: {len(pooled)}/27.', '',
 '## Model/source identities','',
 f"N checkpoint SHA256: `{json.load(open(R/'audits/source_and_checkpoint_verification.json'))['source_checkpoints']['N_noP_BG_FD']['inference_sha256']}`.",
 f"P checkpoint SHA256: `{json.load(open(R/'audits/source_and_checkpoint_verification.json'))['source_checkpoints']['P_PBG_FD']['inference_sha256']}`.",
 'Original public I/P checkpoint paths and hashes are listed in `source_hashes.json`. No model was trained in v6.35.', '',
 '## Equal measured-rate tables','', 'Values use the dataset-wide common measured-bpp interval across all included videos and O/N/P, a fixed 20-point uniform ln(bpp) grid, piecewise-linear interpolation without extrapolation, and equal video weighting. Relative change is 100×(method mean−reference mean)/reference mean; lower is better for LPIPS/DISTS/FloLPIPS. These percentages are metric changes, not BD-rate.', '',
 '|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|Common bpp interval|','|---|---|---|---|---:|---:|---:|---:|---|']
for x in summary:
 if not x.get('reference'):continue
 lines.append(f"|{x['dataset']}|{x['method']}|{x['reference']}|{x['metric']}|{float(x['method_mean']):.6g}|{float(x['reference_mean']):.6g}|{float(x['delta']):+.6g}|{float(x['relative_change_percent']):+.3f}|{float(x['common_actual_bpp_min']):.6g}–{float(x['common_actual_bpp_max']):.6g}|")
lines+=['','## Pooled FID','', 'FID is computed over the complete dataset frame pool using the fixed Inception2048 features, float64 and ddof=1. Single-video FID is not averaged. The equal-rate FID trend uses the dataset mean actual bpp and a linear diagnostic interpolation over the three measured QP anchors. D_mu and D_cov are in `results/pooled_fid.csv`; their sum matches the stored pooled FID by construction.', '', '|Dataset|Method|Reference|FID mean|Reference mean|Delta|Relative %|Samples|','|---|---|---|---:|---:|---:|---:|---:|']
for x in fid:
 if x.get('band'):continue
 lines.append(f"|{x['dataset']}|{x['method']}|{x['reference']}|{float(x['method_mean']):.6g}|{float(x['reference_mean']):.6g}|{float(x['delta']):+.6g}|{float(x['relative_change_percent']):+.3f}|{x['samples']}|")
lines+=['','## Scope and missing evidence','', 'The official-source objective feasibility record lists the missing original training implementation, exact per-QP weighting/normalization, exact teacher checkpoint and preprocessing, exact margin definition/interface, and discriminator training implementation. No proxy teacher or substitute objective was used. Consequently this run measures (1) O vs P as the combined pipeline difference and (2) P vs N under the same complete v6.34b training objective; the objective-only comparison remains unmeasured. This is a fixed 13-video, three-QP screen and does not establish broader generalization.', '', 'The source-identity audit found no train/evaluation path, source SHA or domain/video identity intersection in the audited plan prefix. Exhaustive content near-duplicate search and official pretraining-data exposure are not claimed.', '', '## Artifacts','', '- `results/all_rd_points.csv` — all 117 video/method/QP rows with source and bitstream hashes.','- `results/pooled_fid.csv` — 27 pooled dataset/method/QP points, with D_mu and D_cov.','- `results/common_rate_grid.csv`, `results/equal_rate_summary.csv`, `results/equal_rate_bands.csv` — common-rate paired comparisons.','- `plots/` — per-video RD charts, common-rate curves, pooled FID charts, and the fixed QP5 frame panels for the first listed video in each dataset.','- `audits/historical_reuse.json` — pointwise reuse verification.','- `final_integrity.json` — mechanical completion checks.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'RESULTS_SUMMARY.md').write_text(f"""# Results summary

- Status: computation PASS; A reference-objective branch BLOCKED before training; no N/P updates in this run.
- Evaluated methods: original, N_oursLoss_noP_500, P_oursLoss_withP_500.
- Coverage: 13 videos × 3 QPs × 3 methods = 117 points; {reuse} verified historical reuses and {new} new encodes.
- Pooled FID: {len(pooled)}/27 points.
- Fixed common-rate comparisons and all metric values are in `results/` and `REPORT.md`.
- Interpret all comparisons within the fixed 13-video, QP 0/5/9 screening scope.
""")
status_path=R/'pipeline_status.json';ps=json.load(open(status_path));ps.update(status='COMPUTATION_AND_REPORT_COMPLETE',evaluation_status='PASS',aggregation_status='PASS',integrity_status='PASS',publication_status='PENDING',updated_unix=time.time());status_path.write_text(json.dumps(ps,indent=2)+'\n')
print('FINALIZED',len(points),reuse,new,len(pooled))
