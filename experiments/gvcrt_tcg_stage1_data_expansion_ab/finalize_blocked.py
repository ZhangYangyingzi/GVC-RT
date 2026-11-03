"""Preserve completed independent work and publish a plainly blocked result bundle."""
from ab_common import *
from gtflow import write,METRICS
import csv,collections
def main():
 with lock('blocked_finalize.lock'):
  assert load(ROOT/'audits/data_source_conflict.json')['status']=='BLOCKED'
  assert load(ROOT/'inventory/status.json')['status']=='PASS'
  # Parent GT-flow scoring is independent of the disputed future train split.
  with (ROOT/'gtflow.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX)
  assert load(ROOT/'evaluation/parent_gtflow/run_status.json')['status']=='PASS'
  points=[load(f) for f in sorted((ROOT/'evaluation/parent_gtflow/points').glob('*.json'))];assert len(points)==156
  keys=set();flow_hashes=collections.defaultdict(set)
  for point in points:
   r=point['result'];key=(r['domain'],r['sequence'],r['QP'],r['step']);assert key not in keys;keys.add(key)
   assert point['status']=='PASS' and point['input_stream_reused'] and not point['fresh_encode'] and point['independent_decode']
   name=Path(r['bitstream_file']).stem;old=load(P/f'evaluation/points/{name}.json')
   assert point['reconstruction_sha256']==old['reconstruction_sha256'] and r['checkpoint_sha256']==old['result']['checkpoint_sha256']
   stream=P/'large_artifacts/evaluation/bitstreams'/r['bitstream_file'];assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==point['bytes_consumed']==r['real_bytes']
   frames=list(csv.DictReader((ROOT/f'evaluation/parent_gtflow/frames/{name}.csv').open()));trans=list(csv.DictReader((ROOT/f'evaluation/parent_gtflow/transitions/{name}.csv').open()));assert len(frames)==64 and len(trans)==63
   assert sum(int(f['real_bits']) for f in frames)==8*r['real_bytes']
   for m in METRICS:assert math.isfinite(r[m])
   for m in ('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError'):assert abs(r[m]-old['result'][m])<1e-6
   for m in ('WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC','FloLPIPS'):assert abs(sum(float(t[m]) for t in trans)/63-r[m])<1e-10
   for m in ('MS_SSIM','LPIPS','DISTS'):assert abs(sum(float(f[m]) for f in frames)/64-r[m])<1e-10
   for t in trans:flow_hashes[(r['domain'],r['sequence'],int(t['to_frame']))].add(t['GT_flow_sha256'])
  expected={(v['dataset'],v['name'],q,s) for vs in load(P/'evaluation/manifest.local.json').values() for v in vs for q in (0,5,9) for s in (0,1000,2500,5000)}
  assert keys==expected and len(flow_hashes)==13*63 and all(len(x)==1 for x in flow_hashes.values())
  assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
  entries=load(ROOT/'inventory/qualified.json');assert len(entries)==8859 and all(r['status'] in ('ELIGIBLE','EXCLUDED') for r in entries)
  old=__import__('audit').rows();oldids={Path(r['video']).stem for r in old if r['domain']=='ulong'};available={r['identity']:r for r in entries};assert len(oldids)==936
  old_verified=[]
  for i in sorted(oldids):
   record=available[i];expected_hash=next(r['source_sha256'] for r in old if Path(r['video']).stem==i)
   assert record['status']=='ELIGIBLE' and record['sha256']==expected_hash;old_verified.append(dict(identity=i,sha256=expected_hash))
  reserve=load(ROOT/'manifests/reserved_union.local.json')['records'];reserved_ids={r['identity'] for r in reserve};protected=[r for r in entries if r['status']=='ELIGIBLE' and r['identity'] in reserved_ids]
  relations=[]
  for i in sorted(oldids):
   aa=[int(h,16) for h in available[i]['fingerprint_first64_dhash']]
   for b in protected:
    dist=[(a^int(h,16)).bit_count() for a,h in zip(aa,b['fingerprint_first64_dhash'])]
    if max(dist)<=4:relations.append(dict(train_identity=i,reserved_identity=b['identity'],three_anchor_Hamming_distances=dist,relation='UNRESOLVED_POTENTIAL_SAME_SOURCE_GROUP'))
  dump(ROOT/'audits/complete_old_reserved_fingerprint_screen.json',dict(status='BLOCKED',old_identities_checked=936,protected_archive_identities_checked=len(protected),relations=relations,threshold='All three native first64 anchor dHashes within4 bits',verified_parent_file_hashes=old_verified))
  conflict=load(ROOT/'audits/data_source_conflict.json')
  protocol=dict(task='tcg-stage1-data-expansion-ab-20261103',status='NOT_FROZEN_BLOCKED_SOURCE_GROUP_SPLIT_ISOLATION',baseline_commit=BASELINE,parent_checkpoint_filename='step_5000.pt',parent_checkpoint_sha256=PARENT_SHA,parent_absolute_optimizer_step=5000,N_A=936,N_B=None,S=None,physical_GPUs=[4,5,6,7],A_B_formal_additional_updates=0,reason=conflict['reason'],fixed_intended_protocol='EXECUTION_PROTOCOL.md',no_budget_assigned=True,no_train_dev_test_split_assigned=True,immutable_A_B_plans_created=False,parent_GTFlow_complete=True,parent_GTFlow_points=156,old_WarpingError_alias='WarpingError_ReconFlow_PWC',GTFlow_definition='PWC(GT_t,GT_t-1); existing_backwarp(recon_t-1,flow); full native RGB L1 mean then mean63',PWC_sampler_and_weight_sources='audits/GTFlow_sources.json',never_claim_exact_S2VC_reproduction=True)
  dump(ROOT/'protocol.json',protocol)
  dump(ROOT/'split_audit.json',dict(status='BLOCKED',N_A=936,N_B=None,S=None,parent_training_identity_count=940,reserved_identity_count=len(reserve),identity_and_file_hash_direct_overlap=[],source_group_relation=conflict,full_old_reserved_fingerprint_screen='audits/complete_old_reserved_fingerprint_screen.json',train_dev_test_split_frozen=False,source_independence_certified=False,old936_not_silently_removed=True,reserved_sources_not_released=True))
  dump(ROOT/'source_manifest.json',dict(baseline_commit=BASELINE,parent_checkpoint=dict(filename='step_5000.pt',sha256=PARENT_SHA,size_bytes=(P/'checkpoints/step_5000.pt').stat().st_size),parent_provenance=load(ROOT/'audits/source_and_parent.json'),code={f.name:sha(f) for f in ROOT.glob('*.py')},inventory_sha256=sha(ROOT/'inventory/qualified.json'),archives_manifest_sha256=sha(ROOT/'inventory/archives.json'),GTFlow_sources=load(ROOT/'audits/GTFlow_sources.json'),historical_manifest_index_sha256=sha(ROOT/'manifests/historical_project_identities.json')))
  rows=[dict(p['result'],split='historical_reference',model_state='parent'+str(p['result']['step']),evaluation_kind='parent_gtflow') for p in points]
  write(ROOT/'metrics.csv',rows);write(ROOT/'warping_error.csv',[{k:r[k] for k in ('domain','sequence','QP','step','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')} for r in rows])
  (ROOT/'paired_deltas.csv').write_text('split,domain,sequence,QP,target,reference,metric,absolute_delta,relative_delta\n')
  states={arm:dict(status='BLOCKED',committed_additional_updates=0,absolute_optimizer_step=5000,rank_samples=0,checkpoint_status='NO_NEW_TRAINING_CHECKPOINT; parent5000 read-only and verified') for arm in ('A_old_pool','B_expanded_ulong')}
  for arm,st in states.items():dump(ROOT/f'{arm}/training_summary.json',st)
  dump(ROOT/'final_integrity.json',dict(status='BLOCKED',failure_stage='source_group_split_isolation',training=states,parent_GTFlow=dict(status='PASS',completed_points=156,expected_points=156,complete_Cartesian_product=True,stream_checkpoint_reconstruction_hashes=True,old_metric_agreement_abs_tolerance=1e-6,frame_transition_aggregations=True,common_GT_flow_hashes=True),development=dict(status='NOT_RUN',completed_points=0),new_final_test=dict(status='NOT_RUN',completed_points=0),inventory=load(ROOT/'inventory/status.json'),parent_unchanged=True,source_split_independence=False,scientific_conclusions='NOT_EVALUATED'))
  lines=['# TCG data expansion: blocked source-group isolation','', 'Overall execution status: BLOCKED. A/B formal optimizer updates: 0. No new training checkpoint. Parent absolute optimizer step remains5000.','', 'Completed: local ZIP CRC/SHA/native-first64 qualification and parent156-point GT-flow rescore.','Uncompleted: source-disjoint immutable A/B/dev/test plans, budget assignment, A/B training and development/new final-test evaluations.','', 'The parent-required training identity `'+conflict['train_identity']+'` and protected historical validation identity `'+conflict['reserved_identity']+'` depict the same interview/person/camera composition. Three-anchor dHash distances were3/3/2; ORB comparison found699 inliers among711 descriptor matches at960x540. The two files have different SHA-256 and lengths878/2825 frames. These observations do not establish an exact temporal duplicate or original-upload ID. They prevent certifying source-group separation from the supplied UUID metadata.','The protocol requires the old936 U-Long identities to remain in B and protects reserved sources. Neither condition was relaxed. Source linkage resolution is required before training can proceed.','', 'Parent source checkpoint and code hashes were verified; complete frozen-model reconstruction hash matches the parent. Old-pool input/flow/loss/gradient engineering replay audits passed without optimizer updates.','Parent rescore: input_stream_reused=true, fresh_encode=false, independent_decode=true. Checkpoint/stream/reconstruction hashes and old metric agreement were independently checked. GT-flow hashes agree across all checkpoint/QP evaluations for a fixed sequence transition.','GT-flow PWC definition and sampler tests: gtflow.py and audits/GTFlow_*.json. Existing WarpingError is preserved as WarpingError_ReconFlow_PWC. This is not claimed as exact S2VC reproduction.','No FVD/FID or BD-rate was computed. No scientific effect interpretation or model-selection decision is made.','', 'Engineering deviations: an initial import-name collision failed before scoring and was corrected; failure logs are retained. A foreground source-relation QC decoder briefly overlapped the four inventory workers; the complete inventory and final source-screen checks were subsequently performed and recorded, and no claim of a strict global four-decoder cap is made for that exploratory overlap.','', '| Domain | Sequence | QP | Parent step | '+' | '.join(METRICS)+' |','|---|---|---:|---:|'+'|'.join(['---:']*len(METRICS))+'|']
  for r in rows:lines.append('| '+r['domain']+' | '+r['sequence']+' | '+str(r['QP'])+' | '+str(r['step'])+' | '+' | '.join(str(r[k]) for k in METRICS)+' |')
  (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
  status('source_group_split_isolation','BLOCKED',formal_optimizer_updates=0,N_A=936,N_B=None,S=None,parent_GTFlow_completed_points=156,development_completed_points=0,final_test_completed_points=0,report_path='REPORT.md',resume_command=f'{PYTHON} -B -u {ROOT}/runner.py',resume_precondition='Resolve documented source-group relationship under an explicit protocol; current resume intentionally retains the block')
  result=subprocess.run([PYTHON,'-B','-u',str(ROOT/'publish.py'),'--blocked'],cwd=REPO)
  if result.returncode:raise RuntimeError('BLOCKED_RESULT_PUBLICATION_FAILED; see publication_status.json')
if __name__=='__main__':main()
