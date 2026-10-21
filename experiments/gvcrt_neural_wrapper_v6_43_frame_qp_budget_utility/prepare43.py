from common43 import *
import shutil
assert sha(CP)==CP_SHA
request=Path('/Huang_group/zyyz/home_dir/.codex/attachments/4c9ed50a-959a-4205-b458-ee726e019e33/已粘贴的文本.txt')
p=dict(experiment=ROOT.name,request_sha256=sha(request),optimizer_updates=0,model='G_basic_100',checkpoint_sha256=CP_SHA,datasets={'hevc_b':['ParkScene'],'uvg_holdout':['Jockey']},frames=64,external_qp=5,targets=TARGETS,delta_qp=[2,-2],future_frames=15,primary_metric='LPIPS',seed_count=0,IP=-1,canvas=[1088,1920],crop=[1080,1920],force_zero_thres=.12,codec_precision='float16',metric_input='float32',allowed_gpus=[4,5,6,7],transfer_grid=[[1,1],[1,2],[2,1],[2,2]],transfer_min_separation=16,transfer_selection='max A(r)-C(d), tie lexicographic (r,d)',display_selection='min whole LPIPS within [0.99 B0,B0], tie bytes,(a,b)',max_streams=42,oracle=True,baseline_tolerance='exact bitstream/reconstruction hashes and exact numeric metrics',ranking_cost='same delta_bits_16 for U_now and U_16; positive total and window cost only',state_sync_tolerance='exact tensor SHA256',no_clamp=True)
if (ROOT/'protocol.json').exists():assert load(ROOT/'protocol.json')==p
else:dump(ROOT/'protocol.json',p)
textfile(ROOT/'REQUEST.txt',request.read_text())
for n in ('evaluation_config.json','model_manifest.json'):shutil.copy2(REF/n,ROOT/n)
data=[v for v in load(REF/'data_manifest.json') if v['name'] in ('ParkScene','Jockey')];assert len(data)==2
dump(ROOT/'data_manifest.json',data)
streams=[s for s in load(REF/'stream_manifest.json') if s['sequence'] in ('ParkScene','Jockey') and s['QP']==5];assert len(streams)==2
for s in streams:
 assert sha(s['path'])==s['sha256'] and Path(s['path']).stat().st_size==s['bytes']
 ref=REF/'tasks'/s['dataset']/s['sequence']/'q5/pair.json'
 s['reference_pair_path']=str(ref);s['reference_pair_sha256']=sha(ref)
 s['G_basic']=next(r for r in load(ref)['results'] if r['method']=='G_basic')
dump(ROOT/'baseline_manifest.json',streams)
tasks=[]
for s in streams:
 b=dict(dataset=s['dataset'],sequence=s['sequence'],variant='baseline',actual_qps=s['actual_qps'],kind='baseline',changes={})
 tasks.append(b)
 for t in TARGETS:
  for name,delta in [('qplus2',2),('qminus2',-2)]:
   q=s['actual_qps'].copy();q[t]+=delta
   assert all(0<=x<10 for x in q)
   tasks.append(dict(dataset=s['dataset'],sequence=s['sequence'],variant=f't{t:02d}_{name}',kind='impulse',target=t,label=name,changes={str(t):delta},actual_qps=q))
dump(ROOT/'intervention_plan.json',tasks)
# Fingerprint every historical Python dependency loaded by this adapter and codec.
files=set(REPO.glob('src/**/*.py'))
for name in ['gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control','gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit','gvcrt_parallel_v5a3_v60','gvcrt_neural_wrapper_v6_2b_final_generalization_test','gvcrt_neural_wrapper_v6_4_vimeo_pretrain','gvcrt_neural_wrapper_v4_progressive_joint','gvcrt_neural_wrapper_v6_1_large_diverse_qp09','gvcrt_neural_wrapper_v6_2_objective_calibration']:
 files.update((ROOT.parent/name).glob('*.py'))
for path in load(ROOT/'evaluation_config.json')['source_model_files']:files.add(Path(path))
for n in ['data_manifest.json','stream_manifest.json','evaluation_config.json','model_manifest.json','protocol.json']:files.add(REF/n)
dump(ROOT/'source_manifest.json',[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)])
textfile(ROOT/'PROTOCOL.md','# Fixed protocol\n\nThe full authorized specification is in REQUEST.txt. Machine-readable conditions are in protocol.json. This is an offline GT-assisted oracle diagnostic, with zero training updates.\n\nActual QPs are copied from the v6.42 q5 streams; only listed frame indices are changed, with no shift or clamp. Every stream is encoded from native I0 and all dependent P frames are re-encoded. Every persisted stream is independently decoded using its own QP fields. All state comparisons and baseline reproduction use exact SHA256 checks. Frame packet accounting assigns preceding SPS bytes to that frame, including the initial SPS in frame 0.\n\nWhole PSNR comes from summed RGB SSE. Frame/transition metrics are separate. LPIPS is the primary selection metric; DISTS and L1 are supplemental. Positive efficiencies use a shared window-bit denominator and require positive whole-stream and window cost. Nonpositive costs and negative gains remain in raw tables. Transfer rules, grid, tie breaks and budget intervals are fixed before encoding.\n\nTwo videos and multiple within-video interventions are not independent video samples. Same QP offsets do not imply equal bit costs; impulse gains do not imply fixed-budget gains. QP affects encoding features and Generator conditions together. No pure-latent causal interpretation, deployable policy claim, FID, BD-rate or speed evaluation.\n')
textfile(ROOT/'README.md','# v6.43: fixed G_basic frame-QP budget utility\n\nSee PROTOCOL.md, REQUEST.txt and REPORT.md. Historical experiments are read-only. This experiment has zero optimizer updates.\n')
dump(ROOT/'run_status.json',dict(status='PREPARED',optimizer_updates=0,planned_baselines=2,planned_impulses=24,max_transfers=16))
print('PREPARED',len(tasks))
