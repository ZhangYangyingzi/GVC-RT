from pathlib import Path
import json,shutil,hashlib
R=Path(__file__).resolve().parent;V=R.parent/'gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control'
for d in ['audits','logs','plans','manifests','evaluation','diagnostics','results','plots','cache']: (R/d).mkdir(exist_ok=True)
for n in ['io21.py','adapter.py','evaluate.py','window_training.py','gradient_tools.py','illm_runtime.py','publish_github.py','index_io.py','restore_text_shards.py']:
 s=(V/n).read_text()
 if n=='io21.py':
  s=s.replace("V25=ROOT.parent/", "V26=ROOT.parent/'gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control'\nV25=ROOT.parent/",1).replace("DATASETS=('ulong','uvg_holdout','hevc_b','uvg_validation')","DATASETS=('mcljcv',)")
 if n=='illm_runtime.py':s=s.replace("OFFICIAL=ROOT/'vendor/", "OFFICIAL=V26/'vendor/").replace("WEIGHT=ROOT/'weights/", "WEIGHT=V26/'weights/")
 if n=='evaluate.py':s=s.replace("evaluator_sha256=sha(BASE/'evaluate.py')","evaluator_sha256=sha(ROOT/'evaluate.py')")
 if n=='publish_github.py':s=s.replace('v6.26','v6.27').replace('v626_','v627_').replace('V626_PUBLICATION_PHASE','V627_PUBLICATION_PHASE').replace('MS-ILLM decoder control experiment results','MS-ILLM effectiveness and MCL-JCV audit results')
 (R/n).write_text(s)
for n in ['train_manifest_ulong.json','train_manifest_uvg.json','qp_semantics_audit.json']:shutil.copyfile(V/n,R/n)
shutil.copyfile(V/'plans/shared.json',R/'plans/shared.json')
c=json.loads((V/'config.json').read_text());c.update(schema='v627',seed=20261027,no_training=True,updates=0,expected_points=1500,diagnostic_checkpoints=[0,250,500,1000],gradient_zero_threshold=1e-12)
(R/'config.json').write_text(json.dumps(c,indent=2)+'\n')
e=json.loads((V/'evaluation/config.json').read_text());(R/'evaluation/config.json').write_text(json.dumps(e,indent=2)+'\n')
p=dict(schema='v627',no_training=True,optimizer_steps=0,seed=20261027,methods=c['methods'],diagnostic_checkpoints=[0,250,500,1000],seen_requested=40,unseen_requested=20,unseen_definition='not used in v626 plan; not historically never seen',frames=64,target_positions=list(range(49,64)),condition='actual bridge input pixel_unshuffle decoded feature 2; detached; same real/fake',labels='GT VQ index0..1023; real target y+1; fake0',blur=dict(kernel_size=5,sigma=1.0,padding='reflect'),condition_shuffle='same domain and externalQP, next distinct window, cyclic sorted panel ID',gradient=dict(rho=.05,eps=1e-12,unit_scale=4096,parameters=['bridge','generator'],no_optimizer_step=True,simulate_global_clip=1.),MCL=dict(frames=list(range(64)),temporal_stride=1,IP=-1,external_qps=list(range(10)),original_size=[1920,1080],padding=[1920,1088],color=dict(matrix='BT709',source_range='limited',rgb_range='full',source_range_metadata='raw YUV untagged; fixed historical conversion convention, not inferred intrinsic metadata',tool='ffmpeg scale'),rate='complete bytes*8/(64*W*H); sourcefps and display20fps'),equal_rate=dict(methods=5,grid=100,spacing='uniform ln(bpp)',interpolation='PCHIP',extrapolation=False,components='shared linear ln(bpp) weights; diagnostic interpolation'),limitations=['unseen is plan-level only','unknown official pretraining exposure cannot be excluded','raw YUV has no embedded color metadata; historical BT709 limited-to-full fixed and disclosed'],no_checkpoint_selection=True)
(R/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
(R/'.gitignore').write_text('cache/\nlabel_cache/\nbitstreams/\nfeatures/\ndecoded/\n*.pt\n*.npz\n*.npy\n*.lock\n__pycache__/\n')
(R/'pipeline_status.json').write_text(json.dumps(dict(status='PREPARING',no_training=True,completed_RD=0,expected_RD=1500,active={},blocked={}),indent=2)+'\n')
(R/'RUNBOOK.md').write_text(f'''# V6.27\n\nNo optimizer updates. Historical experiments are read only.\n\nStart or resume:\n```bash\ncd {R.parents[1]}\nnohup /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u {R}/run_pipeline.py > {R}/logs/pipeline.log 2>&1 < /dev/null &\n```\nStatus: read `pipeline_status.json`; tail `logs/pipeline.log`. The scheduler uses flock and validates PID start times.\n''')
print(R)
