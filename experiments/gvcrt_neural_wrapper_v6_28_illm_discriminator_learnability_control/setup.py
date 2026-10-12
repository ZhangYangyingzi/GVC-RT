from pathlib import Path
import hashlib,json,shutil,collections,os
R=Path(__file__).resolve().parent;V27=R.parent/'gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit';V26=R.parent/'gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):return json.loads(p.read_text())
assert not (R/'config.json').exists(),'Existing configuration must be verified, not overwritten'
for name in ('io21.py','illm_runtime.py','window_training.py','publish_github.py','restore_text_shards.py','index_io.py'):
 s=(V27/name).read_text()
 if name=='publish_github.py':s=s.replace('v6.27','v6.28').replace('v627','v628').replace('V627','V628').replace('MS-ILLM effectiveness and MCL-JCV audit results','fixed-B1 MS-ILLM discriminator learnability results')
 (R/name).write_text(s)
cfg=load(V27/'config.json');cp=cfg['source_checkpoint'];assert sha(cp['path'])=='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
idx=load(V26/'branches/C_decoder_illm/checkpoint_index.json')['0'];assert sha(idx['path'])==idx['sha256']
weight=V26/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt';assert sha(weight)=='3f58659bf2449b17f1f7eaf5ee64fd64d281da2f640d46148536ea6183eb9183'
(R/'plans').mkdir();shutil.copyfile(V26/'plans/shared.json',R/'plans/shared.json');assert sha(R/'plans/shared.json')=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
for name in ('train_manifest_ulong.json','train_manifest_uvg.json'):shutil.copyfile(V27/name,R/name)
shutil.copyfile(V27/'diagnostic_plan.json',R/'plans/validation.json');shutil.copytree(V27/'audits/diagnostic_plan_revisions',R/'audits/validation_plan_revisions')
rows=load(R/'plans/shared.json')['updates'];ct=collections.Counter((r['domain'],r['external_qp']) for r in rows);assert len(rows)==1000 and len(ct)==20 and set(ct.values())=={50}
tiny=[]
for dom in ['ulong','uvg']:
 for q in [0,9]:
  r=next(r for r in rows if r['domain']==dom and r['external_qp']==q);tiny.append(dict(row_id=f"main_{r['adaptation_step']:04d}",positions=[49,63],plan_step=r['adaptation_step'],domain=dom,external_qp=q))
dump(R/'plans/tiny.json',dict(windows=tiny,pairs=8,selection='earliest shared-plan window per domain and QP0/9; positions49/63 fixed before metrics'))
val=load(R/'plans/validation.json')['updates'];assert len(val)==60
for r in val:
 if r['panel']=='unseen':
  for t in rows:
   if t['source_path']==r['source_path']:assert not set(t['source_frame_indices']).intersection(r['source_frame_indices'])
branches={'A_raw_current':dict(condition='raw',optimizer='Adam',lr=2e-4,betas=[0,.9]),'B_rms_current':dict(condition='rms',optimizer='Adam',lr=2e-4,betas=[0,.9]),'C_zero_current':dict(condition='zero',optimizer='Adam',lr=2e-4,betas=[0,.9]),'D_raw_official_optimizer':dict(condition='raw',optimizer='AdamW',lr=4e-4,betas=[.5,.9])}
config=dict(schema='v628',seed=20261028,source_checkpoint=cp,D_source_checkpoint=idx,labeler_path=str(weight),labeler_sha256=sha(weight),branches=branches,stages={'TINY':{'updates':2000,'checkpoints':[0,100,250,500,1000,2000],'frames':8},'MAIN':{'updates':5000,'checkpoints':[0,250,1000,2500,5000],'frames':15}},allowed_gpus=[4,5,6,7],no_G_updates=True,no_new_codec_evaluation=True,fresh_D_optimizer=True,main_cycles=5,clip_length=64,targets=list(range(49,64)),crop=256,batch=1,force_zero_thres=.12,lambda_dists=.5,FP32_D=True,D_grad_clip=None,AMP=False,scheduler=None,reservation_margin_MiB=1024,reservation_multiplier=1.15,CPU_threads_per_worker=2)
dump(R/'config.json',config)
off=V26/'vendor/NeuralCompression-3f122808255a2278eea049119367eb3abc434862';opt=off/'projects/illm/conf/optimizer/model_adamw_disc_const.yaml'
assert 'lr: 0.0004' in opt.read_text() and 'betas: [0.5, 0.9]' in opt.read_text();shutil.copyfile(off/'LICENSE',R/'LICENSE');shutil.copyfile(opt,R/'audits/official_optimizer_source.txt')
protocol=dict(source='MS-ILLM local categorical discriminator; fixed GVC-RT/B1',official_commit='3f122808255a2278eea049119367eb3abc434862',official_optimizer={'path':str(opt),'sha256':sha(opt),'D_recipe':branches['D_raw_official_optimizer']},architecture=dict(output_channels=1025,output_downsampling_factor=8,context_in=1024,context_out=16,norm_type='identity'),D_loss='.5*mean(CE(D(GT,c), y+1))+.5*mean(CE(D(rec.detach,c),0)); whole-step frames averaged before one optimizer.step',B_rms='one detached sqrt(mean(c.float()**2)) over C/H/W per frame; divisor max(r,1e-6)',C_zero='zero condition during BOTH training and validation; same conditional architecture; adapter biases allowed',cache_forward='exact V27 fixed-B1 STE hard-forward; codec half; wrapper eval FP32; bridge/generator train FP32 but ALL parameters frozen/no_grad; native I0; causal48 warmup and15 targets; refs detached',validation='TINY all6 saved points:8 fit pairs +20 unseen windows; MAIN all5 saved points:40 seen+20 unseen windows; no validation-driven selection',unseen='no source-frame overlap with shared-plan windows; does not assert absent historical exposure',predictive_frequency_reference='training labels only; content Laplace alpha=1; balanced p_fake=.5; validation frequencies descriptive only',uniform_CE_1025=__import__('math').log(1025),shared_plan_sha256=sha(R/'plans/shared.json'),validation_plan_sha256=sha(R/'plans/validation.json'),tiny_plan_sha256=sha(R/'plans/tiny.json'),no_G_updates=True,no_new_RD_FID=True)
dump(R/'protocol.json',protocol)
# Verify relevant historical files before and after; inventory metadata also protects caches/checkpoints.
deps={str(p):sha(p) for folder in (V26,V27) for p in folder.rglob('*') if p.is_file() and (p.suffix=='.py' or p.name in ('config.json','protocol.json','checkpoint_index.json','final_integrity.json','shared.json','diagnostic_plan.json'))}
for p in [Path(cp['path']),Path(idx['path']),weight,opt]:deps[str(p)]=sha(p)
dump(R/'audits/dependencies.json',deps);dump(R/'audits/dependency_metadata.json',{p:dict(bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in deps})
dump(R/'audits/source_validation.json',dict(status='PASS',B1_SHA256=sha(cp['path']),D0_SHA256=sha(idx['path']),labeler_SHA256=sha(weight),shared_plan_SHA256=sha(R/'plans/shared.json'),domain_QP_counts={str(k):v for k,v in ct.items()},validation_seen=40,validation_unseen=20,unseen_shared_source_intersection=False,no_historical_writes=True))
dump(R/'pipeline_status.json',dict(status='READY',phase='preflight',no_G_updates=True,training={},validation={},cache={},publication='pending'))
(R/'.gitignore').write_text('cache/\nlabel_cache/\ncheckpoints/\n__pycache__/\n*.pt\n*.tmp\n*.lock\n*.pid\nweights/\n')
(R/'RUNBOOK.md').write_text('# V6.28 execution\n\nLaunch/resume: `'+config.get('python','/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python')+' '+str(R/'launch_pipeline.py')+'`\n\nStatus: `cat '+str(R/'pipeline_status.json')+'`\n\nTraining: fixed B1; D-only TINY2000 / MAIN5000, all four branches. No new G updates or RD tests.\n\nCompleted caches/checkpoints are hash-verified; owned live scheduler is preserved.\n')
print('SETUP PASS',R)
