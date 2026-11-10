"""Freeze source identities, corrected protocol and the first400 logical samples."""
from common_s2 import *
from collections import Counter

def main():
 env()
 if (ROOT/'protocol.json').exists():return
 assert load(G_SOURCE/'status.json')['status']=='PAUSED','OLD_G_EXPERIMENT_MUST_REMAIN_PAUSED'
 p=load(PARENT_EXPERIMENT/'protocol.json');ef=load(EF_SOURCE/'protocol.json');expected={B300:ef['B300']['SHA256'],INIT:p['parent_checkpoint']['sha256'],PILOT:p['original_codec_and_receiver']['pilot_SHA256'],BASELINE:p['original_codec_and_receiver']['B1500_SHA256'],RAFT:RAFT_SHA}
 for n in ['I','P']:expected[PROJECT/f'checkpoints/GVC-RT_{n}.pt']=p['original_codec_and_receiver'][f'original_{n}_SHA256']
 for path,h in expected.items():assert sha(path)==h,('SOURCE_IDENTITY',str(path))
 assert p['loss']['LPIPS']==1 and p['loss']['DISTS']==.5 and p['loss']['RGB_L1']==.1383589248954108 and p['loss']['RAFT_motion']==.05
 audit=load(PARENT_EXPERIMENT/'gradient_audit.json');assert audit['calibration']['lambda_temp']==LAMBDA_TEMP==2.603297563918803
 efcode=load(EF_SOURCE/'source_manifest.json')['code'];assert sha(EF_SOURCE/'experiment_model.py')==efcode['experiment_model.py']==sha(ROOT/'b300_loader.py')
 for n,h in efcode.items():
  if n.startswith(('runtime_tree/','reference_tree/','vendor/')):assert sha(EF_SOURCE/n)==h==sha(ROOT/n);expected[EF_SOURCE/n]=h
 for src in [PARENT_EXPERIMENT,EF_SOURCE,G_SOURCE]:
  for n in ['protocol.json','source_manifest.json','initialization_audit.json','gradient_audit.json']:
   if (src/n).exists():expected[src/n]=sha(src/n)
 for n in ['gvc_rt1110.md','gvc_rt1110_review_20261110.md']:expected[PROJECT/'experiments'/n]=sha(PROJECT/'experiments'/n)
 idx=load(PARENT_EXPERIMENT/'plans/index.json');rows=[]
 for entry in idx['shards']:
  path=PARENT_EXPERIMENT/entry['file'];assert sha(path)==entry['sha256'];rows.extend(load(path))
 rows=rows[:400];assert len(rows)==400 and len({r['sample_key'] for r in rows})==400 and len({r['source_group'] for r in rows})==400
 pool=load(ROOT/'manifests/effective_ULong_pool.json');allowed={v['source_group'] for v in pool};held={v.get('source_group') for v in load(ROOT/'manifests/test_videos.json')}|{v['source_group'] for v in load(ROOT/'manifests/development_videos.json')}
 assert {r['source_group'] for r in rows}<=allowed and {r['source_group'] for r in rows}.isdisjoint(held) and all(r['domain']=='ulong' for r in rows)
 dump(ROOT/'plans/train.json.gz',rows);dump(ROOT/'plans/index.json',dict(shards=[dict(file='plans/train.json.gz',sha256=sha(ROOT/'plans/train.json.gz'))],windows=400,source_index=str(PARENT_EXPERIMENT/'plans/index.json'),source_index_SHA256=sha(PARENT_EXPERIMENT/'plans/index.json')))
 calibration=[]
 for q in [0,5,9]:calibration.extend([r for r in rows if r['external_qp']==q][:3])
 calibration.append(next(r for r in rows if r['sample_key'] not in {x['sample_key'] for x in calibration}));calibration.sort(key=lambda r:r['plan_slot']);assert len(calibration)==10
 dump(ROOT/'manifests/calibration_windows.json',calibration)
 rng=random.Random(202611101);directions=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
 dump(ROOT/'plans/auxiliary_shifts.json',{r['sample_key']:{str(i):list(rng.choice(directions)) for i in [52,56,60,63]} for r in rows})
 dump(ROOT/'source_inputs.json',dict(status='VERIFIED',base_commit=SOURCE_COMMIT,readonly={str(k):v for k,v in expected.items()},B300_expected_from=str(EF_SOURCE/'protocol.json'),dependency_paths_from_frozen_protocols=True,old_G_remains_PAUSED=True))
 dump(ROOT/'audits/data_isolation.json',dict(status='PASS',windows=400,unique_source_groups=400,QP_counts=dict(Counter(r['external_qp'] for r in rows)),inherited_plan_prefix=True,no_rescan_or_resplit=True,calibration_QP_counts=dict(Counter(r['external_qp'] for r in calibration)),development_and_test_excluded=True))
 protocol=dict(task='gvcrt_tcg_flicker_h_series_b300_20261110',status='PRE_CALIBRATION',base_commit=SOURCE_COMMIT,branch=BRANCH,B300=dict(path=str(B300),SHA256=sha(B300),overlay_last_keys=52),sources='source_inputs.json',groups={GROUPS[0]:dict(GPU=0,latent=False,equiv=False),GROUPS[1]:dict(GPU=1,latent=True,equiv=False),GROUPS[2]:dict(GPU=2,latent=True,equiv=True,requires_mechanical_gate=True)},trainables=dict(Bridge=801124,TCG=137760,LoRA=59392,total=998276),s_train=.5,s_eval=.5,training=dict(max_updates=100,effective_batch=4,single_GPU_sequential_windows=4,seed=20261110,frames=64,crop=[256,256],warmup=[1,48],supervised=[49,63],segments=[[49,52],[53,56],[57,60],[61,63]],save_steps=[0,25,50,75,100],optimizer='fresh AdamW',learning_rates=dict(Bridge=2e-6,TCG=1e-5,LoRA=5e-5),betas=[.9,.999],eps=1e-8,weight_decay=1e-4,global_clip=1.,computation='FP32 reconstruction master/compute; inherited frozen codec FP16'),base_loss=dict(LPIPS_Alex=1.,DISTS=.5,RGB_L1=.1383589248954108,RAFT_motion=.05,output_temporal=LAMBDA_TEMP,Sobel_MC_factor=.5,LPIPS_VGG_training=False),new_losses=dict(lambda_lat=None,lambda_equiv=None,tau=.02,RMS_eps=1e-8,RMS_min=1e-4,RMS_gradient=True,pair_shared_RMS=True,denominator='hard_valid_area; valid-framepairs/window; nonempty-windows/globalbatch once',equiv_calls=[52,56,60,63],equiv_window_divisor=4,equiv_crop_RGB=64,equiv_both_sides='use_tcg=False update_buffer=False; target no_grad; current model/QP/precision',no_edge_floor=True,no_Teacher=True),calibration=dict(windows=10,QP_counts=dict(Counter(r['external_qp'] for r in calibration)),rule='.1*median(norm_Bridge spatial/norm_Bridge new)',optimizer_updates=0,auxiliary_table_independent_rng=True),H2_gate=dict(pre_registered_before_measurement=True,windows=10,frames=[52,56,60,63],feature_shift_units=[0,1,2],crop_RGB=[16,32,64],padding='unchanged deployment padding',pass_rule='delta0 exact0 and median(error16px/error8px at crop64) <0.1; a control within one order of magnitude or larger fails',ratio_threshold=.1,failure_action='H2 MECHANICAL_CHECK_FAILED; no H2 training; H0/H1 continue'),development=dict(videos=32,QP=[0,5,9],points=96,steps=[50,100],baseline='B300 s=.5',weights='equal video within QP',training_early_stop='only numerical/implementation fault; complete100 otherwise'),screen=dict(per_QP_max_ratio=1.005,guard_metrics=['DISTS','LPIPS_Alex','LPIPS_VGG','FloLPIPS'],score='mean_q(mean_video PtoP Sobel(model)/mean_video PtoP Sobel(B300))',S_strictly_less_than=1,allowed_steps=[50,100],tie_delta=.0005,tie_prefers='earlier',no_qualified='NOT_QUALIFIED; no automatic final test; no step0 substitution'),final=dict(datasets=load(G_SOURCE/'protocol.json')['final']['datasets'],QP=[0,5,9],points_per_state=225,only_if_any_numerically_qualified=True,states=['Original','A2000','B300','qualified_candidates'],no_cross_dataset_pooling=True,no_BD_rate=True),metrics=dict(RGB_range=[0,1],LPIPS_normalize=True,LPIPS_version='0.1',spatial_frames=64,temporal='I_to_P and P_to_P separate',flow='frozen RAFT estimated on GT RGB; not known ground-truth motion',sampler='FP32 pixel-center, align_corners=False, zero padding, FB consistency',edge='normalized /8 Sobel components with replicate boundaries; local16x16 energy bias diagnostic only'),deployment_precision=DEPLOYMENT,only_GPUs=[0,1,2,3],reporting='execution facts and numbers only; H0 combines wider Bridge boundary and continuation',time=time.time())
 dump(ROOT/'protocol.json',protocol);dump(ROOT/'scheduler_config.json',dict(training_prefetch_windows=24,startup_windows=4,cache_GPUs=[3],max_cache_workers=1,derived_byte_cap=60*2**30,media_byte_cap=24*2**30,min_free_bytes=80*2**30,max_native_ready=5))
 dump(ROOT/'status.json',dict(status='PREPARING',complete=False,current_stage='mechanical_preflight',time=time.time()))
 print('H_PROTOCOL_SOURCE_AND_PLAN_PASS',400,10)
if __name__=='__main__':main()
