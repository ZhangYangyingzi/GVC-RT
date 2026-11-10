"""Metadata-only plan, identities and preregistered protocol; no test quality read."""
from common_s2 import *
from collections import Counter
from fractions import Fraction

def dataset(v):
    return {'ulong_new':'ULong_new_holdout_32','ulong':'ULong_historical_6','hevc_b':'HEVC_B_5','uvg_holdout':'UVG_historical_HoneyBee_Jockey_2','mcljcv':'MCL_JCV_30'}[v['domain']]

def main():
    env()
    if (ROOT/'protocol.json').exists():return
    efp=load(EF_SOURCE/'protocol.json');assert sha(B300)==efp['B300']['SHA256']
    origin=load(REFERENCE_EXPERIMENT/'protocol.json')['original_codec_and_receiver']
    expected={B300:B300_SHA,PILOT:origin['pilot_SHA256'],BASELINE:origin['B1500_SHA256'],INIT:load(REFERENCE_EXPERIMENT/'protocol.json')['parent_checkpoint']['sha256'],TEACHER_CHECKPOINT:TEACHER_SHA}
    for n in ['I','P']:expected[PROJECT/f'checkpoints/GVC-RT_{n}.pt']=origin[f'original_{n}_SHA256']
    for p,h in expected.items():assert sha(p)==h,('SOURCE_SHA256',str(p))
    old=load(EF_SOURCE/'source_manifest.json')
    for n in ['common_s2.py','model_context.py','experiment_model.py','temporal.py','train.py','cache_codec.py']:
        assert sha(EF_SOURCE/n)==old['code'][n],('EF_FROZEN_SOURCE',n)
    pool=load(ROOT/'manifests/effective_ULong_pool.json');poolgroups={r['source_group'] for r in pool}
    assert sha(ROOT/'manifests/effective_ULong_pool.json')==sha(PARENT_EXPERIMENT/'manifests/effective_ULong_pool.json')
    rows=plan_rows();assert sha(ROOT/'plans/index.json')==sha(PARENT_EXPERIMENT/'plans/index.json')
    dev_rows=[r for r in load(ROOT/'manifests/evaluation_rows.json') if r['split']=='development'];assert len(dev_rows)==32
    dev=[]
    from inherited_reader import expected_native_shape
    for r in dev_rows:
        h,w,_=expected_native_shape(r)
        dev.append(dict(domain='ulong_new',dataset='ULong_development_32',sequence=Path(r['video']).stem,split='development',source_path=r['source_path'],source_sha256=r['source_sha256'],source_frame_indices=r['source_frame_indices'],height=h,width=w,frame_rgb_sha256=r['full_RGB_sha256'],rgb_sha256=r['rgb_sha256'],source_group=r['source_group'],fps=float(Fraction(r['source_fps'])),source_row=r))
    final=load(ROOT/'manifests/final_videos.json')+load(ROOT/'manifests/MCL_JCV_30.json');assert len(final)==75
    for v in final:v['dataset']=dataset(v)
    assert Counter(v['dataset'] for v in final)==dict(ULong_new_holdout_32=32,ULong_historical_6=6,HEVC_B_5=5,UVG_historical_HoneyBee_Jockey_2=2,MCL_JCV_30=30)
    held={r['source_group'] for r in dev_rows}|{r.get('source_group') for r in final}
    assert {r['source_group'] for r in rows}.issubset(poolgroups) and {r['source_group'] for r in rows}.isdisjoint(held)
    assert len({r['source_group'] for r in rows})==1200 and Counter(r['external_qp'] for r in rows)=={q:120 for q in range(10)}
    dump(ROOT/'manifests/development_videos.json',dev);dump(ROOT/'manifests/test_videos.json',final)
    dump(ROOT/'manifests/preflight_windows.json',rows[:8])
    dump(ROOT/'audits/source_isolation.json',dict(status='PASS',inherited_audit=str(PARENT_EXPERIMENT/'audits/source_isolation.json'),inherited_audit_SHA256=sha(PARENT_EXPERIMENT/'audits/source_isolation.json'),pool_SHA256=sha(ROOT/'manifests/effective_ULong_pool.json'),source_plan=str(PARENT_EXPERIMENT/'plans/index.json'),source_plan_SHA256=sha(ROOT/'plans/index.json'),unique_training_source_groups=1200,development32_excluded=True,heldout_metadata_excluded=True,training_domains=['ULong'],historical_exposure_records_preserved=True,no_original_dataset_rescan=True))
    sources={str(p):h for p,h in expected.items()}
    for src in [PARENT_EXPERIMENT,EF_SOURCE,REFERENCE_EXPERIMENT,TEACHER_SOURCE,DIAGNOSIS_SOURCE]:
        for n in ['protocol.json','source_manifest.json','final_integrity.json','teacher.py','objective.py','config.json','audits/teacher_gate.json']:
            p=src/n
            if p.exists():sources[str(p)]=sha(p)
    dump(ROOT/'source_inputs.json',dict(status='VERIFIED',base_commit=SOURCE_COMMIT,readonly=sources,B300_expected_from=str(EF_SOURCE/'protocol.json'),Teacher_expected_from=str(TEACHER_SOURCE/'audits/teacher_gate.json'),original_protocol=str(ORIGINAL_EVAL/'protocol.json')))
    cfg=dict(task='gvcrt_tcg_teacher_temporal_nolora_ab_20261110',status='PRE_CALIBRATION',branch=BRANCH,base_commit=SOURCE_COMMIT,groups={GROUPS[0]:dict(GPUs=[0,1],latent_temporal=False),GROUPS[1]:dict(GPUs=[2,3],latent_temporal=True)},B300=dict(path=str(B300),SHA256=B300_SHA),initialization='complete original/pilot+buffers/B1500/A2000 chain, B300 trainable overlay LAST, physically remove five LoRA wrappers without merging; preserve Bridge/TCG and base groups/dispatch',historically_LoRA_trained_Bridge_TCG=True,
        teacher=dict(candidate='pretrain262144',EMA=True,checkpoint_SHA256=TEACHER_SHA,output='official continuous18 channels before LFQ sign',normalization='none extra',frozen=True,train_only=True,historical_exact_teacher_identity_claim=False,old_coordinate_audit_does_not_prove_temporal_stability=True),
        training=dict(seed=PLAN_SEED,global_batch=4,ranks=2,windows_per_rank=2,max_additional_updates=300,frames=64,crop=[256,256],warmup=[1,48],supervised=[49,63],segments=[[49,52],[53,56],[57,60],[61,63]],fresh_optimizer='AdamW',learning_rates=dict(Bridge=2e-6,TCG=1e-5),betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip_norm=1.,master_compute='FP32',codec='inherited FP16',trainables=dict(Bridge=801124,TCG=137760),plan_SHA256=sha(ROOT/'plans/index.json'),checkpoints=[0,50,100,150,200,250,300]),
        losses=dict(LPIPS_VGG=1.,DISTS=.5,RGB_L1=.1383589248954108,cosine=.2,cosine_dim=1,cosine_eps=1e-8,RAFT_motion=.05,output_temporal='lambda_out*(RGB_MCResidual+.5*Sobel_MCResidual)',latent_temporal='G1 only: masked mean abs((S_t-warp(S_prev))-(T_t-warp(T_prev))); previous student latent differentiable within segments',lambda_out=None,lambda_latent=None,calibration_windows=[r['sample_key'] for r in rows[:8]],calibration_rule='0.1*median(unclipped gradient norm ratios); content includes cosine',margin_implemented=False,extra_Sobel_detail=False),
        validation=dict(videos=32,points_per_group_checkpoint=96,QP=[0,5,9],steps=[0,50,100,200,300],shared_step0_may_be_measured_once_and_identity_reused=True),selection=dict(score='mean_q(mean_video_Sobel(model,q)/mean_video_Sobel(noLoRA_step0,q))',qualified='S<1, mean_q temporal ratio to B300<1; every QP mean DISTS and LPIPS-VGG <=1.01*each reference',min_delta=.0005,patience=2,joint_stop='both groups satisfy patience at same validated update; faults/missing points never count',initial_anchor=1.,tie='earlier checkpoint',fallback='lowest S trained checkpoint; NOT_QUALIFIED; never step0'),
        final=dict(videos=75,points_per_state=225,states=['Original','B300_original_reference','B300_noLoRA_step0','G0_selected','G1_selected'],expected_points=1125,QP=[0,5,9],frames=64,datasets=dict(Counter(v['dataset'] for v in final)),selection_frozen_before_test=True,metric_names=['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','MCResidual_RGB_PtoP','MCResidual_Sobel_PtoP'],no_BD_rate=True,no_extrapolation=True),
        metrics=dict(RGB_range=[0,1],LPIPS_normalize=True,LPIPS_version='0.1',spatial_frames=64,temporal='I_to_P and P_to_P separate',RAFT_GT_flow='same frozen RAFT and FB-consistency/pixel-center align_cornersFalse as E/F',weighting='equal videos within dataset/split/QP; no cross-dataset pooled curve'),deployment_precision=DEPLOYMENT,only_GPUs=[0,1,2,3],time=time.time())
    dump(ROOT/'protocol.json',cfg)
    dump(ROOT/'scheduler_config.json',dict(training_prefetch_windows=16,startup_windows=8,cache_GPUs=[0],max_cache_workers=1,derived_byte_cap=90*2**30,media_byte_cap=30*2**30,min_free_bytes=100*2**30,max_native_ready=8,source_cache_reuse=True,heartbeat_seconds=10,readable_status_seconds=60))
    dump(ROOT/'status.json',dict(status='PREPARING',complete=False,current_stage='preflight',groups={g:dict(committed_updates=0) for g in GROUPS},time=time.time()))
    print('PLAN_SOURCE_POOL_AND_PROTOCOL_VERIFIED',len(rows),len(dev),len(final))
if __name__=='__main__':main()
