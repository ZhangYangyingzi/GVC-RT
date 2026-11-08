"""Freeze the user-specified continuation and all input/code/plan identities."""
from common_s2 import *

def main():
    env();assert not (ROOT/'source_manifest.json').exists(),'ALREADY_FROZEN_RESUME_RUNNER'
    parent=load(SOURCE/'protocol.json');old=load(SOURCE/'source_manifest.json');gain=load(GAIN_SOURCE/'sources.json')
    assert sha(INIT)==INIT_SHA and sha(HIST/'manifests/B_expanded_ulong.json')=='0526eace3882402792105fae5cff7817b624f0e5b0ff49ddd6cce03829d1de63'
    assert sha(ROOT/'loss.py')==old['code']['loss.py'] and sha(ROOT/'convlora.py')==sha(GAIN_SOURCE/'convlora.py')
    isolation=load(ROOT/'audits/source_isolation.json');assert isolation['status']=='PASS'
    protocol=dict(task='ULong_only_LoRA_A2000_continuation_20261108',revision='ULONG_group_temporal_bins_v1',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
        branch='codex/tcg-lora-ulong-only-a2000-20261108',parent_checkpoint=dict(path=str(INIT),sha256=INIT_SHA,LoRA_absolute_optimizer_step=2000,TCG_cumulative_step=6500),
        original_codec_and_receiver=dict(pilot=str(PILOT),pilot_SHA256=PILOT_SHA,B1500_TCG=str(BASELINE),B1500_SHA256=BASELINE_SHA,
            original_I_SHA256=expected_source_sha(PROJECT/'checkpoints/GVC-RT_I.pt'),original_P_SHA256=expected_source_sha(PROJECT/'checkpoints/GVC-RT_P.pt')),
        trainable=dict(names=parent['groups']['A_control']['targets'],parameter_count=59392,only_LoRA=True,TCG_Bridge_codec_original_generator_and_quality_frozen=True,Router=False,
            rank=4,intrinsic_alpha=4,intrinsic_alpha_div_rank=1,s_train=1.,s_eval=.5,LoRA_internal_and_gain_and_add='FP32, cast to incoming base-output dtype once'),
        optimizer=dict(parent['optimizer'],new=False,restored_from_A2000=True,absolute_initial_step=2000,additional_initial_step=0),
        training=parent['training'],precision=dict(codec='FP16',training_Bridge_generator='FP32',deployment_Bridge_generator='FP16',TCG_LoRA_internal='FP32',RGB='matched_rgb_fp32_v2'),
        data=dict(effective_ULong_pool=isolation['effective_videos'],effective_source_groups=isolation['effective_groups'],existing_quarantined_groups=236,MCL_reserved_sources_checked=30,
            UVG_training_samples=0,Vimeo_training_samples=0,historically_UVG_exposed_parent=True,lifelong_ULong_only_claim=False,seed=PLAN_SEED,plan_index_SHA256=sha(ROOT/'plans/index.json'),
            new_plan_cursor_start=0,old_mixed_plan_cursor_reused=False,logical_samples=12000,global_batch=4,groups_balanced=True,four_distinct_groups_each_batch=True,
            temporal_bins='up to4 nonempty intervals over starts0..frames-64;fixed per-video shuffled bin order,rotate visits,random unseen start',
            geometry='native continuous64frames,one256crop across sequence,no resize/repeat/pad/stitch',RGB_window_hashes='verified on actual read and stored separately,never borrowed from old windows'),
        RNG_migration=dict(parent_physical_world_size=2,new_physical_world_size=4,parent_logical_streams=4,new_logical_streams=4,one_logical_sample_per_rank=True,plan_RNG_independent=True),
        gradient=dict(implementation='inherited manual VJP and SUM all_reduce;divide4 once,clip1 once,optimizer.step once per identical replica',DDP_automatic_reduction=False,
            tolerance_relative_L2=1e-3,loss_absolute_tolerance=1e-5,VJP4096_unscale_once=True),
        runtime=dict(GPU_physical=[4,5,6,7],CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7',Torch_TF32_precision_flags='inherit unchanged',
            GT_flow_selection='Disposable same-initial-state benchmark: four-rank inline versus audited frozen GT-RAFT helper;choose lower two-step processing time excluding startup',
            runtime_choice_file='audits/runtime_selection.json',four_GPUs_one_model=True,effective_batch=4),
        stopping=dict(max_additional_updates=3000,validation_interval=250,development_points=96,development_videos=32,external_QPs=[0,5,9],reference='fixed A2000 s_eval0.5',
            Fq='mean32video FloLPIPS(candidate,q)/mean32video FloLPIPS(A2000_s05,q)',Dq='mean32video DISTS(candidate,q)/mean32video DISTS(A2000_s05,q)',
            S='(F0+F5+F9)/3',qualified='all Dq<=1 and S<1, strict comparison;no tolerance added',anchor_initial=1.,min_delta=.0005,patience=4,
            anchor_update='qualified and S<anchor-0.0005; otherwise increment bad checks;invalid engineering/evaluation is failure,not a bad check',
            final_selection='lowest S among qualified checkpoints,ties earlier;retain A2000_s05 if none',raw_best_unconstrained_recorded=True,minimum_updates=None,no_test_selection=True),
        evaluation=dict(existing_final_videos=45,new_MCL_videos=30,total_final_videos=75,QP=[0,5,9],points_per_unique_model=225,
            states=['Original GVC-RT','TCG B+1500','TCG+LoRA A2000 s0.5','selected ULong-only continuation s0.5 if distinct'],deduplicate_A2000_retained_state=True,
            MCL_frames=list(range(64)),MCL_native=[1920,1080],MCL_fps='source24/25/30 preserved',MCL_color='audited BT709 limited YUV420p8 -> full RGB',
            first_I='actual original I codec bitstream counted;GT_RGB_bypass prohibited',SPS_use_ada_i=0,SPS_ec_part=1,codec_reference_refresh='inherit unchanged',
            fresh_MCL_original_TCG_A2000_RANS_probe_before_shared_streams=True,MCL_old_quality_or_RGB_bypass_stream_reuse=False,
            Original_existing135='independent decode and rescore on verified real streams;no adapted receiver',TCG_existing135='audited reuse only',A2000_existing135='new s0.5 measurements,never reuse s1 quality',
            quality_metrics=parent['evaluation']['metrics'],PWC_sampler=parent['evaluation']['inherited_PWC_sampler'],
            PSNR='total pixel SSE/native pixels;not average frame PSNR',spatial_frames=64,temporal_transitions=63,weighting='equal videos within each dataset/split/QP',
            split_historical=True,MCL_blinded_new_test_claim=False,final_after_selection_frozen=True,rate_identity='same video/QP bytes,bpp,SHA and codec features exact',BD_rate=False),
        resource_limits=parent['resource_limits'],checkpoints='additional0,2,every250,stop/selected plus atomic latest;optimizer,all logical RNG,plan cursor,protocol/code hashes,receipt,gains',
        publication=dict(repository='ZhangYangyingzi/GVC-RT',branch='codex/tcg-lora-ulong-only-a2000-20261108',allowlist_only=True,no_weights_media_cache=True,force_push=False),time=time.time())
    protocol['training']=dict(protocol['training']);protocol['training']['smoke']='Disposable states only;formal starts again at unmodified A2000 with restored moments/RNG'
    dump(ROOT/'protocol.json',protocol)
    protected={}
    def add(p,h=None):
        p=Path(p);value=sha(p)
        if h is not None:assert value==h,('SOURCE_HASH_MISMATCH',str(p))
        protected[str(p)]=value
    add(INIT,INIT_SHA);add(PILOT,PILOT_SHA);add(BASELINE,BASELINE_SHA);add(RAFT,RAFT_SHA)
    for name,h in old['code'].items():add(SOURCE/name,h)
    for name,h in old['immutable_manifests'].items():add(SOURCE/name,h)
    for name,h in old['inherited_external_sources'].items():add(name,h)
    for name,h in gain['protected_source_files'].items():add(name,h)
    for name,h in gain['local_files'].items():add(GAIN_SOURCE/name,h)
    add(GAIN_SOURCE/'sources.json');add(GAIN_SOURCE/'protocol.json',gain['protocol_sha256']);add(GAIN_SOURCE/'metrics_s05_all_qps.csv')
    for name in ['README.md','audit.json','sampling_proposal.json']:add(REVIEW/name)
    for name in ['manifests/B_expanded_ulong.json','manifests/development.json','manifests/new_holdout_test.json','manifests/reserved_union.local.json',
                 'graph/nodes.json','graph/final_components.json','graph/final_identity_to_group.json','graph/quarantine.json','graph/screen_protocol.json','inventory/qualified.json']:add(HIST/name)
    add(PROJECT/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/eval_manifest.json')
    op=load(ORIGINAL_EVAL/'protocol.json');add(ORIGINAL_EVAL/'protocol.json')
    for name,h in op['original_source_sha256'].items():add(ORIGINAL/name,h)
    for name,h in op['checkpoint_sha256'].items():add(PROJECT/'checkpoints'/name,h)
    code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','logs'])}
    immutable={str(p.relative_to(ROOT)):sha(p) for name in ['manifests','vendor/sealed'] for p in (ROOT/name).rglob('*.json')}
    dump(ROOT/'source_manifest.json',dict(status='FROZEN',code=code,protected_sources=protected,immutable_manifests=immutable,
        protocol_sha256=sha(ROOT/'protocol.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),parent_A2000_SHA256=INIT_SHA,loss_SHA256=sha(ROOT/'loss.py'),time=time.time()))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',formal_training_updates=0,all_stages_complete=False))
    dump(ROOT/'status.json',dict(status='PREPARED',complete=False,current_stage='source_plan_protocol_frozen',additional_updates=0,absolute_LoRA_optimizer_step=2000,
        TCG_cumulative_step=6500,s_train=1.,s_eval=.5,experiment_directory=str(ROOT),time=time.time()))
    verify_sources();print('ULONG_CONTINUATION_PROTOCOL_AND_SOURCES_FROZEN')

if __name__=='__main__':main()
