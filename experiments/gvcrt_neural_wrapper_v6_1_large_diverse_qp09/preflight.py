"""Record actual V4.1 implementation; no old files are changed."""
import ast,zipfile
from v61_io import *
def main():
    for sub in ('logs','parts','checkpoints','profiling','training_videos','validation_videos','bitstreams','features'):(ROOT/sub).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'parts/train_done.json').exists()
    selected=load(V41/'selected_checkpoint.json');cfg=load(V41/'config.json');path=Path(selected['checkpoint'])
    assert selected['status']=='PASS' and selected['step']==20000 and sha(path)==selected['checkpoint_sha256']
    import torch
    torch.set_num_threads(4);cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['step']==20000 and cp['beta']==cfg['beta'] and cp['optimizer']['state']
    counts={k:sum(v.numel() for v in cp[k].values()) for k in ('wrapper','bridge','generator')}
    hooks=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py'
    core=ROOT.parent/'gvcrt_neural_wrapper_v2_joint/core.py'
    assert 'force_zero_thres=0.12' in hooks.read_text() and 'sigma > .12' in core.read_text()
    training=(V41/'train_extension.py').read_text();assert 'qp=rng.randrange(4)' in training and 'joint_ste_forward(p_model,proxy,qp)' in training
    assert "loss=(distortion+cfg['beta']*rate_bpp+cfg['lambda_proxy']*l1)/3" in training
    shift=REPO/'src/models/video_model_gvcrt.py';assert 'qp_shift = [0, 2, 1]' in shift.read_text() and 'return qp + self.qp_shift[fa_idx]' in shift.read_text()
    hashes={}
    for directory in (V3,V4,V41,V52,OLD_ENGINE):
        for pattern in ('*.py','*.json'):
            for p in directory.glob(pattern):
                if p.name not in ('pipeline_status.json',):hashes[str(p)]=sha(p)
    hashes.update(load(V52/'config.json')['source_hashes']);hashes.update(load(V52/'config.json')['additional_source_hashes'])
    hashes.update(load(V52/'metric_implementation_audit.json')['source_and_weight_sha256'])
    hashes.update(load(V52/'checkpoint_audit.json')['original_base_checkpoints'])
    for p in (path,SPLIT,hooks,core,shift,ROOT.parent/'gvcrt_neural_wrapper_v2_joint/wrapper_model.py',V720/'config.json',V720/'source_integrity.json'):
        hashes[str(p)]=sha(p)
    dump(ROOT/'old_source_hashes.json',hashes)
    old=load(V41/'train_manifest.json')['videos'];val=load(V41/'validation_manifest.json')['videos'];tests=load(V41/'test_manifest.json')['videos']
    assert len(old)==256 and len(tests)==8
    split=load(SPLIT)['splits'];reserved={r['file_sha256'] for k in ('validation','confirmation') for r in split[k]}
    reserved.update(r['source_sha256'] for r in tests);assert not ({r['sha256'] for r in old}&reserved)
    candidates=[]
    for archive in sorted(ARCHIVES.glob('clips_long_1920_*.zip')):
        with zipfile.ZipFile(archive) as z:
            for i in z.infolist():
                if i.filename.lower().endswith('.mp4'):candidates.append(dict(archive=str(archive),filename=i.filename,crc32=f'{i.CRC:08x}',bytes=i.file_size))
    assert len(candidates)>=1024
    dump(ROOT/'candidate_manifest.json',dict(candidate_count=len(candidates),total_bytes=sum(r['bytes'] for r in candidates),videos=candidates))
    qp=[]
    for q in range(10):qp.append(dict(external_qp=q,actual_i_qp=q,training_p_qps=[q]*3,evaluation_first_16_p_qps=[q+[0,2,1][[0,1,0,2,0,2,0,2][i%8]] for i in range(1,17)],clamp='none; P has two extra QP slots (max actual QP=11)'))
    dump(ROOT/'qp_semantics_audit.json',dict(status='PREFLIGHT_PASS',rows=qp,source_training=str(V41/'train_extension.py'),source_evaluation=str(V4/'eval_core.py'),
        training_rule='I external QP, each of three STE P frames uses the same external QP directly; preserve V4.1, no new training shift',
        evaluation_rule='shift_qp(external_qp, INDEX_MAP[frame_index % 8]); offset=[0,2,1]; no clamp',per_qp_sampling_counts={str(q):0 for q in range(10)},uniform_probability=.1))
    config=dict(schema='gvcrt_v6_1_large_diverse_qp09',seed=20260928,gpus=[4,5,6,7],training_gpu=4,validation_gpus=[5,6,7],
        multi_gpu_mode='Existing process-per-GPU orchestration: one training process, independent validation/evaluation workers; no DDP or optimizer changes',
        source_checkpoint=str(path),source_checkpoint_sha256=sha(path),source_checkpoint_step=20000,additional_updates=40000,effective_final_step=60000,
        checkpoint_additional_updates=[0,5000,10000,20000,30000,40000],qps=list(range(10)),force_zero_thres=.12,
        clip_length=cfg['clip_length'],crop=cfg['crop'],beta=cfg['beta'],lambda_proxy=cfg['lambda_proxy'],optimizer=cfg['optimizer'],
        restore_optimizer=True,scheduler=cfg['scheduler'],per_gpu_batch=1,detach_dpb_every_frame=True,
        validation_count=32,validation_frames=32,validation_qps=[0,2,4,6,8,9],selection_datasets=['validation_ulong'],
        selection_rule='Among checkpoints passing LPIPS/DISTS equal-rate-negative gates and nonpositive valid BD-rate gates, maximize mean same-QP bitrate reduction on validation; earliest additional step breaks ties. If none pass, rank validation-only normalized mean equal-rate LPIPS/DISTS delta, earliest tie; record fallback.',
        final_datasets=['ulong','uvg','virat720'],optional_virat480='Not scheduled; optional scope deferred',
        old_train_count=256,train_count=1024,profiling=dict(workers=8,uniform_frame_pairs=4,statistics_canvas=[256,144],training_preprocessing_unchanged=True),
        stratification=dict(temporal_bins=4,texture_bins=4,temporal='temporal_rgb_L1',texture='sobel_edge_energy'),
        entropy_metrics_note='Training retains existing differentiable rate loss; all validation/final reported rates use full real RANS bitstreams only')
    assert cfg['clip_length']==4 and cfg['crop']==[256,256]
    dump(ROOT/'config.json',config)
    dump(ROOT/'preflight_audit.json',dict(status='PASS',main_checkpoint=str(path),main_checkpoint_sha256=sha(path),source_step=cp['step'],
        architectures=dict(P='Unchanged NeuralWrapper: 64 channels, 5 two-convolution residual blocks, residual RGB clamp',Bridge='Unchanged recon_generation_net.mlp',Generator='Unchanged recon_generation_net.decoder'),
        module_state_counts=counts,module_hashes={k:tensor_hash(cp[k]) for k in counts},trainable=['wrapper','bridge','generator'],
        frozen=['I codec','P compression core excluding recon_generation_net.mlp and decoder','quality metric networks'],optimizer=cfg['optimizer'],
        beta=cfg['beta'],lambda_proxy=cfg['lambda_proxy'],clip_length=4,crop=[256,256],old_external_qps=[0,1,2,3],new_external_qps=list(range(10)),
        force_zero_thres_training=.12,force_zero_thres_evaluation=.12,train_manifest=str(V41/'train_manifest.json'),
        validation_manifest=str(V41/'validation_manifest.json'),test_manifest=str(V41/'test_manifest.json'),
        real_rans_path=str(V4/'eval_core.py'),metric_path=str(V41/'metric_runtime.py'),
        deviations_from_assumptions=['Existing multi-GPU mode is independent processes, not distributed gradient training','Training STE uses direct external QP for P frames; evaluation applies internal shift_qp'],
        source_sampling='Unchanged V4.sample_clip; 4 consecutive original RGB frames, random 256 crop, batch1',
        losses='(LPIPS + DISTS + beta*rate_bpp + lambda_proxy*proxy_L1)/3 per P frame, sum across three P frames; DPB detached each frame',
        candidate_count=len(candidates),reserved_validation_confirmation_count=len(reserved),old_artifact_hash_count=len(hashes),no_old_file_writes=True))
    frozen();status(status='PREFLIGHT_PASS',phase='profiling_pending')
    print('PREFLIGHT PASS',str(path),len(candidates),flush=True)
if __name__=='__main__':main()
