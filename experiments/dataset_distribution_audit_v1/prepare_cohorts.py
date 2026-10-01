"""Freeze independent cohorts and protocols before reading reference statistics."""
import random
import traceback
from audit_io import *

def inventory_old():
    result={}
    for d in ROOT.parent.iterdir():
        if d.is_dir() and d.name.startswith(('gvcrt_neural_wrapper_v4','gvcrt_neural_wrapper_v5','gvcrt_neural_wrapper_v6')):
            for p in d.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':
                    s=p.stat();result[str(p)]=[s.st_size,s.st_mtime_ns]
    return result

def main():
    assert load(ROOT/'vimeo_inventory.json')['status']=='PASS'
    assert load(ROOT/'manifests/vimeo_train2048.json')['status']=='PASS'
    if (ROOT/'cohort_audit.json').exists():assert load(ROOT/'cohort_audit.json')['status']=='PASS';return
    dump(ROOT/'old_inventory_before.json',inventory_old())
    # Explicit user reply authorizes the seven-frame exception; no implicit padding.
    dump(ROOT/'vimeo_codec_frame_count_approval.json',dict(status='APPROVED',user_reply='允许 Vimeo 使用全部 7 帧',
        vimeo_codec_frames=7,ulong_codec_frames=32,no_duplication=True,no_interpolation=True,no_cross_sequence_concatenation=True))
    ready=load(ROOT/'vimeo_training_readiness.json');ready['codec_frame_count_decision']='User approved seven native frames, no padding/concatenation';dump(ROOT/'vimeo_training_readiness.json',ready)
    rawtrain=load(V61/'train_manifest_1024.json');split=load(V61/'data_split_integrity.json')
    assert sha(V61/'train_manifest_1024.json')==split['train_manifest_sha256']
    train_sha={v['sha256'] for v in rawtrain['videos']}
    validation={v['sha256'] for s in ('normal','hard') for v in load(V62/f'validation_{s}_manifest.json')['videos']}
    final=load(V62B/'source_manifest.json')['videos'];test_sha={v['source_sha256'] for v in final if v['dataset']=='ulong'}
    excluded=train_sha|validation|test_sha|set(split['reserved_sha256'])|set(split['validation_sha256'])
    stats=read(V61/'training_source_statistics.csv');by_sha={v['sha256']:v for v in stats if v['status']=='PASS'}
    unused=sorted([v for h,v in by_sha.items() if h not in excluded],key=lambda v:v['sha256'])
    chosen=random.Random(SEED).sample(unused,1024);videos=[]
    for dataset,records in (('ulong_train1024',rawtrain['videos']),('ulong_unused1024',chosen)):
        out=[]
        for i,r in enumerate(records):
            s=by_sha.get(r['sha256']);assert s is not None
            v=dict(dataset=dataset,sample_id=dataset+'_'+f'{i:04d}',filename=r['filename'],sha256=r['sha256'],
                path=r.get('path',str(ROOT/'cache/ulong_unused'/f'{r["sha256"]}.mp4')),
                archive=s['archive'],archive_member=s['filename'],width=int(s['width']),height=int(s['height']),fps=float(s['fps']),
                source_frames=int(s['frames']),kind='video',role='candidate_training',source_bytes=int(s['bytes']))
            out.append(v);videos.append(v)
        dump(ROOT/'manifests'/f'{dataset}.json',dict(status='PASS',seed=SEED,exact_source_manifest_sha256=sha(V61/'train_manifest_1024.json') if dataset=='ulong_train1024' else None,
             excluded_sha256=sorted(excluded) if dataset=='ulong_unused1024' else [],videos=out))
    vm=load(ROOT/'manifests/vimeo_train2048.json')['videos']
    for r in vm:videos.append(dict(r,kind='vimeo',role='candidate_training',source_frames=7))
    refs=[]
    for r in final:
        if r['dataset']=='uvg':
            v=dict(r,dataset='uvg_reference_test',sample_id='uvg_'+r['name'],kind='frozen',frozen_dataset='uvg',role='reference_test',
                source_frames=r['frames'],fps=30.0)
            videos.append(v);refs.append(v)
        if r['dataset']=='virat720':
            counterpart=next(x for x in final if x['dataset']=='virat480' and x['paired_sequence']==r['paired_sequence'])
            assert r['source_frame_indices']==counterpart['source_frame_indices']
            # One content observation per underlying segment. Keep both resolutions for codec reuse.
            fps_meta=r['source_fps_metadata']['avg_frame_rate'];num,den=map(int,fps_meta.split('/'));source_fps=num/den
            indices=r['source_frame_indices'];effective=(len(indices)-1)*source_fps/(indices[-1]-indices[0])
            v=dict(r,dataset='virat_reference',sample_id='virat_'+str(r['video_index']),kind='frozen',frozen_dataset='virat720',role='reference_test',
                source_frames=r['frames'],fps=effective,codec_reference_records=[r,counterpart])
            videos.append(v);refs.append(v)
    assert len([v for v in refs if v['dataset']=='uvg_reference_test'])==7 and len([v for v in refs if v['dataset']=='virat_reference'])==8
    dump(ROOT/'manifests/reference_cohorts.json',dict(status='PASS',videos=refs,source_manifest_sha256=sha(V62B/'source_manifest.json')))
    dump(ROOT/'manifests/all_sources.json',dict(status='PASS',videos=videos))
    protocol=dict(status='PASS',seed=SEED,training_loader_source=str(V4/'train.py'),training_loader_sha256=sha(V4/'train.py'),
        current_wrapper_training=dict(clip_length=4,frame_stride=1,selection='rng.choice video; uniform start, then shared random spatial crop',
            source_fps_used=False,crop=[256,256],resize=False,RGB_range=[0,1],normalization='uint8 RGB / 255; codec subsequently casts half and applies 2*x-1'),
        original_training_code_availability='README states training code not released; no claim about unobserved official training loader',
        evaluation_loader=str(V52/'canonical_loader.py'),
        standardized_content_view=dict(max_frames=32,long_side=512,aspect_ratio_preserved=True,resize='OpenCV INTER_AREA shrinking / INTER_LINEAR enlarging',
            video_temporal='first up-to-32 nearest timestamp frames at <=30fps, unique ordered indices, no duplicate/interpolation',
            vimeo_temporal='all seven images in official order, fps unknown',RGB_range=[0,1]),
        actual_codec_input_view=dict(ulong='first 32 raw frames decoded with ffmpeg RGB24; preserves V5-A.2 color/order pipeline; no resize or temporal resampling; comparison fps30',
            uvg='first32 of exact frozen canonical64; full64 existing Original codec profile reused',
            virat='first32 frozen720 PNGs, unique source segments; existing full-length720/480 codec profiles reused',
            vimeo='all seven native448x256 RGB images; current wrapper would still apply random256crop during training, but this diagnostic uses full-frame inference',
            vimeo_comparison_fps='30 assumed only for kbps; source fps remains missing, use bpp for coverage'))
    dump(ROOT/'training_input_protocol_audit.json',protocol)
    dump(ROOT/'statistics_method_audit.json',dict(status='FROZEN',luminance='0.2126R+0.7152G+0.0722B',
        spatial='frame-average moments; Sobel 3x3 scale1/8 reflect border; gradient magnitude; Laplacian ksize3 variance',
        high_frequency_energy='mean((Y-GaussianBlur(Y,5x5,sigma1,reflect))**2)',edge_threshold=0.05,edge_definition='fraction of normalized Sobel magnitude >0.05',
        temporal_percentiles='percentiles of adjacent-frame mean absolute RGB differences',
        camera='goodFeaturesToTrack(400,quality=.01,minDistance=8); pyramidal LK21x21,maxLevel3; estimateAffinePartial2D RANSAC3px,2000iterations,confidence.99; deterministic RNG0',
        camera_failure_policy='Missing numeric camera fields and explicit status, never silent identity/zero fill; average over successful pairs with coverage count',
        compensated_residual='Warp previous RGB by estimated partial affine to current coordinates; exclude invalid border mask; mean absolute/squared RGB residual'))
    compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback/flolpips_compat.py')
    dump(ROOT/'motion_method_audit.json',dict(status='FROZEN',method='Existing locally cached PWC-Net via established FloLPIPS compatibility layer',
        source=str(compat),source_sha256=sha(compat),weights='/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/pwc-network-default.pytorch',
        units='pixels per frame transition at each explicitly recorded view resolution; normalized-by-image-diagonal fields also retained',proxy_not_ground_truth=True,no_download=True))
    weights=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    dump(ROOT/'visual_embedding_audit.json',dict(status='AVAILABLE' if weights.exists() else 'SKIPPED',model='torchvision ResNet50 ImageNet, classifier removed',
        weights=str(weights),weights_sha256=sha(weights) if weights.exists() else None,frames='first/middle/last standardized-content frames',
        transform='bilinear224x224; ImageNet mean/std; mean2048-d feature over3 frames',no_download=True,used_in_primary_coverage=False))
    paths=[V61/'train_manifest_1024.json',V61/'training_source_statistics.csv',V61/'training_distribution_summary.csv',V61/'stratification_audit.json',
        V62/'validation_normal_manifest.json',V62/'validation_hard_manifest.json',V62B/'source_manifest.json',V62B/'final_integrity.json',PUBLIC/'cohort.json',
        V4/'train.py',V52/'canonical_loader.py',REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',compat,
        Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/pwc-network-default.pytorch')]
    if weights.exists():paths.append(weights)
    dump(ROOT/'frozen_dependencies.json',{str(p):sha(p) for p in paths})
    dump(ROOT/'data_leakage_audit.json',dict(status='PENDING_FIT_CHECKS',source_sampling_seed=SEED,no_reference_guided_source_sampling=True,
        unused_disjoint=True,unused_excluded_sha256=sorted(excluded),no_training=True,
        scaler_fit_allowed=['ulong_train1024','ulong_unused1024','vimeo_train2048'],PCA_fit_allowed=['ulong_train1024','ulong_unused1024','vimeo_train2048'],
        reference_datasets=['uvg_reference_test','virat_reference'],thresholds_fixed_before_reference_profiling=True,
        caution='If this diagnosis informs a future recipe, UVG cannot remain the only final generalization evidence; retain another unseen benchmark.'))
    dump(ROOT/'cohort_audit.json',dict(status='PASS',counts={d:sum(v['dataset']==d for v in videos) for d in sorted({v['dataset'] for v in videos})},
        source_manifest_sha256=sha(ROOT/'manifests/all_sources.json'),no_final_ulong_in_fit=True,Vimeo7frames_user_approved=True))
    dump(ROOT/'pipeline_status.json',dict(status='COHORTS_FROZEN',updated_unix=time.time()))
    print('COHORTS FROZEN',len(videos),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/cohort_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
