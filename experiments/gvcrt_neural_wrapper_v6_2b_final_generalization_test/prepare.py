"""Freeze the explicitly requested candidate and audit existing frozen inputs."""
import ast
import math
import traceback
from v62b_io import *

def main():
    for d in ('logs','parts','results','bitstreams','features','visualizations'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    if (ROOT/'preflight_audit.json').exists() and load(ROOT/'preflight_audit.json')['status']=='PASS':
        check_frozen();print('ALREADY PREPARED');return
    dump(ROOT/'final_integrity.json',dict(status='PENDING',reason='Preflight, fresh full-QP evaluation and all reports required'))
    if not (ROOT/'old_inventory_before.json').exists():dump(ROOT/'old_inventory_before.json',old_inventory())
    import torch
    checkpoints={}
    for name,path in dict(v41=V41/'checkpoints/beta_high/step_20000.pt',v62=V62/'branches/schedule_s1p0/checkpoints/step_1000.pt').items():
        digest=sha(path);assert digest==EXPECTED[name],('checkpoint SHA mismatch',name,digest)
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['step']==(1000 if name=='v62' else 20000)
        checkpoints[name]=dict(path=str(path),sha256=digest,module_hashes={k:tensor_hash(cp[k]) for k in ('wrapper','bridge','generator')})
        del cp
    dump(ROOT/'frozen_candidate_audit.json',dict(status='PASS',branch='schedule_s1p0',step=1000,checkpoint=checkpoints['v62'],
        selection_source='V6.2-A validation only; exact candidate prescribed by user before final tests',
        model_selection_performed_in_this_experiment=False,retraining=False,legacy_checkpoint=checkpoints['v41']))
    assert load(V62/'final_integrity.json')['status']=='PASS'
    frozen={}
    for folder in (V41,V52,V61,V62,ENGINE,REPO/'src',ROOT.parent/'gvcrt_neural_wrapper_v2_joint',ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint',REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src'):
        for p in folder.rglob('*.py'):frozen[str(p)]=sha(p)
    for p in (V52/'canonical_source_audit.json',V52/'test_manifest.json',V61/'final_sources.json',PUBLIC/'cohort.json',PUBLIC/'prepare_inputs.py',
              V62/'qp_train_eval_semantics_audit.json',V62/'validation_normal_manifest.json',V62/'validation_hard_manifest.json',V61/'train_manifest_1024.json',
              REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',FID_SOURCE):frozen[str(p)]=sha(p)
    compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback/flolpips_compat.py')
    frozen[str(compat)]=sha(compat)
    dump(ROOT/'frozen_source_hashes.json',frozen)
    loader=module('v62b_loader',V52/'canonical_loader.py');records=load(V52/'canonical_source_audit.json')['records']
    final61={(v['dataset'],v['name']):v for v in load(V61/'final_sources.json')['videos'] if v['dataset'] in ('ulong','uvg')}
    videos=[];origin_hashes={}
    for r in records:
        dataset=r['dataset'];assert dataset in ('ulong','uvg')
        path=Path(r['original_path']);origin_hashes[str(path)]=sha(path);assert origin_hashes[str(path)]==r['original_sha256']
        rgbhash=hashlib.sha256();rgb=[];files=[]
        for i,a in enumerate(loader.canonical_arrays(r['canonical_dir'])):
            data=a.tobytes();h=hashlib.sha256(data).hexdigest();rgb.append(h);rgbhash.update(data)
            p=Path(r['canonical_dir'])/f'frame_{i:06d}.png';files.append(sha(p))
        assert files==r['canonical_frame_sha256'] and rgb==r['canonical_frame_rgb_sha256']
        assert rgbhash.hexdigest()==r['whole_sequence_rgb_hash']==final61[(dataset,r['name'])]['rgb_sha256']
        assert len(rgb)==r['canonical_num_frames']==64 and r['canonical_width']==1920 and r['canonical_height']==1080
        v=dict(dataset=dataset,video_index=r['video_index'],name=r['name'],input_dir=r['canonical_dir'],frames=64,width=1920,height=1080,
            rate_accounting_fps=30.0,source_path=str(path),source_sha256=r['original_sha256'],source_fps_metadata=r['original_fps'],
            source_frame_indices=r['selected_original_frame_indices'],rgb_sha256=rgbhash.hexdigest(),frame_rgb_sha256=rgb,frame_file_sha256=files,
            loader='V5-A.2 canonical_loader.py::canonical_arrays',temporal_resampling_this_experiment=False)
        videos.append(v);print('CANONICAL VERIFIED',dataset,r['name'],flush=True)
    assert len([v for v in videos if v['dataset']=='ulong'])==8 and len([v for v in videos if v['dataset']=='uvg'])==7
    assert {v['name'] for v in videos if v['dataset']=='uvg'}=={'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'}
    # Reuse frozen PNGs directly. Audit their conversion against the original helper without writing anything.
    import cv2
    import numpy as np
    prep=module('v62b_old_prepare',PUBLIC/'prepare_inputs.py');cohort=load(PUBLIC/'cohort.json');virat_audit=[];decoded={};ids={'virat720':0,'virat480':0}
    for r in cohort:
        dataset='virat720' if r['sequence'].endswith('__720p') else 'virat480'
        assert (r['width'],r['height'])==((1280,720) if dataset=='virat720' else (854,480))
        indices=r['source_frame_indices'];assert len(indices)==r['frames'] and all(b>a for a,b in zip(indices,indices[1:]))
        key=(r['source'],tuple(indices))
        if key not in decoded:decoded[key]=prep.decode_indices(Path(r['source']),indices)
        if r['source'] not in origin_hashes:origin_hashes[r['source']]=sha(r['source'])
        frame_dir=PUBLIC/'input_frames'/r['sequence'];expected=[frame_dir/f'im{i+1:05d}.png' for i in range(r['frames'])]
        assert sorted(frame_dir.glob('*.png'))==expected
        v=dict(dataset=dataset,video_index=ids[dataset],name=r['sequence'],paired_sequence=r['sequence'].rsplit('__',1)[0],input_dir=str(frame_dir),
            frames=r['frames'],width=r['width'],height=r['height'],rate_accounting_fps=20.0,
            source_path=r['source'],source_sha256=origin_hashes[r['source']],source_frame_indices=indices,
            frozen_cohort_fps_metadata=r['fps_num']/r['fps_den'],loader='frozen PNGs in PUBLIC/input_frames; no conversion or video recoding',temporal_resampling_this_experiment=False)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,r_frame_rate,width,height','-of','json',r['source']]))['streams'][0]
        v['source_fps_metadata']=probe
        h=hashlib.sha256();rgb=[];filehash=[]
        for i,a in enumerate(arrays(v)):
            bgr=decoded[key][i]
            if bgr.shape[:2]!=(v['height'],v['width']):bgr=cv2.resize(bgr,(v['width'],v['height']),interpolation=cv2.INTER_LANCZOS4)
            assert np.array_equal(a,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),('Frozen VIRAT source mismatch',r['sequence'],i)
            data=a.tobytes();h.update(data);rgb.append(hashlib.sha256(data).hexdigest());filehash.append(sha(expected[i]))
        v.update(rgb_sha256=h.hexdigest(),frame_rgb_sha256=rgb,frame_file_sha256=filehash,
                 source_frame_indices_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest())
        videos.append(v);ids[dataset]+=1
        virat_audit.append(dict(sequence=v['name'],dataset=dataset,resolution=[v['width'],v['height']],num_frames=v['frames'],
            source_frame_indices_sha256=v['source_frame_indices_sha256'],source_fps_metadata=probe,frozen_cohort_fps_metadata=v['frozen_cohort_fps_metadata'],
            rate_accounting_fps=20.0,temporal_resampling=False,source_to_frozen_png_exact=True,rgb_sha256=v['rgb_sha256']))
        print('VIRAT VERIFIED',dataset,v['name'],flush=True)
    assert ids=={'virat720':8,'virat480':8}
    for a in [v for v in videos if v['dataset']=='virat720']:
        b=next(v for v in videos if v['dataset']=='virat480' and v['paired_sequence']==a['paired_sequence'])
        assert a['source_frame_indices']==b['source_frame_indices'] and a['source_sha256']==b['source_sha256']
    train={r['sha256'] for r in load(V61/'train_manifest_1024.json')['videos']}
    validation={r['sha256'] for split in ('normal','hard') for r in load(V62/f'validation_{split}_manifest.json')['videos']}
    tests={v['source_sha256'] for v in videos if v['dataset']=='ulong'};assert not tests&(train|validation)
    dump(ROOT/'source_manifest.json',dict(status='PASS',videos=videos,origin_hashes=origin_hashes))
    dump(ROOT/'frozen_cohort_audit.json',dict(status='PASS',counts={d:sum(v['dataset']==d for v in videos) for d in DATASETS},
        canonical_sources_exact=True,virat_frozen_pngs_verified_against_source_indices=True,ulong_train_validation_test_disjoint=True))
    dump(ROOT/'virat_rate_accounting_audit.json',dict(status='PASS',rows=virat_audit,
        kbps_formula='real_bytes * 8 / num_frames * 20.0 / 1000',bpp_formula='real_bytes * 8 / (num_frames * width * height)',temporal_resampling=False))
    dump(ROOT/'baseline_reuse_audit.json',dict(status='PASS',baseline_reuse=False,policy='All datasets, methods and QPs freshly encoded and independently decoded',
        old_kbps_copied=False,old_metrics_copied=False,reuse_parity_check='Not applicable: no raw baseline results reused'))
    sys.path.insert(0,str(ENGINE));from engine import eco
    from src.models.video_model_gvcrt import DMC
    model=DMC();qps={}
    for q in range(10):
        seq=[q if i==0 else model.shift_qp(q,eco.INDEX_MAP[i%8]) for i in range(max(v['frames'] for v in videos))]
        assert seq[:4]==load(V62/'qp_train_eval_semantics_audit.json')['rows'][q]['evaluation'];qps[str(q)]=seq
    del model
    dump(ROOT/'qp_semantics_audit.json',dict(status='PASS',actual_qps=qps,INDEX_MAP=eco.INDEX_MAP,
        source=str(REPO/'src/models/video_model_gvcrt.py'),evaluation_source=str(ENGINE/'engine.py'),V62_A_exact_match=True))
    dump(ROOT/'metric_implementation_audit.json',dict(status='PASS',engine=str(ENGINE/'engine.py'),metric_runtime=str(V41/'metric_runtime.py'),
        FID_source=str(FID_SOURCE),equal_rate_source=str(V52/'rd_analysis.py'),PSNR='Aggregate pixel MSE per sequence for all datasets',
        SSIM='Existing V4 eval_core frame_metrics SSIM, unchanged',FID='Existing InceptionPool3; low_rank_fid pooled real/reconstruction features',
        FloLPIPS='Existing frame-transition flow-difference-weighted implementation; unchanged'))
    cfg=dict(schema='gvcrt_v62b_final_generalization',checkpoints=checkpoints,methods=list(METHODS),datasets=list(DATASETS),
        external_qps=list(range(10)),force_zero_thres=.12,no_training=True,no_model_selection=True,gpus=[4,5,6,7],
        expected_points=len(videos)*len(METHODS)*10,fresh_all_points=True,
        visualization_policy='First two U-Long; Beauty/Jockey/ReadySteadyGo; first two paired VIRAT sequences at both resolutions; QP0/4/9; save all decoded frames via existing engine helper',
        audit_hashes={n:sha(ROOT/n) for n in ('source_manifest.json','frozen_candidate_audit.json','baseline_reuse_audit.json','virat_rate_accounting_audit.json','qp_semantics_audit.json','metric_implementation_audit.json')})
    dump(ROOT/'config.json',cfg);check_frozen()
    dump(ROOT/'preflight_audit.json',dict(status='PASS',expected_points=cfg['expected_points'],checkpoint_hashes_verified=True,all_sources_verified=True))
    print('V62B PREFLIGHT PASS',cfg['expected_points'],flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'preflight_audit.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
