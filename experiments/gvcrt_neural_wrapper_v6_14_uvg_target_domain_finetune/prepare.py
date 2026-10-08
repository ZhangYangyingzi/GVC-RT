"""Verify frozen canonical RGB, disjoint video groups, and source checkpoint."""
import traceback
from v614_io import *
def main():
    import numpy as np,torch
    from PIL import Image
    command([PYTHON,'-B',str(Path(__file__).resolve())]);torch.set_num_threads(2)
    assert not list((ROOT/'checkpoints').glob('*.pt'))
    cp=load(BASE/'config.json')['checkpoints']['v62'];assert sha(cp['path'])==cp['sha256']==EXPECTED
    state=torch.load(cp['path'],map_location='cpu',weights_only=True);assert state['step']==1000
    assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes'];del state
    vs=[v for v in load(BASE/'source_manifest.json')['videos'] if v['dataset']=='uvg'];expected=set(sum(SPLITS.values(),[]))
    assert len(vs)==7 and {v['name'] for v in vs}==expected,('missing video',expected-{v['name'] for v in vs})
    assert len({v['source_sha256'] for v in vs})==len({v['source_path'] for v in vs})==7
    for v in vs:
        assert v['frames']==64 and (v['width'],v['height'])==(1920,1080)
        files=[Path(v['input_dir'])/f'frame_{i:06d}.png' for i in range(64)];assert sorted(Path(v['input_dir']).glob('*.png'))==files
        h=hashlib.sha256()
        for i,p in enumerate(files):
            assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:assert im.mode=='RGB' and im.size==(1920,1080);a=np.asarray(im,dtype=np.uint8)
            raw=a.tobytes();assert hashlib.sha256(raw).hexdigest()==v['frame_rgb_sha256'][i];h.update(raw)
        assert h.hexdigest()==v['rgb_sha256'];print('RGB VERIFIED',v['name'],flush=True)
    dump(ROOT/'source_manifest.json',dict(videos=vs,source=str(BASE/'source_manifest.json'),source_sha256=sha(BASE/'source_manifest.json')))
    identities=[]
    for s,names in SPLITS.items():
        records=[dict(v,split=s,exposure='training_exposed' if s=='train_fit' else 'held_out_from_this_training',canonical_frame_indices=list(range(64)),canonical_time_seconds=[i/v['source_fps_metadata'] for i in v['source_frame_indices']],actual_time_intervals_seconds=[(b-a)/v['source_fps_metadata'] for a,b in zip(v['source_frame_indices'],v['source_frame_indices'][1:])]) for v in vs if v['name'] in names]
        dump(ROOT/('train_manifest.json' if s=='train_fit' else s+'_manifest.json'),dict(videos=records,source_video_disjoint=True))
        identities.extend(dict(name=v['name'],split=s,path=v['source_path'],source_sha256=v['source_sha256'],rgb_sha256=v['rgb_sha256']) for v in records)
    dump(ROOT/'dataset_split.json',dict(status='PASS',splits=SPLITS,identities=identities,aliases_used=[],full_source_video_disjoint=True,training_frame_pool='4 train videos only; existing 64 canonical frames',source_manifest_sha256=sha(BASE/'source_manifest.json')))
    deps=load(BASE/'frozen_source_hashes.json');deps={p:h for p,h in deps.items() if Path(p).suffix=='.py' or Path(p).name in ('GVC-RT_I.pt','GVC-RT_P.pt')}
    for p,h in deps.items():assert sha(p)==h,('historical implementation changed',p)
    for p in (V62/'fullqp_train.py',V62/'fullqp_common.py',BASE/'evaluate.py',BASE/'v62b_io.py',BASE/'metric_implementation_audit.json',Path(cp['path'])):deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',inventory())
    cfg=dict(schema='v614_uvg_only',branch='uvg_only',source_step=1000,source_checkpoint=cp,seed=20261008,updates=5000,checkpoint_steps=list(STEPS),train_fit_steps=[0,1000,3000,5000],lambda_proxy=.01,force_zero_thres=.12,objective='unchanged fullqp_train.run_clip schedule_s1p0; no rate gradient fix',optimizer=dict(wrapper_lr=5e-5,bridge_lr=3e-6,generator_lr=5e-7,weight_decay=1e-4,grad_clip=1.),fresh_optimizer=True,restore_source_optimizer=False,clip_length=4,crop=256,batch=1,external_qps=list(range(10)),allowed_gpus=[4,5,6,7],primary_checkpoint=5000)
    dump(ROOT/'config.json',cfg);dump(ROOT/'qp_semantics_audit.json',load(BASE/'qp_semantics_audit.json'))
    dump(ROOT/'evaluation/config.json',dict(checkpoints={'v62_initial':cp},compression_hash=load(BASE/'parts/uvg/original/video_00_qp0.json')['compression_hash_before']))
    dump(ROOT/'audits/initialization_source.json',dict(status='PASS',checkpoint=cp,source_step=1000,adaptation_step=0,full_wrapper_bridge_generator=True,fresh_optimizer=True))
    dump(ROOT/'protocol.json',dict(source='V6.2b unchanged codec/evaluator/metrics',source_manifest_sha256=sha(BASE/'source_manifest.json'),rate_axis='actual bpp',kbps_fps=30,frames=64,source_frame_indices='inherited verbatim',canvas=[1920,1088],metric_crop=[1920,1080],force_zero_thres=.12,QP_sha256=sha(ROOT/'qp_semantics_audit.json'),validation_selection='common all-candidate ln(bpp) interval, 100 uniform points, PCHIP mean LPIPS+DISTS, tie earliest adaptation step',holdout_gate='training PASS + frozen checkpoint index; no holdout selection',FID='separate split-specific GT feature pools; never mean per-video FID'))
    for d in ('logs','parts','audits','checkpoints','evaluation/checkpoints','training_logs','results/train_fit','results/validation','results/holdout'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    dump(ROOT/'preflight_audit.json',dict(status='PASS',sequences=7,training_videos=4,validation_videos=1,holdout_videos=2,RGB_frames_verified=448))
    seal();print('PREFLIGHT PASS',flush=True)
def seal():
    assert not list((ROOT/'checkpoints').glob('*.pt'))
    ps=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'protocol.json',ROOT/'dataset_split.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',*ROOT.glob('*_manifest.json')]
    dump(ROOT/'audits/local_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'preflight_audit.json',dict(status='FAIL',error=traceback.format_exc()));raise
