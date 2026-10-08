"""Freeze inputs, training schedule, split identities and historical code."""
from mixed_io import *
def main():
    import torch, numpy as np
    from PIL import Image
    torch.set_num_threads(2)
    for d in ('audits','logs','checkpoints','evaluation/checkpoints','training_logs','parts','features','validation','results','manifests'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'checkpoints/latest.pt').exists()
    cfg=load(V614/'config.json');cp=cfg['source_checkpoint'];assert sha(cp['path'])==cp['sha256']==EXPECTED
    state=torch.load(cp['path'],map_location='cpu',weights_only=True)
    assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes'];del state
    cfg.update(schema='v615_mixed_domain_replay',branch='mixed',domain_seed=20261009,train_fit_steps=[],fixed_report_steps=[1000,5000],primary_checkpoint=5000)
    dump(ROOT/'config.json',cfg)
    for src,dst in [(V614/'train_manifest.json','train_manifest.json'),(V61/'train_manifest_1024.json','train_manifest_ulong.json'),(V62/'validation_normal_manifest.json','validation_normal_manifest.json'),(V62/'validation_hard_manifest.json','validation_hard_manifest.json'),(V614/'qp_semantics_audit.json','qp_semantics_audit.json')]:dump(ROOT/dst,load(src))
    log=V614/'training_logs/uvg_only.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
    assert [r['adaptation_step'] for r in rows]==list(range(1,5001))
    qs=[r['external_qp'] for r in rows];assert all(type(q)==int and 0<=q<=9 for q in qs)
    dump(ROOT/'qp_schedule.json',dict(source=str(log),source_sha256=sha(log),external_qp=qs))
    dump(ROOT/'sampling_config.json',dict(seed=cfg['seed'],domain_seed=cfg['domain_seed'],domain_probability={'ulong':.5,'uvg':.5},domain_selection='independent random.Random; random()<0.5',video_selection='uniform within selected domain',ulong_sampler=str(V4/'train.py')+':sample_clip',uvg_pool='existing 64 canonical frames per training video',shared_crop=256,frames=4,QP='fixed external_qp replay from v6.14; independent of both RNG streams',validation_QPs=[0,2,4,6,8,9],validation_ulong_frames=32))
    for d in ('ulong','hevc_b'):dump(ROOT/'manifests'/f'{d}.json',load(V614B/'manifests'/f'{d}.json'))
    for d,src in [('uvg_holdout','holdout_manifest.json'),('uvg_validation','validation_manifest.json')]:
        vs=load(V614/src)['videos'];dump(ROOT/'manifests'/f'{d}.json',dict(videos=[dict(v,dataset=d) for v in vs],origin=str(V614/src),origin_sha256=sha(V614/src)))
    # Preserve historical validation's first 32 decoded frames and native fps.
    sys.path.insert(0,str(V62));e=module('mixed_prepare_validation',V62/'fullqp_evaluate.py');e.ROOT=ROOT
    for split in ('normal','hard'):
        vs=[]
        for v in load(ROOT/f'validation_{split}_manifest.json')['videos']:
            fs,h=e.read_frames(v);del fs
            vs.append(dict(v,dataset='ulong_'+split,source_path=v['path'],source_sha256=v['sha256'],rgb_sha256=h,source_frame_indices=list(range(v['frames'])),rate_accounting_fps=v['fps']))
        dump(ROOT/'manifests'/f'ulong_{split}.json',dict(videos=vs,origin=str(V62/f'validation_{split}_manifest.json')))
        print('VALIDATION MANIFEST',split,len(vs),flush=True)
    train=load(ROOT/'train_manifest_ulong.json')['videos'];uvg=load(ROOT/'train_manifest.json')['videos'];held=[v for d in (*DATASETS,'uvg_validation','ulong_normal','ulong_hard') for v in sources(d)]
    assert len(train)==1024 and len({v['filename'] for v in train})==1024
    assert {v['name'] for v in uvg}=={'Beauty','ReadySteadyGo','ShakeNDry','YachtRide'}
    heldhash={v['source_sha256'] for v in held};heldpath={str(Path(v['source_path']).resolve()) for v in held};heldnames={v['name'] for v in held}
    audit=[]
    for v in train:
        assert v['sha256'] not in heldhash and str(Path(v['path']).resolve()) not in heldpath and Path(v['filename']).stem not in heldnames
        assert sha(v['path'])==v['sha256'];audit.append(dict(path=v['path'],sha256=v['sha256']))
    for v in uvg:
        assert v['source_sha256'] not in heldhash and v['name'] not in heldnames and v['frames']==64
    # Verify exact canonical frame pool for training and final/UVG validation.
    for v in [*uvg,*[v for d in (*DATASETS,'uvg_validation') for v in sources(d)]]:
        h=hashlib.sha256()
        for i in range(v['frames']):
            p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if v['dataset']=='hevc_b' else f'frame_{i:06d}.png')
            assert sha(p)==v['frame_png_sha256' if v['dataset']=='hevc_b' else 'frame_file_sha256'][i]
            with Image.open(p) as im:assert im.mode=='RGB' and im.size==(v['width'],v['height']);raw=np.asarray(im,dtype=np.uint8).tobytes()
            assert hashlib.sha256(raw).hexdigest()==v['frame_rgb_sha256'][i];h.update(raw)
        assert h.hexdigest()==v['rgb_sha256']
        print('RGB PASS',v['name'],flush=True)
    dump(ROOT/'dataset_split.json',dict(status='PASS',ulong_train_count=1024,uvg_train=[v['name'] for v in uvg],excluded={d:[v['name'] for v in sources(d)] for d in (*DATASETS,'uvg_validation','ulong_normal','ulong_hard')},source_hash_disjoint=True,path_disjoint=True,name_disjoint=True,training_sources_verified=audit))
    ecfg=load(V614B/'config.json');ecfg.update(schema='v615_evaluation',methods=METHODS,datasets=DATASETS,no_training=False,expected_points=900)
    dump(ROOT/'evaluation/config.json',ecfg)
    protocol=load(V614B/'protocol.json');protocol.update(no_training=False,methods=METHODS,comparison='six-model common actual bpp interval; 100 uniform ln(bpp) points; PCHIP without extrapolation',fixed_report_steps=[1000,5000],validation='historical U-Long normal/hard 32 frames QP 0,2,4,6,8,9; Bosphorus 64 frames QP 0-9; no selection')
    dump(ROOT/'protocol.json',protocol)
    deps={}
    for source in (V614/'audits/dependencies.json',V614B/'audits/dependencies.json'):
        for p,h in load(source).items():
            assert sha(p)==h,('dependency changed',p);deps[p]=h
    for folder in (V614,V614B):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in [V61/'train_manifest_1024.json',log,V614/'train_manifest.json',V614/'validation_manifest.json',V614/'holdout_manifest.json',V62/'validation_normal_manifest.json',V62/'validation_hard_manifest.json']:deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/initialization_source.json',dict(status='PASS',checkpoint=cp,source_optimizer_restored=False))
    dump(ROOT/'preflight_audit.json',dict(status='PASS',checkpoint_SHA256=True,training_sources=1028,QP_sequence_length=5000,source_RGB_verified=True,split_disjoint=True))
    seal();print('PREFLIGHT PASS',flush=True)
def seal():
    ps=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'protocol.json',ROOT/'qp_schedule.json',ROOT/'sampling_config.json',ROOT/'dataset_split.json',ROOT/'qp_semantics_audit.json',*ROOT.glob('*manifest*.json'),*(ROOT/'manifests').glob('*.json')]
    dump(ROOT/'audits/local_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
if __name__=='__main__':main()
