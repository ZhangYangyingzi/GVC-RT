"""Freeze teacher provenance, isolated pools, and all 5000 matched plans."""
import random,collections,subprocess
from v68_io import *
from io_utils import command
def seal():
    assert not list((ROOT/'branches').glob('*/checkpoints/step_*.pt'))
    files=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',*list((ROOT/'manifests').glob('*.json')),ROOT/'audits/teacher_gate.json']
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
def main():
    import cv2,numpy as np
    command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    cfg=load(ROOT/'config.json');selected=ROOT/'teacher_checks/pretrain262144_ema'
    raw=load(selected/'teacher_compatibility_raw.json');loading=load(selected/'teacher_loading.json')
    assert raw['metric_rows']==384 and loading['strict_encoder_loading'] and not loading['missing_encoder_keys']
    matrix=np.array(raw['coordinate_checks']['channel_correlation'])
    assert np.array_equal(np.abs(matrix).argmax(axis=1),np.arange(18)) and bool((matrix.diagonal()>0).all())
    rows=read(selected/'teacher_compatibility_per_frame.csv')
    assert all(float(r['latent_cosine'])>0 for r in rows if r['frame_type']=='P')
    assert raw['teacher_frozen'] and raw['teacher_rng_unchanged'] and raw['native_models_unchanged']
    evidence=dict(status='PASS',selected_candidate='pretrain262144',EMA=True,selection_rule='official use_ema=True; coordinate compatibility, not quality ranking',official_source=loading['source'],strict_encoder_loading=True,continuous_before_sign=True,normalization='official identity (no extra normalization)',coordinate_verification=dict(channels=18,identity_argmax_all_channels=True,positive_same_index_correlations=True,no_channel_transform=True,no_latent_interpolation=True,source='fixed 16 U-Long + 16 Vimeo non-test samples'),records={str(p.relative_to(ROOT)):sha(p) for p in selected.glob('*.json')},raw_csv_sha256=sha(selected/'teacher_compatibility_per_frame.csv'),historical_ori_training_teacher_exact_SHA256='not embedded in released ori checkpoints; no assertion of exact historical checkpoint identity',native_interface_binding='verified empirically in unchanged latent axes on fixed non-test inputs, backed by official continuous-LFQ definition and native Open-MAGVIT2 encoder architecture')
    dump(ROOT/'audits/teacher_gate.json',evidence)
    cfg.update(teacher_gate_passed=True,teacher_candidate='pretrain262144',teacher_EMA=True,teacher_checkpoint=loading['source']['checkpoint'],lambda_proxy=.01,lambda_struct=.045785124942642835,methods=list(METHODS),validation_steps=cfg['checkpoint_steps'],evaluation_steps=[5000],gradient_check_steps=[1,100,500,1000,2000,3000,5000],master_dtype='float32',core_compute_dtype='float16',bridge_generator_compute_dtype='float32',deployment_dtype='native ori half',compression_hash=load(V612/'config.json')['compression_hash'],source_checkpoint='native ori I/P only',initial_optimizer_state='empty')
    (ROOT/'source_manifest.json').write_bytes((V612/'source_manifest.json').read_bytes())
    (ROOT/'qp_semantics_audit.json').write_bytes((V612/'qp_semantics_audit.json').read_bytes())
    for name in ('GT_feature_cache.json','fid_protocol_audit.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS]):
        (ROOT/'audits'/name).write_bytes((V612/'audits'/name).read_bytes())
    dump(ROOT/'audits/evaluation_protocol_resolution.json',dict(status='PASS',source_manifest_sha256=sha(V612/'source_manifest.json'),QP_sha256=sha(V612/'qp_semantics_audit.json'),datasets=list(DATASETS),source='V6.12 frozen original per-video frames; no recutting',GT_and_bootstrap_reused=True))
    cp=load(V612/'config.json')['baseline_checkpoints']['v62'];assert sha(cp['path'])==cp['sha256']
    dump(ROOT/'evaluation/config.json',dict(checkpoints={'v62':cp},core_hashes={m:cfg['compression_hash'] for m in ('original','v62')}))
    compat=load(ROOT/'manifests/teacher_compatibility_samples.json')
    validation=compat['samples'];assert len(validation)==32
    dump(ROOT/'manifests/validation_manifest.json',dict(samples=validation,external_qps=[0,4,9],teacher_compatibility_cohort=True,final_test_used=False,checkpoint_selection=False))
    final=load(ROOT/'source_manifest.json')['videos'];excluded={v['source_sha256'] for v in final}|{r['source_sha256'] for r in validation if r['domain']=='ulong'}
    upool=V64.parent/'gvcrt_neural_wrapper_v6_3_dataset_ablation/manifests/ulong_train1024.json'
    ur=[]
    for i,v in enumerate(load(upool)['videos']):
        assert v['sha256'] not in excluded and sha(v['path'])==v['sha256']
        v=dict(v)
        if not all(k in v for k in ('frames','width','height')):
            c=cv2.VideoCapture(v['path']);v.update(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)));c.release()
        assert v['frames']>=4 and min(v['width'],v['height'])>=256;ur.append(v)
        if i%100==0:print('VERIFY U-LONG',i,flush=True)
    dump(ROOT/'manifests/ulong_training.json',dict(videos=ur,source=str(upool),source_sha256=sha(upool),uniform_unit='source video',disjoint_validation_and_final_test=True))
    vm=load(V64/'manifests/vimeo_official_train.json');assert sha(vm['official_train_file'])==vm['official_train_sha256']
    excluded_groups=set(compat['reserved_vimeo_groups'])|{v['sequence'].split('/')[0] for v in load(V612/'audits/vimeo_heldout_manifest.json')['clips']}
    grouped={}
    for sid in vm['sequences']:
        group=sid.split('/')[0]
        if group not in excluded_groups:grouped.setdefault(group,[]).append(sid)
    assert grouped
    dump(ROOT/'manifests/vimeo_training.json',dict(sequence_root=vm['sequence_root'],groups=grouped,excluded_groups=sorted(excluded_groups),official_train_file=vm['official_train_file'],official_train_sha256=vm['official_train_sha256'],uniform_unit='first-level source group, then uniformly sampled septuplet',source_group_definition='Vimeo first-level folder; conservative grouped holdout'))
    rng=random.Random(cfg['seed']);plans=[];qps=load(ROOT/'qp_semantics_audit.json')['actual_qps']
    for step in range(1,5001):
        domain='ulong' if rng.random()<.5 else 'vimeo';q=rng.randrange(10)
        if domain=='ulong':
            v=rng.choice(ur);start=rng.randrange(v['frames']-3);x=rng.randrange(v['width']-255);y=rng.randrange(v['height']-255)
            plan=dict(domain=domain,video_id=Path(v['filename']).stem,path=v['path'],source_sha256=v['sha256'],frame_indices=list(range(start,start+4)),crop=[x,y,256,256])
        else:
            group=rng.choice(sorted(grouped));sid=rng.choice(grouped[group]);start=rng.randrange(4)+1;x=rng.randrange(193)
            paths=[Path(vm['sequence_root'])/sid/f'im{i}.png' for i in range(start,start+4)]
            plan=dict(domain=domain,video_id=sid,source_group=group,paths=list(map(str,paths)),source_sha256=[sha(p) for p in paths],frame_indices=list(range(start,start+4)),crop=[x,0,256,256])
        plans.append(dict(step=step,plan_id=f'v613-{step:05d}',external_qp=q,actual_qps=qps[str(q)][:4],augmentation='none',stride=1,**plan))
    dump(ROOT/'manifests/training_plans.json',dict(seed=cfg['seed'],plans=plans,counts=dict(collections.Counter(p['domain'] for p in plans)),updates=5000))
    dump(ROOT/'config.json',cfg)
    deps={}
    for folder in (V612,V611,V69,V66,V64,V62,V62B,V4,ENGINE,REPO/'src',ROOT.parent/'gvcrt_neural_wrapper_v2_joint'):
        for p in (folder.rglob('*.py') if folder==REPO/'src' else folder.glob('*.py')):deps[str(p)]=sha(p)
    for d in cfg['init_checkpoints'].values():deps[d['path']]=d['sha256']
    deps[loading['source']['checkpoint']['path']]=loading['source']['checkpoint']['sha256']
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    for b in BRANCHES:
        for d in ('checkpoints','parts','logs'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    for d in ('parts','training_logs','results','logs/failures'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    seal();dump(ROOT/'preflight_audit.json',dict(status='PASS',teacher_gate=True,training_plans=5000,validation_samples=32,native_initialization_only=True))
    print('TRAINING PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
