"""Freeze B1000, exact continued sampler plans, old artifacts and protocols."""
import random
from v68_io import *
def main():
    if (ROOT/'preflight_audit.json').exists():frozen();return
    import torch
    torch.set_num_threads(2)
    for d in ('audits','logs','parts','results','training_logs','evaluation','fid_cache','plots'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','logs','parts'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    (ROOT/'audits/nvidia_smi_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    assert load(V67/'final_integrity.json')['status']=='PASS'
    meta=load(SOURCE.parent.parent/'checkpoint_hashes.json')['1000'];actual=sha(SOURCE)
    assert actual==meta['sha256']
    s=torch.load(SOURCE,map_location='cpu',weights_only=True);assert s['step']==1000 and s['lambda_struct']==0.045785124942642835
    hashes={k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')};assert hashes==meta['module_hashes']
    audit=dict(status='PASS',path=str(SOURCE),sha256=actual,step=s['step'],module_hashes=hashes,compression_hash=s['compression_hash'],optimizer=optimizer_summary(s['optimizer']),lambda_struct=s['lambda_struct'],source_code_hashes=s['code_hashes'],source_config_sha256=s['config_sha256'],sample_rng_state=s['sample_rng_state'],torch_rng_state=s['torch_rng_state'].tolist(),cuda_rng_state=s['cuda_rng_state'].tolist(),rng_hashes={k:state_hash(s[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')})
    dump(ROOT/'audits/source_B1000_audit.json',audit)
    cfg=load(V66/'config.json');cfg.update(schema='v68a_controlled_continuation',source_checkpoint=str(SOURCE),source_checkpoint_sha256=actual,branches=list(BRANCHES),updates=500,absolute_steps=[1001,1500],checkpoint_steps=[1000,1250,1500],alignment_mode=MODE,lambda_struct=s['lambda_struct'],compression_hash=s['compression_hash'],methods=list(METHODS))
    dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation/config.json',dict(checkpoints={'B1000':meta}))
    for name in ('source_manifest.json','qp_semantics_audit.json'):dump(ROOT/name,load(V67/name))
    for name in ('fid_protocol_audit.json','GT_feature_cache.json','vimeo_heldout_manifest.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS]):dump(ROOT/'audits'/name,load(V67/'audits'/name))
    manifest=load(V64/'manifests/vimeo_official_train.json');rng=random.Random();rng.setstate(s['sample_rng_state']);qr=random.Random(cfg['qp_seed'])
    previous=[qr.randrange(10) for _ in range(1000)];assert previous==load(V66/'qp_sequence_audit.json')['sequence']
    plans=[]
    for i in range(1,501):
        sid=rng.choice(manifest['sequences']);start=rng.randrange(4);x=rng.randrange(193);y=rng.randrange(1)
        plans.append(dict(continuation_step=i,absolute_step=1000+i,source='vimeo',sample_id=sid,video=sid,start=start,frame_start=start+1,crop_x=x,crop_y=y,temporal_indices=list(range(start,start+4)),native_image_indices=list(range(start+1,start+5)),external_qp=qr.randrange(10),frame_sha256=[sha(Path(manifest['sequence_root'])/sid/f'im{j}.png') for j in range(start+1,start+5)]))
    for p in plans:p['actual_qps']=[p['external_qp']+j for j in (0,2,0,1)]
    dump(ROOT/'audits/continuation_training_plans.json',dict(status='PASS',plans=plans,initial_sample_rng_hash=state_hash(s['sample_rng_state']),final_sample_rng_hash=state_hash(rng.getstate()),QP_continuation='random.Random(qp_seed) advanced by exactly 1000 prior randrange(10) calls',no_test_data=True))
    held=load(ROOT/'audits/vimeo_heldout_manifest.json')
    dump(ROOT/'audits/interface_diagnostic_plan.json',dict(selection='fixed before any new training; first two frozen UVG videos and first four official heldout sequences; first four frames; QP 0,4,9',uvg_indices=[v['video_index'] for v in sources('uvg')[:2]],vimeo_indices=[v['index'] for v in held['clips'][:4]],frames=4,qps=[0,4,9],no_training_or_calibration_use=True))
    # B1000 data are reused only after artifact and protocol validation.
    for d in DATASETS:
        for v in sources(d):
            for q in range(10):
                p=V67/'parts'/d/'B1000'/f'video_{v["video_index"]:02d}_qp{q}.json';r=load(p);validate_point(r,v,q,'B1000');r.update(reused=True,reused_from=str(p),reused_point_sha256=sha(p));dump(point(d,'B1000',v['video_index'],q),r)
        for q in range(10):
            p=V67/'parts/fid'/d/'B1000'/f'qp{q}.json';r=load(p);assert sha(r['bootstrap_path'])==r['bootstrap_sha256'];dump(ROOT/'parts/fid'/d/'B1000'/f'qp{q}.json',r)
    for p in (V67/'parts/heldout/B1000').glob('*.json'):
        r=load(p)
        for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
        dump(ROOT/'parts/heldout/B1000'/p.name,r)
    assert load(ROOT/'source_manifest.json')==load(V67/'source_manifest.json')
    assert load(ROOT/'audits/fid_protocol_audit.json')==load(V67/'audits/fid_protocol_audit.json')
    dump(ROOT/'audits/B1000_reuse_audit.json',dict(status='PASS',checkpoint_hash_exact_match=True,source_manifest_exact_match=True,evaluation_protocol_exact_match=True,FID_protocol_exact_match=True,formal_points=310,heldout_points=96,FID_points=40,source_integrity_sha256=sha(V67/'final_integrity.json'),GT_cache='unchanged V6.7b paths with file and array hashes',bootstrap='identical frozen V6.7b indices'))
    deps={}
    for folder in (V67,V66,V64,V62,V62B,V4,ENGINE,ROOT.parent/'gvcrt_neural_wrapper_v2_joint',REPO/'src/models'):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in (SOURCE,REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',V64/'manifests/vimeo_official_train.json'):deps[str(p)]=sha(p)
    deps.update(load(V67/'audits/dependencies.json'))
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'audits/continuation_training_plans.json',ROOT/'audits/interface_diagnostic_plan.json']})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',B1000_verified=True));dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='interface_audit_and_calibration'))
    print('PREFLIGHT PASS; reused B1000 310 formal, 96 heldout, 40 FID points',flush=True)
if __name__=='__main__':main()
