"""Freeze byte-identical plans and settled V6.9/V6.10 protocol before any training."""
from v68_io import *
EXPECTED='97b44de2aadfcebfab03148619bf2b30a94ae80c95b885bf3bed741147130b0f'
def seal():
    assert not list((ROOT/'branches').glob('*/checkpoints/step_*.pt'))
    files=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',*list((ROOT/'audits').glob('*manifest.json')),ROOT/'audits/continuation_training_plans.json',ROOT/'audits/visualization_plan.json']
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
def main():
    import torch,numpy as np
    torch.set_num_threads(2)
    for d in ('audits/gradient_checks','logs/failures','results','parts','training_logs','evaluation'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','parts','logs'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    assert sha(SOURCE)==EXPECTED
    s=torch.load(SOURCE,map_location='cpu',weights_only=True)
    assert s['lambda_struct']==0.045785124942642835
    assert [g['lr'] for g in s['optimizer']['param_groups']]==[5e-5,3e-6,5e-7]
    meta=load(SOURCE.parent.parent/'checkpoint_hashes.json')['1000'];assert meta['sha256']==EXPECTED
    hashes={k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')};assert hashes==meta['module_hashes']
    audit=dict(status='PASS',sha256=EXPECTED,module_hashes=hashes,optimizer=optimizer_summary(s['optimizer']),compression_hash=s['compression_hash'],lambda_struct=s['lambda_struct'],rng_hashes={k:state_hash(s[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')})
    dump(ROOT/'audits/source_B1000_audit.json',audit)
    cfg=load(V610/'config.json')
    for k in ('discriminator','lambda_adv_max','lambda_adv_ramp_updates'):cfg.pop(k,None)
    cfg.update(schema='v6_11_joint_compression_adaptation',branches=list(BRANCHES),methods=list(METHODS),branch_generator_lr=LRS,training_objective='Exact V6.6 B / V6.10 C_plain',compression_core_lr=1e-6,compression_core_dtype='float32',core_grad_clip=1.,PBG_grad_clip=1.,baseline_experiment=str(V610))
    dump(ROOT/'config.json',cfg)
    for name in ('source_manifest.json','qp_semantics_audit.json'):
        (ROOT/name).write_bytes((V610/name).read_bytes());assert sha(ROOT/name)==sha(V69/name)
    names=('continuation_training_plans.json','visualization_plan.json','fid_protocol_audit.json','GT_feature_cache.json','vimeo_heldout_manifest.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS])
    for name in names:
        (ROOT/'audits'/name).write_bytes((V610/'audits'/name).read_bytes())
    plans=load(ROOT/'audits/continuation_training_plans.json')['plans'];assert len(plans)==1000
    assert sha(ROOT/'audits/continuation_training_plans.json')=='8e70ec113bb4ce741312cfe54090744ab10cc8fda35f89210fdfacd26d47ed1c'
    cache=load(ROOT/'audits/GT_feature_cache.json')['datasets']
    for d in DATASETS:
        c=cache[d];assert sha(c['path'])==c['sha256'];a=np.load(c['path'])
        assert array_hash(a)==c['array_sha256'] and a.shape==(sum(v['frames'] for v in sources(d)),2048)
        assert sha(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json')==sha(V69/'audits'/f'fid_bootstrap_indices_{d}.json')
    for v in sources():
        assert v['frames'] in ((96,33) if v['dataset'].startswith('virat') else (64,))
        assert len(v['source_frame_indices'])==v['frames']
        assert v['rate_accounting_fps']==(20 if v['dataset'].startswith('virat') else 30)
    resolution=load(V610/'audits/evaluation_protocol_resolution.json');assert resolution['status']=='PASS'
    resolution.update(reused_from=str(V610/'audits/evaluation_protocol_resolution.json'),source_sha256=sha(V610/'audits/evaluation_protocol_resolution.json'),verified_cache_hashes=True)
    dump(ROOT/'audits/evaluation_protocol_resolution.json',resolution)
    for name in ('evaluation_frames_per_video.json','evaluation_frames_per_video.csv'):
        (ROOT/'audits'/name).write_bytes((V610/'audits'/name).read_bytes())
    dump(ROOT/'evaluation/config.json',dict(checkpoints={'B1000':meta},core_hashes={m:cfg['compression_hash'] for m in ('original','B1000','F1500','F2000')}))
    deps={}
    for folder in (V610,V69,V66,V64,V62,V62B,V4,ENGINE,REPO/'src'):
        for p in (folder.rglob('*.py') if folder==REPO/'src' else folder.glob('*.py')):deps[str(p)]=sha(p)
    for p in (SOURCE,REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',V64/'manifests/vimeo_official_train.json'):deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/source_code_state.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)))
    dump(ROOT/'audits/old_inventory.json',old_inventory());seal()
    dump(ROOT/'preflight_audit.json',dict(status='PASS',initialization=True,training_plans=True,evaluation_protocol_verified=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='implementation_checks'))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
