"""Freeze initialization, exact shared 1000 plans, and read-only source evidence."""
import random
from v68_io import *
EXPECTED='97b44de2aadfcebfab03148619bf2b30a94ae80c95b885bf3bed741147130b0f'
def seal():
    assert not list((ROOT/'branches').glob('*/checkpoints/step_*.pt')),'Training already started'
    files=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',
           ROOT/'qp_semantics_audit.json',*list((ROOT/'audits').glob('*manifest.json')),
           ROOT/'audits/continuation_training_plans.json',ROOT/'audits/visualization_plan.json']
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
def main():
    import torch
    torch.set_num_threads(2)
    for d in ('audits','logs/failures','results','parts','training_logs','evaluation'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','parts','logs'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    actual=sha(SOURCE)
    if actual!=EXPECTED:
        dump(ROOT/'audits/source_error.json',dict(expected=EXPECTED,actual=actual,path=str(SOURCE)))
        raise RuntimeError('B1000 checkpoint SHA256 mismatch')
    meta=load(SOURCE.parent.parent/'checkpoint_hashes.json')['1000'];assert meta['sha256']==actual
    s=torch.load(SOURCE,map_location='cpu',weights_only=True)
    hashes={k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')};assert hashes==meta['module_hashes']
    assert s['lambda_struct']==0.045785124942642835
    assert [g['lr'] for g in s['optimizer']['param_groups']]==[5e-5,3e-6,5e-7]
    dump(ROOT/'audits/source_B1000_audit.json',dict(status='PASS',sha256=actual,module_hashes=hashes,
        optimizer=optimizer_summary(s['optimizer']),compression_hash=s['compression_hash'],
        lambda_struct=s['lambda_struct'],rng_hashes={k:state_hash(s[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')}))
    cfg=load(V69/'config.json')
    cfg.update(schema='v6_10_conditional_perceptual_supervision',branches=list(BRANCHES),
        updates=1000,absolute_steps=[1001,2000],checkpoint_steps=[1250,1500,2000],
        evaluation_steps=[1500,2000],methods=list(METHODS),branch_generator_lr=LRS,
        discriminator=dict(feature_channels=256,seed=20261005,lr=1e-4,betas=[0.,.99],weight_decay=0.,grad_clip=1.,warmup_updates=100,supervised_P_indices=[2,3]),
        lambda_adv_max=.01,lambda_adv_ramp_updates=100,training_objective='V6.6 exact B + conditional hinge adversarial on P2/P3',
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=actual)
    dump(ROOT/'config.json',cfg)
    for name in ('source_manifest.json','qp_semantics_audit.json'):
        dump(ROOT/name,load(V69/name))
    for name in ('fid_protocol_audit.json','GT_feature_cache.json','vimeo_heldout_manifest.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS]):
        dump(ROOT/'audits'/name,load(V69/'audits'/name))
    manifest=load(V64/'manifests/vimeo_official_train.json');rng=random.Random();rng.setstate(s['sample_rng_state'])
    qr=random.Random(cfg['qp_seed'])
    for _ in range(1000):qr.randrange(10)
    plans=[]
    for k in range(1,1001):
        sid=rng.choice(manifest['sequences']);start=rng.randrange(4);x=rng.randrange(193);y=rng.randrange(1)
        q=qr.randrange(10)
        plans.append(dict(continuation_step=k,absolute_step=1000+k,source='vimeo',sample_id=sid,video=sid,
            start=start,frame_start=start+1,crop_x=x,crop_y=y,temporal_indices=list(range(start,start+4)),
            native_image_indices=list(range(start+1,start+5)),external_qp=q,
            frame_sha256=[sha(Path(manifest['sequence_root'])/sid/f'im{j}.png') for j in range(start+1,start+5)],
            actual_qps=[q+j for j in (0,2,0,1)]))
    assert plans[:500]==load(V69/'audits/continuation_training_plans.json')['plans']
    held=load(ROOT/'audits/vimeo_heldout_manifest.json')
    assert not set(p['sample_id'] for p in plans)&set(c['sequence'] for c in held['clips'])
    dump(ROOT/'audits/continuation_training_plans.json',dict(status='PASS',plans=plans,first500_exact_match=True,
        source_plans_sha256=sha(V69/'audits/continuation_training_plans.json'),
        initial_sample_rng_hash=state_hash(s['sample_rng_state']),final_sample_rng_hash=state_hash(rng.getstate())))
    dump(ROOT/'audits/visualization_plan.json',dict(selection='Existing V6.2b visualization(v,q) rule',qps=[0,4,9],comparison_frames=[0,1,31,63],frozen_before_evaluation=True))
    dump(ROOT/'audits/evaluation_frame_conflict.json',dict(status='AWAITING_USER_CHOICE',
        requested_frames=64,actual=[dict(dataset=v['dataset'],name=v['name'],frames=v['frames'],fps=v['rate_accounting_fps']) for v in sources()],
        note='V6.9 VIRAT variable frames and existing FID indices conflict with 64-frame requirement'))
    dump(ROOT/'evaluation/config.json',dict(checkpoints={'B1000':meta}))
    deps={}
    for folder in (V69,V68A,V66,V64,V62,V62B,V4,ENGINE,REPO/'src'):
        paths=folder.rglob('*.py') if folder==REPO/'src' else folder.glob('*.py')
        for p in paths:deps[str(p)]=sha(p)
    for p in (SOURCE,REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',V64/'manifests/vimeo_official_train.json'):deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/source_code_state.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),dependency_hashes=deps,
        configs={str(p):sha(p) for p in (V69/'config.json',V68A/'config.json',V66/'config.json')}))
    dump(ROOT/'audits/old_inventory.json',old_inventory())
    seal();dump(ROOT/'preflight_audit.json',dict(status='PASS',initialization=True,training_plans=True,evaluation_choice_pending=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='implementation_checks'))
    print('SOURCE/PLANS PASS; evaluation frame-count choice pending',flush=True)
if __name__=='__main__':main()
