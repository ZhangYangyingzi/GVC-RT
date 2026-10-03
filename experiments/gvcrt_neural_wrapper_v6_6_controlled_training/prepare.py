"""Freeze baseline, source checkpoint, sequence equivalence and dependencies."""
import random
from v66_io import *
def main():
    if (ROOT/'preflight_audit.json').exists():frozen();return
    for d in ('audits','logs','parts','training_logs','results','evaluation'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','logs','parts'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    cfg=load(V64/'config.json');assert load(V64/'final_integrity.json')['status']=='PASS'
    cp=load(V64/'checkpoint_hashes.json')['vimeo_only']
    assert sha(cp['path'])==cp['sha256']=='1c7c9a3f89cbbe4abd4ba6f6f7c963d2fab22737e9928fe222f3aa4c8a8c75e1'
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']=='c185e1f69fd5f249a4aa3753ddee1dc23d4402d5d6f74d18ef6b51b559fbdb73'
    cfg.update(schema='v66_controlled_v1',branches=list(BRANCHES),baseline=cp,gpu_policy='GPU 4/5/6/7 sharing allowed by latest user instruction; memory thresholds; never stop other processes',calibration_clips=32,calibration_target_ratio=.10)
    dump(ROOT/'config.json',cfg)
    for name in ('source_manifest.json','qp_semantics_audit.json','qp_sequence_audit.json','qp_train_eval_semantics_audit.json'):dump(ROOT/name,load(V64/name))
    init=load(V64/'initialization_audit.json')['branches']['vimeo_only']
    assert init['source_checkpoint_sha256']==cfg['source_checkpoint_sha256']
    dump(ROOT/'audits/source_checkpoint_audit.json',dict(status='PASS',source_path=cfg['source_checkpoint'],source_sha256=cfg['source_checkpoint_sha256'],A_checkpoint=cp,initialization=init,new_training_from_A=False))
    records=[json.loads(line) for line in (V64/'training_logs/vimeo_only.jsonl').read_text().splitlines()];assert len(records)==1000
    manifest=load(V64/'manifests/vimeo_official_train.json');rng=random.Random(cfg['sample_seed']);qr=random.Random(cfg['qp_seed']);qps=load(V64/'qp_sequence_audit.json')['sequence'];plans=[]
    for i,r in enumerate(records):
        plan=dict(update=i+1,sample_id=rng.choice(manifest['sequences']),start=rng.randrange(4),crop_x=rng.randrange(193),crop_y=rng.randrange(1),external_qp=qr.randrange(10))
        assert all(r[k]==z for k,z in plan.items()) and plan['external_qp']==qps[i]
        plans.append(plan)
    dump(ROOT/'audits/training_sequence_equivalence.json',dict(status='PASS',updates=1000,branches=list(BRANCHES),reference=str(V64/'training_logs/vimeo_only.jsonl'),reference_sha256=sha(V64/'training_logs/vimeo_only.jsonl'),sample_sequence_identical=True,qp_sequence_identical=True,plans=plans,formal_logs_rechecked_after_training=True))
    deps=load(V64/'audits/frozen_dependencies.json')
    for folder in (V64,V65,V62B,V4,ENGINE):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in [Path(cfg['source_checkpoint']),Path(cp['path']),V64/'training_logs/vimeo_only.jsonl',V64/'manifests/vimeo_official_train.json',V64/'source_manifest.json',V64/'qp_sequence_audit.json',V64/'config.json',V64/'initialization_audit.json',V65/'final_integrity.json']:deps[str(p)]=sha(p)
    for p,h in deps.items():assert sha(p)==h
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    dump(ROOT/'audits/old_write_policy.json',dict(write_root=str(ROOT),old_completed_experiments='size and mtime snapshot plus dependency SHA256',concurrently_running_experiment=str(ACTIVE_OLD),concurrent_policy='Independent active experiment excluded: its sources and outputs are changing externally; not a V6.6 dependency; no task code writes outside new root'))
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_sequence_audit.json',ROOT/'qp_semantics_audit.json']})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',same_initialization=True,same_sequence=True,A_checkpoint_hash_pass=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING'))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
