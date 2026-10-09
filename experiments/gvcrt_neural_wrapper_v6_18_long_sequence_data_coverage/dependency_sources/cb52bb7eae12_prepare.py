"""Freeze isolated V6.17 inputs and inherited V6.16 implementation audits."""
from io17 import *
def main():
    import torch
    assert not (ROOT/'config.json').exists(),'Existing preparation must be resumed, not overwritten'
    assert load(V16/'final_integrity.json')['status']=='PASS'
    cfg=load(V16/'config.json');assert sha(cfg['source_checkpoint']['path'])==EXPECTED
    cfg.update(schema='v617_dists_weight_control',branch='dists_w05',lambda_dists=.5,objective='LPIPS + 0.5*DISTS(require_grad=True) + beta(q)*rate_bpp + 0.01*proxy_L1')
    dump(ROOT/'config.json',cfg)
    for n in ('replay_plan.json','train_manifest_ulong.json','train_manifest_uvg.json','dataset_split.json','qp_semantics_audit.json'):dump(ROOT/n,load(V16/n))
    assert sha(ROOT/'replay_plan.json')==sha(V16/'replay_plan.json')
    for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V16/'manifests'/f'{d}.json'))
    for n in ('replay_integrity.json','implementation.json','loaded_quality_runtime.json','initialization.json','GT_feature_pools.json'):
        dump(ROOT/'audits/inherited'/n,load(V16/'audits'/n))
    dump(ROOT/'audits/inherited/gradient_audit.json',load(V16/'gradient_audit.json'))
    assert load(ROOT/'audits/inherited/gradient_audit.json')['status']=='PASS'
    assert load(ROOT/'audits/inherited/replay_integrity.json')['verified']==1000
    ec=load(V16/'evaluation/config.json');ec.update(schema='v617_evaluation',methods=METHODS,datasets=DATASETS,expected_points=800,no_model_selection=True)
    ec['checkpoints']={k:v for k,v in ec['checkpoints'].items() if k in METHODS[:-1]}
    ec['deployment_receiver_hashes']={k:v for k,v in ec['deployment_receiver_hashes'].items() if k in METHODS[:-1]}
    for cp in ec['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
    dump(ROOT/'evaluation/config.json',ec)
    protocol=load(V16/'protocol.json');protocol.update(methods=METHODS,comparison='five-model common actual bpp interval; 100 uniform ln(bpp); PCHIP without extrapolation',new_branch='dists_w05_1000',evaluation_DISTS='unweighted; unchanged implementation',fixed_evaluation_step=1000)
    dump(ROOT/'protocol.json',protocol)
    deps=load(V16/'audits/dependencies.json')
    for p in [*V16.glob('*.py'),V16/'config.json',V16/'evaluation/config.json',V16/'protocol.json',V16/'final_integrity.json',V16/'replay_plan.json',V16/'checkpoint_index.json',V16/'gradient_audit.json',*(V16/'audits').glob('*.json'),*(V16/'manifests').glob('*.json')]:deps[str(p)]=sha(p)
    pools={}
    for d in DATASETS:
        p=V16/'features'/f'GT_{d}.npy';assert p.exists();deps[str(p)]=sha(p)
        pools[d]=dict(path=str(p),sha256=sha(p),frame_pool=load(V16/'audits/GT_feature_pools.json')[d])
    dump(ROOT/'audits/frozen_GT_pools.json',pools)
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    ps=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'replay_plan.json',ROOT/'qp_semantics_audit.json',ROOT/'dataset_split.json',*ROOT.glob('train_manifest*.json'),*(ROOT/'manifests').glob('*.json')]
    dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
    copies=[V16/n for n in ('train.py','io16.py','adapter.py','report.py','config.json','evaluation/config.json')]
    copies += [V62/'fullqp_train.py',V62/'fullqp_common.py']
    for i,p in enumerate(copies):
        target=ROOT/'dependency_sources'/f'{i:02d}_{p.name}';target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
    dump(ROOT/'preflight_audit.json',dict(status='PASS',source_checkpoint_sha256=EXPECTED,replay_plan_sha256=sha(ROOT/'replay_plan.json'),inherited_gradient_audit_sha256=sha(V16/'gradient_audit.json'),training_changes=['DISTS coefficient 1.0 -> 0.5'],expected_updates=1000,expected_reused_RD=640,expected_new_RD=160,expected_total_RD=800,GT_pools=pools))
    command([PYTHON,'-B','-u',str(Path(__file__).resolve())]);print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
