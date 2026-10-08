from io16 import *
def main():
    import importlib.metadata,torch
    from DISTS_pytorch import DISTS
    assert not (ROOT/'config.json').exists()
    cfg=load(V15/'config.json');assert sha(cfg['source_checkpoint']['path'])==EXPECTED
    cfg.update(schema='v616_dists_gradient_verification',updates=1000,primary_checkpoint=1000,checkpoint_steps=[0,250,500,1000],fixed_report_steps=[1000],objective='historical v6.15 objective with only explicit DISTS require_grad=True, conditional on probe gate')
    dump(ROOT/'config.json',cfg);dump(ROOT/'qp_semantics_audit.json',load(V15/'qp_semantics_audit.json'))
    for old,new in [('train_manifest_ulong.json','train_manifest_ulong.json'),('train_manifest.json','train_manifest_uvg.json')]:dump(ROOT/new,load(V15/old))
    rows=[json.loads(x) for x in (V15/'training_logs/mixed.jsonl').read_text().splitlines()][:1000];assert len(rows)==1000 and [r['adaptation_step'] for r in rows]==list(range(1,1001))
    dump(ROOT/'replay_plan.json',dict(source=str(V15/'training_logs/mixed.jsonl'),source_sha256=sha(V15/'training_logs/mixed.jsonl'),updates=rows))
    dump(ROOT/'dataset_split.json',load(V15/'dataset_split.json'))
    for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V15/'manifests'/f'{d}.json'))
    ec=load(V15/'evaluation/config.json');ec['checkpoints']={k:v for k,v in ec['checkpoints'].items() if k in METHODS};ec.update(methods=METHODS,datasets=DATASETS,expected_points=640)
    dump(ROOT/'evaluation/config.json',ec)
    protocol=load(V15/'protocol.json');protocol.update(methods=METHODS,comparison='four fixed model common actual bpp interval; 100 uniform ln(bpp); PCHIP without extrapolation',fixed_report_steps=[1000]);dump(ROOT/'protocol.json',protocol)
    deps=load(V15/'audits/dependencies.json')
    for p in [*V15.glob('*.py'),V15/'config.json',V15/'training_logs/mixed.jsonl',V15/'checkpoint_index.json',V15/'evaluation/config.json',V15/'final_integrity.json',Path(inspect.getfile(DISTS))]:deps[str(p)]=sha(p)
    for d in DATASETS:deps[str(V15/'manifests'/f'{d}.json')]=sha(V15/'manifests'/f'{d}.json')
    snapshots=[]
    files=[V15/'train.py',V62/'fullqp_train.py',V62/'fullqp_common.py',V4/'train.py',ROOT.parent/'gvcrt_neural_wrapper_v2_joint/core.py',Path(inspect.getfile(DISTS))]
    for i,p in enumerate(files):
        target=ROOT/'dependency_sources'/f'{i:02d}_{p.name}';target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes());snapshots.append(dict(path=str(p),sha256=sha(p),snapshot=str(target.relative_to(ROOT))))
    dump(ROOT/'audits/implementation.json',dict(status='PASS',sources=snapshots,DISTS_class_file=inspect.getfile(DISTS),forward_signature=str(inspect.signature(DISTS.forward)),forward_source=inspect.getsource(DISTS.forward),forward_once_source=inspect.getsource(DISTS.forward_once),package_versions={k:importlib.metadata.version(k) for k in ['DISTS-pytorch','torch','torchvision','lpips']},historical_call='quality[1](output,target).mean()',fixed_call='quality[1](output,target,require_grad=True).mean()',default_require_grad=inspect.signature(DISTS.forward).parameters['require_grad'].default))
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    ps=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'replay_plan.json',ROOT/'qp_semantics_audit.json',ROOT/'dataset_split.json',*ROOT.glob('train_manifest*.json'),*(ROOT/'manifests').glob('*.json')]
    dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',exact_replay_source=sha(V15/'training_logs/mixed.jsonl'),updates=1000,no_new_sampling=True))
    print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
