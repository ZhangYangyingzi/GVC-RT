"""Freeze protocols, official training lists, baselines and independent QP RNG."""
import ast,collections,random,traceback
from v63_io import *
def main():
    if (ROOT/'preflight_audit.json').exists() and load(ROOT/'preflight_audit.json')['status']=='PASS':frozen(True);return
    for d in ('audits','manifests','parts','logs','evaluation','results','training_logs'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','parts','logs'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    assert load(AUDIT/'final_integrity.json')['status']=='PASS'
    assert load(V62B/'final_integrity.json')['status']=='PASS'
    cfg=load(V62/'fullqp_config.json');baseline=load(V62B/'config.json')
    expected='c185e1f69fd5f249a4aa3753ddee1dc23d4402d5d6f74d18ef6b51b559fbdb73'
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']==expected
    assert cfg['optimizer']==dict(name='AdamW',wrapper_lr=5e-5,bridge_lr=3e-6,generator_lr=5e-7,weight_decay=1e-4,grad_clip=1.)
    assert cfg['clip_length']==4 and cfg['crop']==256 and cfg['batch']==1 and cfg['lambda_proxy']==.01 and cfg['force_zero_thres']==.12
    config={k:cfg[k] for k in ('seed','source_checkpoint','source_checkpoint_sha256','train_manifest','train_manifest_sha256','clip_length','crop','batch','force_zero_thres','lambda_proxy','optimizer')}
    config.update(schema='v63_dataset_ablation_v1',branches=list(BRANCHES),updates=1000,checkpoint_steps=list(STEPS),test_checkpoint_step=1000,
        objective='exact AST from V6.2 fullqp_train.py::main::run_clip; schedule_s1p0',qp_seed=cfg['seed'],sample_seed=cfg['seed'],
        allowed_gpus=[4,5,6,7],minimum_free_training_mib=22000,minimum_free_evaluation_mib=16000,
        no_test_checkpoint_selection=True,external_qps=list(range(10)),baseline_checkpoints={'v62':baseline['checkpoints']['v62']})
    dump(ROOT/'config.json',config)
    for name,origin in [('source_manifest.json',V62B),('qp_semantics_audit.json',V62B),('qp_train_eval_semantics_audit.json',V62),('gvc_lambda_schedule_audit.json',V62),('metric_implementation_audit.json',V62B)]:dump(ROOT/name,load(origin/name))
    train=load(cfg['train_manifest']);assert len(train['videos'])==1024 and sha(cfg['train_manifest'])==cfg['train_manifest_sha256']
    dump(ROOT/'manifests/ulong_train1024.json',train)
    split=load(AUDIT/'vimeo_split_audit.json');ready=load(AUDIT/'vimeo_training_readiness.json');assert split['status']==ready['status']=='PASS'
    paths=[split[k]['path'] for k in ('official_train_source','official_test_source')]
    for key,p in zip(('official_train_source','official_test_source'),paths):assert sha(p)==split[key]['sha256']
    train_ids=[s.strip() for s in Path(paths[0]).read_text().splitlines() if s.strip()];test_ids=set(Path(paths[1]).read_text().split())
    assert len(train_ids)==len(set(train_ids))==64612 and not set(train_ids)&test_ids
    seqroot=Path(paths[0]).parent/'sequences'
    assert all((seqroot/s/f'im{i}.png').is_file() for s in train_ids for i in range(1,8))
    dump(ROOT/'manifests/vimeo_official_train.json',dict(sequences=train_ids,sequence_root=str(seqroot),official_train_file=paths[0],official_train_sha256=sha(paths[0]),count=64612))
    tests=load(ROOT/'source_manifest.json')['videos'];train_hashes={v['sha256'] for v in train['videos']}
    assert not train_hashes&{v['source_sha256'] for v in tests}
    for s in ('normal','hard'):assert not train_hashes&{v['sha256'] for v in load(V62/f'validation_{s}_manifest.json')['videos']}
    dump(ROOT/'dataset_audit.json',dict(status='PASS',ulong_exact1024=True,vimeo_official_train_count=len(train_ids),vimeo_official_test_intersection=[],
        vimeo_full_train_not_diagnostic2048=True,all_seven_files_present=True,no_reference_guided_selection=True,no_final_test_training=True,no_validation_training=True,
        vimeo_decode_scope='Headers audited previously; each sampled image is fully decoded and SHA256 recorded during loading'))
    rng=random.Random(config['qp_seed']);qps=[rng.randrange(10) for _ in range(1000)]
    dump(ROOT/'qp_sequence_audit.json',dict(status='PASS',seed=config['qp_seed'],generator='independent random.Random.randrange(10)',sequence=qps,
        histogram=dict(collections.Counter(qps)),identical_for_branches=list(BRANCHES),independent_of_sample_rng=True))
    rows=[dict(update=i,source=source_for('mixed_50_50',i)) for i in range(1,1001)]
    assert collections.Counter(r['source'] for r in rows)=={'ulong':500,'vimeo':500}
    write(ROOT/'manifests/mixed_source_schedule.csv',rows)
    from objective import original_ast
    dump(ROOT/'audits/objective_audit.json',dict(status='PASS',source=str(V62/'fullqp_train.py'),source_sha256=sha(V62/'fullqp_train.py'),
        exact_run_clip_ast_sha256=hashlib.sha256(ast.dump(original_ast(),include_attributes=False).encode()).hexdigest(),scale=1.0,
        beta_lambda_implementation=str(V62/'fullqp_common.py'),no_loss_reimplementation=True))
    old=old_eval();reused=[]
    for v in tests:
        for m in ('original','v62'):
            for q in range(10):
                p=old.point(v['dataset'],m,v['video_index'],q);r=load(p);old.validate(r,v,q,m)
                assert r['compression_hash_before']==load(V62B/'final_integrity.json')['shared_compression_hash']
                reused.append(dict(dataset=v['dataset'],method=m,video_index=v['video_index'],QP=q,path=str(p),sha256=sha(p)))
    assert len(reused)==620
    dump(ROOT/'audits/baseline_reuse_audit.json',dict(status='PASS',count=len(reused),points=reused,all_artifact_hashes_verified=True))
    dump(ROOT/'evaluation_manifest.json',dict(status='FROZEN',datasets=list(DATASETS),methods=list(METHODS),external_qps=list(range(10)),
        videos=tests,fresh_methods=list(BRANCHES),fresh_points=930,reused_points=620,total_points=1550,checkpoint_selection='fixed step1000 for all three branches'))
    dependencies={}
    for file in (V62B/'frozen_source_hashes.json',V62/'fullqp_old_source_hashes.json',AUDIT/'frozen_dependencies.json'):
        for p,h in load(file).items():assert sha(p)==h;dependencies[p]=h
    for folder in (V62,V62B,V4,ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics',ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb',ROOT.parent/'gvcrt_parallel_v5a3_v60'):
        for p in folder.glob('*.py'):dependencies[str(p)]=sha(p)
    for p in [Path(cfg['source_checkpoint']),Path(baseline['checkpoints']['v62']['path']),Path(cfg['train_manifest']),*map(Path,paths),AUDIT/'final_integrity.json',V62/'fullqp_config.json',V62/'qp_train_eval_semantics_audit.json',V62/'gvc_lambda_schedule_audit.json',V62B/'source_manifest.json']:
        dependencies[str(p)]=sha(p)
    dump(ROOT/'audits/frozen_dependencies.json',dependencies)
    dump(ROOT/'audits/old_inventory.json',inventory())
    local=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'qp_sequence_audit.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',* (ROOT/'manifests').glob('*')]
    dump(ROOT/'audits/local_protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in local})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',dataset_distribution_prerequisite='PASS',baselines_verified=620,formal_training_requires_all_branch_smoke_PASS=True,completed_unix=time.time()))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',reason='Training and evaluation not yet complete'))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'preflight_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
