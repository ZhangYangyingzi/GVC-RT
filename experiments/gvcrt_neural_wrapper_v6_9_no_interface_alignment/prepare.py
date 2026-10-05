"""Verify baseline raw artifacts and seal an isolated loss-only ablation."""
import copy
from v68_io import *

def main():
    if (ROOT/'preflight_audit.json').exists():
        frozen();print('ALREADY PREPARED');return
    import torch
    torch.set_num_threads(2)
    for folder in ('audits','logs/failures','parts/memory','results','training_logs','evaluation'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    for branch in BRANCHES:
        for folder in ('checkpoints','logs','parts'):
            (ROOT/'branches'/branch/folder).mkdir(parents=True,exist_ok=True)
    cfg=load(V68B/'config.json')
    assert cfg['source_checkpoint']==str(SOURCE) and sha(SOURCE)==cfg['source_checkpoint_sha256']
    state=torch.load(SOURCE,map_location='cpu',weights_only=True)
    source_audit=load(V68B/'audits/source_B1000_audit.json')
    assert state['step']==1000 and source_audit['sha256']==sha(SOURCE)
    assert {key:tensor_hash(state[key]) for key in ('wrapper','bridge','generator')}==source_audit['module_hashes']
    assert optimizer_summary(state['optimizer'])==source_audit['optimizer']
    assert {key:state_hash(state[key]) for key in ('sample_rng_state','torch_rng_state','cuda_rng_state')}==source_audit['rng_hashes']
    assert state['lambda_struct']==cfg['lambda_struct']
    dump(ROOT/'audits/source_B1000_audit.json',source_audit)
    original_cfg=copy.deepcopy(cfg)
    cfg.update(schema='v69_no_interface_alignment',branches=list(BRANCHES),methods=list(METHODS),
               branch_generator_lr=LRS,alignment_mode=MODE,alignment_loss_present=False,
               training_objective='L_B only; exact V6.6 B binder',baseline_experiment=str(V68B))
    dump(ROOT/'config.json',cfg)
    cps=load(V68B/'evaluation/config.json')['checkpoints']
    evaluation=load(V68B/'evaluation/config.json');evaluation['checkpoints']={m:cps[m] for m in REUSED}
    for m,cp in evaluation['checkpoints'].items():assert sha(cp['path'])==cp['sha256']
    dump(ROOT/'evaluation/config.json',evaluation)
    names=['source_manifest.json','qp_semantics_audit.json']
    names+=['audits/'+name for name in ('continuation_training_plans.json','fid_protocol_audit.json',
             'GT_feature_cache.json','vimeo_heldout_manifest.json',
             *[f'fid_bootstrap_indices_{d}.json' for d in DATASETS])]
    for name in names:
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((V68B/name).read_bytes());assert sha(target)==sha(V68B/name)
    plans=load(ROOT/'audits/continuation_training_plans.json')['plans']
    assert len(plans)==500 and [p['absolute_step'] for p in plans]==list(range(1001,1501))
    assert len(sources())==31
    for branch in BRANCHES:
        old='G'+tag(branch)[1:]+'_interface_preserve'
        assert LRS[branch]==original_cfg['branch_generator_lr'][old]
        branch_cfg=copy.deepcopy(cfg);branch_cfg.update(branch=branch,matched_baseline_branch=old,
            effective_generator_lr=LRS[branch],optimizer_state_restored=True)
        dump(ROOT/'branches'/branch/'training_config.json',branch_cfg)
    # Only completed, individually checked baseline artifacts are reused. Its
    # failed final reporting status is recorded, not treated as a success gate.
    reused_points=reused_fid=reused_heldout=0
    baseline_records=[]
    for method in REUSED:
        for dataset in DATASETS:
            for video in sources(dataset):
                for qp in range(10):
                    origin=V68B/'parts'/dataset/method/f'video_{video["video_index"]:02d}_qp{qp}.json'
                    record=load(origin)
                    record.update(reused=True,reused_from=str(origin),reused_point_sha256=sha(origin))
                    validate_point(record,video,qp,method)
                    dump(point(dataset,method,video['video_index'],qp),record);reused_points+=1
            for qp in range(10):
                origin=V68B/'parts/fid'/dataset/method/f'qp{qp}.json';record=load(origin)
                assert record['status']=='PASS' and sha(record['bootstrap_path'])==record['bootstrap_sha256']
                for pt in record['source_points']:assert sha(pt['path'])==pt['sha256']
                assert math_finite(record['FID'])
                dump(ROOT/'parts/fid'/dataset/method/origin.name,record);reused_fid+=1
        for origin in sorted((V68B/'parts/heldout'/method).glob('video_*_qp*.json')):
            record=load(origin)
            for key in ('bitstream','feature','frame_metrics','transitions'):
                assert sha(record[key+'_path'])==record[key+'_sha256']
            assert record['status']=='PASS' and record['checkpoint_sha256']==cps[method]['sha256']
            record.update(reused=True,reused_from=str(origin),reused_point_sha256=sha(origin))
            dump(ROOT/'parts/heldout'/method/origin.name,record);reused_heldout+=1
    assert (reused_points,reused_fid,reused_heldout)==(1550,200,480)
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',RD_points=reused_points,
         FID_points=reused_fid,heldout_points=reused_heldout,
         original_pipeline_status=load(V68B/'pipeline_status.json')['status'],
         original_final_integrity=load(V68B/'final_integrity.json'),raw_artifacts_verified=True))
    dump(ROOT/'audits/controlled_difference.json',dict(status='PASS',source_checkpoint_sha256=sha(SOURCE),
         initialization='same B1000 tensors, optimizer moments and RNG states',
         same_compression_hash=cfg['compression_hash'],same_architectures=True,
         same_training_plan_sha256=sha(ROOT/'audits/continuation_training_plans.json'),
         same_optimizer_and_learning_rates=True,same_bit_rate_and_qp_protocol=True,
         same_datasets_and_metric_implementations=True,only_training_difference='remove L_align computation entirely',
         wrapper_trainable=True,bridge_trainable=True,generator_trainable=True,compression_core_frozen=True))
    deps=load(V68B/'audits/dependencies.json')
    for p in [SOURCE,*V68B.glob('*.py'),V68B/'config.json',*[V68B/name for name in names]]:
        deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/old_inventory.json',old_inventory())
    dump(ROOT/'preflight_audit.json',dict(status='PASS',plans=500,formal_sequences=31,
         fresh_RD_target=1240,total_RD_target=2790,fresh_FID_target=160,total_FID_target=360,
         fresh_heldout_target=384,total_heldout_target=864))
    fixed=[*ROOT.glob('*.py'),ROOT/'config.json',*[ROOT/name for name in names]]
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in fixed})
    dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='objective_smoke'))
    print('PREFLIGHT PASS: source/config/plans verified; baseline raw artifacts reused',flush=True)

def math_finite(x):
    import math
    return math.isfinite(x)

if __name__=='__main__':main()
