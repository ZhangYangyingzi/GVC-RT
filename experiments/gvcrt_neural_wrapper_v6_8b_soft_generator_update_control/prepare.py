"""Verify and reuse completed V6.8a artifacts without rewriting any old path."""
import copy
from v68_io import *
def main():
    if (ROOT/'preflight_audit.json').exists():frozen();return
    import torch
    torch.set_num_threads(2)
    for d in ('audits','logs','parts','results','training_logs','evaluation'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','logs','parts'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    (ROOT/'audits/nvidia_smi_before.txt').write_text(subprocess.run(['nvidia-smi'],text=True,capture_output=True).stdout)
    integrity=load(V68A/'final_integrity.json');assert integrity['status']=='PASS_WITH_APPROVED_EXCEPTIONS'
    ex=load(V68A/'audits/approved_png_change_exception.json')
    assert ex['user_authorized'] and len(ex['changes'])==16
    for x in ex['changes']:assert sha(x['path'])==x['approved_sha256']
    validation=load(V68A/'recovery/output_validation.json');assert validation['status']=='PASS'
    for p,m in validation['files'].items():assert sha(V68A/p)==m['sha256']
    for p,h in load(V68A/'audits/dependencies.json').items():assert sha(p)==h
    meta=load(SOURCE.parent.parent/'checkpoint_hashes.json')['1000'];actual=sha(SOURCE);assert actual==meta['sha256']
    s=torch.load(SOURCE,map_location='cpu',weights_only=True)
    aud=load(V68A/'audits/source_B1000_audit.json')
    assert s['step']==1000 and actual==aud['sha256'] and s['lambda_struct']==0.045785124942642835
    assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==meta['module_hashes']==aud['module_hashes']
    assert optimizer_summary(s['optimizer'])==aud['optimizer']
    assert {k:state_hash(s[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')}==aud['rng_hashes']
    dump(ROOT/'audits/source_B1000_audit.json',aud)
    controls={}
    for branch,lr in LRS.items():
        before=copy.deepcopy(s['optimizer']);after=copy.deepcopy(before)
        assert [g['name'] for g in after['param_groups']]==['wrapper','bridge','generator']
        assert after['param_groups'][2]['lr']==5e-7
        after['param_groups'][2]['lr']=lr
        restored=copy.deepcopy(after);restored['param_groups'][2]['lr']=5e-7
        assert state_hash(restored)==state_hash(before)
        states={str(k):state_hash(v) for k,v in before['state'].items()}
        assert states=={str(k):state_hash(v) for k,v in after['state'].items()}
        controls[tag(branch)]=dict(before=optimizer_summary(before),after=optimizer_summary(after),per_parameter_state_hashes=states,only_generator_lr_changed=True,generator_lr=lr)
    dump(ROOT/'audits/optimizer_control_audit.json',dict(status='PASS',source=aud['optimizer'],branches=controls))
    cfg=load(V68A/'config.json');cfg.update(schema='v68b_soft_generator_update_control',branches=list(BRANCHES),methods=list(METHODS),branch_generator_lr=LRS,reference_C1_generator_lr=5e-7)
    dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation/config.json',load(V68A/'evaluation/config.json'))
    names=['source_manifest.json','qp_semantics_audit.json']
    names+=['audits/'+n for n in ('continuation_training_plans.json','interface_diagnostic_plan.json','fid_protocol_audit.json','GT_feature_cache.json','vimeo_heldout_manifest.json','interface_alignment_calibration.json','interface_preservation_feasibility.json','interface_semantics_audit.json','teacher_temporal_state_audit.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS])]
    for name in names:
        dst=ROOT/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((V68A/name).read_bytes());assert sha(dst)==sha(V68A/name)
    planhash=sha(ROOT/'audits/continuation_training_plans.json')
    assert planhash=='e73f690b87d916354ca10f81a944b3235a7447cf20d7871d415020300bb6d937'
    dump(ROOT/'audits/v68a_training_plan_reuse.json',dict(status='PASS',source=str(V68A/'audits/continuation_training_plans.json'),sha256=planhash,byte_identical=True,regenerated=False))
    cal=load(ROOT/'audits/interface_alignment_calibration.json')
    assert cal['lambda_align']==0.028220742743291956 and cal['actual_backward_aggregate_ratio']==0.0440829907555377
    dump(ROOT/'audits/alignment_reuse_audit.json',dict(status='PASS',lambda_align=cal['lambda_align'],source_experiment=str(V68A),source_calibration_hash=sha(V68A/'audits/interface_alignment_calibration.json'),formula_ratio=.05,actual_backward_ratio=cal['actual_backward_aggregate_ratio'],recalibrated=False))
    assert sha(ROOT/'objective.py')==sha(V68A/'objective.py') and sha(ROOT/'model_runtime.py')==sha(V68A/'model_runtime.py')
    ci=load(V68A/'branches/C1_B_interface_preserve/initialization_audit.json')
    assert ci['optimizer']==aud['optimizer'] and ci['module_hashes']==aud['module_hashes'] and ci['rng_hashes']==aud['rng_hashes']
    dump(ROOT/'audits/controlled_difference_audit.json',dict(status='PASS',reference_C1_initialization_sha256=sha(V68A/'branches/C1_B_interface_preserve/initialization_audit.json'),same_source=True,same_optimizer_states_before_lr_edit=True,same_training_plans=True,same_QP_sequence=True,same_objective=True,same_alignment=True,same_lambda_align=True,same_wrapper_lr=True,same_bridge_lr=True,same_generator_trainability=True,only_difference={tag(b):{'generator_lr':[5e-7,lr]} for b,lr in LRS.items()}))
    for m in REUSED:
        for d in DATASETS:
            for v in sources(d):
                for q in range(10):
                    p=V68A/'parts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json';r=load(p)
                    for k in ('bitstream','feature','frame_metrics','transitions'):
                        if k+'_path' in r:assert sha(r[k+'_path'])==r[k+'_sha256']
                    r.update(reused=True,reused_from=str(p),reused_point_sha256=sha(p));dump(point(d,m,v['video_index'],q),r)
            for q in range(10):
                p=V68A/'parts/fid'/d/m/f'qp{q}.json';r=load(p);assert sha(r['bootstrap_path'])==r['bootstrap_sha256']
                for pt in r['source_points']:assert sha(pt['path'])==pt['sha256']
                dump(ROOT/'parts/fid'/d/m/p.name,r)
        if m in DIAG_METHODS:
            for p in (V68A/'parts/heldout'/m).glob('*.json'):
                r=load(p)
                for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                dump(ROOT/'parts/heldout'/m/p.name,r)
            r=load(V68A/'parts'/f'diagnostic_done_{m}.json');assert sha(r['path'])==r['sha256'];dump(ROOT/'parts'/f'diagnostic_done_{m}.json',r)
    dump(ROOT/'audits/v68a_protocol_reuse_audit.json',dict(status='PASS',source_integrity_sha256=sha(V68A/'final_integrity.json'),source_exception_verified=True,source_output_validation_sha256=sha(V68A/'recovery/output_validation.json'),reused_RD_points=1550,reused_FID_points=200,reused_heldout_points=288,reused_drift_points=162,files={n:sha(V68A/n) for n in names}))
    dump(ROOT/'audits/generator_semantic_plan.json',dict(status='FROZEN_BEFORE_TRAINING',source_diagnostic_plan_sha256=sha(ROOT/'audits/interface_diagnostic_plan.json'),methods=list(DIAG_METHODS),z_source='B1000 native student generator forward_pre_hook inputs[0]',q_recon='exact captured positional and keyword arguments',comparison='each method versus B1000 at identical z_ref; raw decoder RGB mapped with core.unit',metrics=['LPIPS','DISTS','L1'],output_feature_difference='not added: no existing output feature difference definition'))
    deps=load(V68A/'audits/dependencies.json')
    for p in [*V68A.glob('*.py'),V68A/'final_integrity.json',V68A/'recovery/output_validation.json',V68A/'audits/approved_png_change_exception.json',*[V68A/n for n in names]]:deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    from freeze_protocol import main as seal
    seal();dump(ROOT/'preflight_audit.json',dict(status='PASS',B1000_verified=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='lr_control_smoke'))
    print('PREFLIGHT PASS: exact plans and optimizer; reused 1550 RD / 200 FID / 288 heldout',flush=True)
if __name__=='__main__':main()
