"""Copy the immutable v6.22 plan, with fixed weights and no calibration."""
from io23 import *
import shutil,collections

def copy_exact(name,records):
    source=V22/name;dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(source,dest)
    assert sha(source)==sha(dest)
    records.append(dict(path=name,source=str(source),source_sha256=sha(source),copied_sha256=sha(dest)))

def main():
    command([PYTHON,'-B',str(Path(__file__).resolve())])
    import torch
    assert load(V22/'final_integrity.json')['status']=='PASS'
    oldcfg=load(V22/'config.json');cp=oldcfg['source_checkpoint']
    assert cp['sha256']==EXPECTED and sha(cp['path'])==EXPECTED
    s=torch.load(cp['path'],map_location='cpu',weights_only=True)
    assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes']
    assert s['compression_hash']==cp['compression_hash'];del s
    cfg=dict(oldcfg,schema='v623',branches=BRANCHES,methods=METHODS,expected_points=800,
        lambda_rec=None,branch_lambda_rec=WEIGHTS,reference_lambda_rec=REFERENCE_WEIGHT,
        training_plan_reused=True,training_plan_source=str(V22/'plans/shared.json'),
        reconstruction_unit_gradient_scale=4096.,i_frame_modes={m:'native_RGB_bypass' for m in EVAL_METHODS})
    cfg.pop('calibration_seed',None)
    if (ROOT/'config.json').exists():assert load(ROOT/'config.json')==json.loads(json.dumps(cfg))
    else:dump(ROOT/'config.json',cfg)
    names=['plans/shared.json','sampling_order.json','train_manifest_ulong.json','train_manifest_uvg.json',
        'dataset_split.json','inherited_exclusions.json','qp_semantics_audit.json',
        *[f'manifests/{d}.json' for d in DATASETS],
        *['audits/'+n for n in ('forbidden_content_windows.json','pool_filter.json','historical_exposure.json')]]
    records=[]
    for n in names:copy_exact(n,records)
    plan=load(ROOT/'plans/shared.json')['updates'];assert len(plan)==1000
    counts=collections.Counter(r['domain'] for r in plan);assert counts=={'ulong':500,'uvg':500}
    assert [r['adaptation_step'] for r in plan]==list(range(1,1001))
    for r in plan:
        assert len(r['canonical_indices'])==len(r['cropped_RGB_sha256'])==len(r['full_RGB_sha256'])==64
        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(r['external_qp'])][49:64]
    dump(ROOT/'audits/inherited_artifacts.json',dict(status='PASS',records=records,domain_counts=counts,plan_regenerated=False,source_exclusions_unchanged=True))
    old=load(V22/'evaluation/config.json')
    ev=dict(old,schema='v623',methods=METHODS,expected_points=800,IPs=IPS,
        checkpoints={m:old['checkpoints'][m] for m in METHODS[1:3]},
        deployment_receiver_hashes={m:old['deployment_receiver_hashes'][m] for m in METHODS[1:3]})
    if not (ROOT/'evaluation/config.json').exists():dump(ROOT/'evaluation/config.json',ev)
    proto=load(V22/'protocol.json');proto.update(schema='v623',methods=METHODS,expected_points=800,source_protocol_sha256=sha(V22/'protocol.json'))
    proto['training'].update(branches=BRANCHES,lambda_rec=None,branch_lambda_rec=WEIGHTS,reference_lambda_rec=REFERENCE_WEIGHT,plan_source=str(V22/'plans/shared.json'),plan_copied_byte_exact=True)
    proto['training'].pop('calibration_plan_sha256',None);proto['training'].pop('weighted_reconstruction_backward_scale',None)
    if (ROOT/'protocol.json').exists():assert load(ROOT/'protocol.json')==json.loads(json.dumps(proto))
    else:dump(ROOT/'protocol.json',proto)
    gts=load(V22/'audits/GT_feature_pools.json')
    for d,g in gts.items():assert sha(g['path'])==g['sha256']
    dump(ROOT/'audits/inherited_GT_feature_pools.json',dict(status='PASS',source=str(V22/'audits/GT_feature_pools.json'),source_sha256=sha(V22/'audits/GT_feature_pools.json'),pools=gts))
    deps=load(V22/'audits/dependencies.json')
    for p in [*V22.glob('*.py'),V22/'config.json',V22/'protocol.json',V22/'evaluation/config.json',V22/'plans/shared.json',Path(cp['path']),*[Path(g['path']) for g in gts.values()]]:
        deps[str(p)]=sha(p)
    for m in METHODS[1:3]:
        c=ev['checkpoints'][m];assert sha(c['path'])==c['sha256']
        z=torch.load(c['path'],map_location='cpu',weights_only=True);assert {k:tensor_hash(z[k]) for k in ('wrapper','bridge','generator')}==c['module_hashes'];del z
        deps[c['path']]=c['sha256']
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',source=cp,expected_SHA256=EXPECTED,actual_SHA256=sha(cp['path']),PBG_verified=True,compression_verified=True))
    if not (ROOT/'audits/historical_inventory.json').exists():dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    dump(ROOT/'audits/protocol_hashes.json',{n:sha(ROOT/n) for n in ['config.json','protocol.json',*names]})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',source_verified=True,plan_sha256=sha(ROOT/'plans/shared.json'),plan_copied_not_regenerated=True,domain_counts=counts,source_seed=oldcfg['seed'],test_exclusions_inherited=True))
    if not (ROOT/'pipeline_status.json').exists():dump(ROOT/'pipeline_status.json',dict(status='READY',phase='weight_check',active={},training_steps={b:0 for b in BRANCHES},completed_RD=0,expected_RD=800))
    print('PREPARATION PASS',flush=True)
if __name__=='__main__':main()
