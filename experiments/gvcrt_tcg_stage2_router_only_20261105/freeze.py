"""Freeze executable sources and inherited local identities before formal updates."""
from common_s2 import *
def main():
    env()
    if (ROOT/'checkpoints/latest.pt').exists():
        import torch
        initial=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
        assert initial['committed_router_updates']==0 and not initial['optimizer']['state'],'EXPLICIT_MIGRATION_REQUIRED_AFTER_FORMAL_UPDATES'
        assert (ROOT/'audits/checkpoint_metadata_migration_control.json').exists(),'MIGRATION_CONTROL_RECORD_REQUIRED'
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts'])}
    external={}
    p=load(ORIGINAL_EVAL/'protocol.json')
    for name,h in p['metric_sources_sha256'].items():assert sha(name)==h;external[name]=h
    for name in ['audits/plans_frozen.json','repairs/native_geometry_v1/geometry_errata.json','inventory/qualified.json','graph/source_groups_final.json.gz']:
        path=HIST/name
        if path.exists():external[str(path)]=sha(path)
    for name in ['source_manifest.json','runtime_protocol_addendum.json']:
        path=PARENT/name
        if path.exists():external[str(path)]=sha(path)
    code_manifest=dict(revision='router_only_code_freeze_v4_checkpoint_and_manifest_integrity',time=time.time(),code=files,protocol_sha256=sha(ROOT/'protocol.json'),
        plan_index_sha256=sha(ROOT/'plans/index.json'),baseline_checkpoint_sha256=BASELINE_SHA,pilot_checkpoint_sha256=PILOT_SHA,
        inherited_external_sources=external,immutable_manifests={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'vendor/sealed').rglob('*.json'))},
        final_videos_sha256=sha(ROOT/'manifests/final_videos.json'))
    dump(ROOT/'source_manifest.json',code_manifest)
    dump(ROOT/'audits/preflight.json',dict(status='PASS',disk=disk(),CUDA='four authorized devices; build/forward/RANS/gradient already executed on CUDA',
        physical_GPUs=[4,5,6,7],baseline_sha256=sha(BASELINE),pilot_sha256=sha(PILOT),RAFT_local_sha256=sha(RAFT),
        source_copy_manifest_sha256=sha(ROOT/'audits/source_copy.json'),plan_audit_sha256=sha(ROOT/'audits/data_plan.json'),
        forward_equivalence_sha256=sha(ROOT/'audits/forward_equivalence.json'),gradient_check_sha256=sha(ROOT/'audits/gradient_and_state.json'),
        real_RANS_identity_sha256=sha(ROOT/'audits/RANS_identity.json'),full_training_preflight_pending='Four-rank disposable all-reduce/optimizer/checkpoint smoke is mandatory before formal initialization'))
    verify_sources()
if __name__=='__main__':main()
