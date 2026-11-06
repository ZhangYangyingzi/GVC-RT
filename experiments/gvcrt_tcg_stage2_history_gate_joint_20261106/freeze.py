"""Freeze this task's actual sources and already-audited immutable data, never histories."""
from common_s2 import *
def main():
    env();assert not (ROOT/'source_manifest.json').exists(),'SOURCE_ALREADY_FROZEN; explicit versioned repair is required'
    assert not (ROOT/'checkpoints/latest.pt').exists(),'FORMAL_STATE_ALREADY_EXISTS'
    for name in ['initial_joint_load','branch_precision_and_state','gradient_equivalence','matched_M0_equivalence','RANS_identity']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    rows=plan_rows();assert len(rows)==12000
    for i in range(0,len(rows),4):assert sorted(r['domain'] for r in rows[i:i+4])==['ulong','ulong','uvg','uvg']
    assert len(load(ROOT/'vendor/sealed/manifests/B_expanded_ulong.json'))==8470
    quarantine=load(ROOT/'vendor/sealed/graph/quarantine.json');assert len(quarantine['groups'])==236
    for n,h in [('GVC-RT_I.pt','c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'),('GVC-RT_P.pt','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e')]:assert sha(PROJECT/'checkpoints'/n)==h
    assert sha(BASELINE)==BASELINE_SHA and sha(PILOT)==PILOT_SHA and sha(RAFT)==RAFT_SHA
    prior=load(ROOT/'vendor/previous_stage2/source_manifest.json')
    external=dict(prior['inherited_external_sources'])
    external[str(PREVIOUS/'source_manifest.json')]=sha(PREVIOUS/'source_manifest.json')
    external[str(PREVIOUS/'audits/load_probe.json')]=sha(PREVIOUS/'audits/load_probe.json')
    for p,h in external.items():assert sha(p)==h
    manifest=dict(revision='internal_history_gate_joint_sources_v1',time=time.time(),
        code={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts'])},
        protocol_sha256=sha(ROOT/'protocol.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),
        inherited_external_sources=external,immutable_manifests={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'vendor').rglob('*.json'))},
        final_videos_sha256=sha(ROOT/'manifests/final_videos.json'),baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,
        engineering_audits={n:sha(ROOT/f'audits/{n}.json') for n in ['initial_joint_load','branch_precision_and_state','gradient_equivalence','matched_M0_equivalence','RANS_identity']})
    dump(ROOT/'source_manifest.json',manifest);verify_sources()
    dump(ROOT/'audits/data_plan.json',dict(status='PASS',samples=12000,source_cursor_start=1500,source_cursor_at_cap=4500,
        immutable_plan_index_sha256=manifest['plan_index_sha256'],B_pool=8470,quarantined_source_groups=236,
        inherited_window_validation_reused=True,inherited_audit_sha256=sha(ROOT/'vendor/previous_stage2/audits/data_plan.json'),
        no_split_or_window_or_QP_change=True,no_new_independent_sample_claim=True))
    dump(ROOT/'preflight.json',dict(status='ENGINEERING_PASS_DDP_SMOKE_PENDING',physical_GPUs=[4,5,6,7],disk=disk(),
        baseline_sha256=BASELINE_SHA,RAFT_sha256=RAFT_SHA,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        Router_TCG_parameters=171649,Router_initial_M=.2,formal_joint_updates=0))
    print('SOURCE_FREEZE_PASS',sha(ROOT/'source_manifest.json'))
if __name__=='__main__':main()
