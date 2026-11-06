"""Freeze exact sources, corrected protocol, sealed split/plan and stream identities."""
from common_s2 import *

def main():
    env();assert not (ROOT/'checkpoints/latest.pt').exists(),'FORMAL_CHECKPOINT_EXISTS_DO_NOT_REFREEZE'
    for name in ['full_model_load_and_dispatch','real_loss_gradients','RANS_identity','GPU_mapping']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    assert sha(PILOT)==PILOT_SHA and sha(BASELINE)==BASELINE_SHA
    plan=plan_rows();assert len(plan)==12000
    for i in range(3000):assert sorted(r['domain'] for r in plan[4*i:4*i+4])==['ulong','ulong','uvg','uvg']
    pool=load(ROOT/'vendor/sealed/manifests/B_expanded_ulong.json')
    dev=load(ROOT/'vendor/sealed/manifests/development.json');test=load(ROOT/'vendor/sealed/manifests/new_holdout_test.json')
    assert len(pool)==8470 and len(dev)==len(test)==32
    Q={r['group_id'] for r in load(ROOT/'vendor/sealed/graph/quarantine.json')['groups']};assert len(Q)==236
    B={r['source_group'] for r in pool};D={r['source_group'] for r in dev};T={r['source_group'] for r in test}
    assert not (B&D or B&T or D&T or (B|D|T)&Q)
    actual=[r for r in load(ROOT/'vendor/sealed/manifests/evaluation_rows.json') if r['split']=='development']
    assert {r['source_group'] for r in actual}==D
    dump(ROOT/'audits/data_plan.json',dict(status='PASS',source_cursor_start=1500,rank_samples_at_cap=len(plan),updates_at_cap=3000,
        B_ULong_pool=8470,quarantined_source_groups=236,train_dev_test_source_groups_disjoint=True,plan_index_sha256=sha(ROOT/'plans/index.json'),
        all_row_hashes_inherited=True,new_window_or_crop_selection=False,RGB_crop_hashes_verified_at_reader=True))
    inherited=load(ROOT/'vendor/previous_source_manifest.json')
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','logs','audits'])}
    immutable={str(p.relative_to(ROOT)):sha(p) for base in ['vendor/sealed','manifests'] for p in (ROOT/base).rglob('*.json')}
    for n in ['requested_protocol.json','bitstream_manifest.json','audits/sha_correction_20261106/correction.json']:immutable[n]=sha(ROOT/n)
    manifest=dict(status='FROZEN',revision='generator_convlora_v1_corrected_SHA',time=time.time(),code=code,
        protocol_sha256=sha(ROOT/'protocol.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),
        inherited_external_sources=inherited['inherited_external_sources'],immutable_manifests=immutable,
        final_videos_sha256=sha(ROOT/'manifests/final_videos.json'),baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,
        engineering_audits={n:sha(ROOT/f'audits/{n}.json') for n in ['full_model_load_and_dispatch','real_loss_gradients','RANS_identity','GPU_mapping','data_plan']})
    dump(ROOT/'source_manifest.json',manifest);verify_sources()
    dump(ROOT/'final_integrity.json',dict(status='PENDING',experiment_complete=False,committed_optimizer_updates=0,
        blocker=None,full_engineering_checks='PASS',formal_training='NOT_STARTED',final_test_completed_points=0,final_test_expected_points=135))
    dump(ROOT/'audits/sources_frozen.json',dict(status='PASS',protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),code_files=len(code),time=time.time()))
    status('engineering_complete_ready_for_DDP','READY',committed_optimizer_updates=0,blocker=None)
    print('SOURCE_PROTOCOL_DATA_STREAM_IDENTITIES_FROZEN',sha(ROOT/'source_manifest.json'))

if __name__=='__main__':main()
