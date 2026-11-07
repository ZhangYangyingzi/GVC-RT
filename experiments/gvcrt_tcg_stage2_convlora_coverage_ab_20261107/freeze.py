"""Final preflight and source/data/protocol freeze before any formal A/B updates."""
from common_s2 import *

def main():
    env();assert all(not (group_dir(g)/'checkpoints/latest.pt').exists() for g in GROUPS),'FORMAL_CHECKPOINT_EXISTS_DO_NOT_REFREEZE'
    refs=[('pilot',PILOT,PILOT_SHA),('B1500_TCG',BASELINE,BASELINE_SHA),('RAFT',RAFT,RAFT_SHA),
        ('I_codec',PROJECT/'checkpoints/GVC-RT_I.pt','c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'),
        ('P_codec',PROJECT/'checkpoints/GVC-RT_P.pt','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e')]
    rows=[dict(component=n,path=str(p),expected_sha256=h,actual_sha256=sha(p),size_bytes=p.stat().st_size) for n,p,h in refs]
    dump(ROOT/'preflight.json',dict(status='PASS' if all(r['expected_sha256']==r['actual_sha256'] for r in rows) else 'FAILED',weight_checks=rows,
        disk=disk(),formal_updates=0,time=time.time()))
    assert all(r['expected_sha256']==r['actual_sha256'] for r in rows),'WEIGHT_SHA_MISMATCH'
    proof_names=['initialization_and_dispatch','real_loss_gradients_A_control','real_loss_gradients_B_expanded','aggregation_preconditions','RANS_coverage_identity','GPU_mapping']
    for name in proof_names:assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    agg=load(ROOT/'audits/aggregation_comparison.json');assert agg['selected_mode'] in ['two_card_parallel','four_card_alternating_segments']
    if agg['selected_mode']=='two_card_parallel':assert agg['status']=='PASS'
    protocol=load(ROOT/'protocol.json');protocol['scheduling']['selected_mode']=agg['selected_mode']
    protocol['reference_A_model_configuration']=protocol.pop('baseline_and_freezing');dump(ROOT/'protocol.json',protocol)
    plan=plan_rows()
    assert len(plan)==12000 and all(sorted(r['domain'] for r in plan[i:i+4])==['ulong','ulong','uvg','uvg'] for i in range(0,len(plan),4))
    pool=load(ROOT/'vendor/sealed/manifests/B_expanded_ulong.json');dev=load(ROOT/'vendor/sealed/manifests/development.json');test=load(ROOT/'vendor/sealed/manifests/new_holdout_test.json')
    assert len(pool)==8470 and len(dev)==len(test)==32
    q={r['group_id'] for r in load(ROOT/'vendor/sealed/graph/quarantine.json')['groups']};assert len(q)==236
    B={r['source_group'] for r in pool};D={r['source_group'] for r in dev};T={r['source_group'] for r in test}
    assert not(B&D or B&T or D&T or(B|D|T)&q)
    assert load(eval_dir('development',group='baseline')/'integrity.json')['status']=='PASS'
    dump(ROOT/'audits/data_plan.json',dict(status='PASS',rows=12000,source_cursor_start=1500,same_order_A_B=True,
        B_ULong_pool=8470,quarantined_groups=236,source_groups_disjoint=True,plan_index_sha256=sha(ROOT/'plans/index.json')))
    old=load(ROOT/'vendor/reference/source_manifest.json')
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','logs','audits','engineering'])}
    immutable={str(p.relative_to(ROOT)):sha(p) for name in ['vendor/sealed','manifests'] for p in (ROOT/name).rglob('*.json')}
    immutable['bitstream_manifest.json']=sha(ROOT/'bitstream_manifest.json')
    proofs=proof_names+['aggregation_comparison','data_plan']
    manifest=dict(status='FROZEN',revision='coverage_AB_v1',code=code,protocol_sha256=sha(ROOT/'protocol.json'),
        plan_index_sha256=sha(ROOT/'plans/index.json'),common_initialization_sha256=sha(ROOT/'common_initialization/untrained_common.pt'),
        inherited_external_sources=old['inherited_external_sources'],immutable_manifests=immutable,
        final_videos_sha256=sha(ROOT/'manifests/final_videos.json'),pilot_sha256=PILOT_SHA,baseline_sha256=BASELINE_SHA,
        engineering_audits={n:sha(ROOT/f'audits/{n}.json') for n in proofs},time=time.time())
    dump(ROOT/'source_manifest.json',manifest);verify_sources()
    dump(ROOT/'final_integrity.json',dict(status='PENDING',experiment_complete=False,A_updates=0,B_updates=0,
        selected_runtime=agg['selected_mode'],formal_training='NOT_STARTED',final_test_points=0))
    status('engineering_complete_ready_for_formal','READY')
    (ROOT/'README.md').write_text('# Generator Conv-LoRA coverage A/B\n\nFresh shared5 branches,A59392/B135168. Frozen B1500 TC6500,old generator/codec/Bridge. No Router.\n\nAgent-run/resume entry:\n\n```bash\nCUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 '+PYTHON+' -B -u '+str(ROOT/'runner.py')+'\n```\n\nActual status:status.json. Do not launch a second runner while runner.lock is held. Final results:REPORT.md,metrics.csv,plots/. Existing test set is reused.\n')
    print('AB_SOURCE_PROTOCOL_DATA_FREEZE_PASS',sha(ROOT/'source_manifest.json'))

if __name__=='__main__':main()
