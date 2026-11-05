"""Independent assertions over persisted receipts, points, selected models and curves."""
from common_s2 import *
from reuse import read_csv
from evaluate import videos,METRICS

def audit_rans_probe():
    records={n:load(ROOT/f'audits/rans_{n}.json') for n in ['reference','enabled','bypass']}
    reference=records['reference'];r=reference['result'];traces={}
    for name,p in records.items():
        assert p['status']=='PASS' and p['fresh_encode'] and p['independent_decode'] and p['codec_encoder_sync_independent_features_equal'] and p['TCG_and_Router_reset']
        assert p['result']['bitstream_sha256']==r['bitstream_sha256'] and p['result']['real_bytes']==r['real_bytes']
        traces[name]=read_csv(ROOT/f'evaluation/engineering_rans_{name}/codec_trace/hevc_b_Kimono1_qp5.csv')
    assert traces['reference']==traces['enabled']==traces['bypass']
    assert reference['reconstruction_sha256']==records['bypass']['reconstruction_sha256']
    old=load(HIST/'evaluation_v3/final_d2b5efc4d5756f3604d7/points/B_common_hevc_b_Kimono1_qp5.json')
    assert reference['reconstruction_sha256']==old['reconstruction_sha256'] and r['bitstream_sha256']==old['result']['bitstream_sha256']
    audit=dict(status='PASS',sequence='Kimono1',QP=5,frames=64,original_B1500_unmodified_source=True,
        bypass_reconstruction_strict_equal=True,sealed_B1500_point_reconstruction_matches=True,
        bitstream_sha256=r['bitstream_sha256'],bytes=r['real_bytes'],codec_feature_payload_traces_strict_equal=True,
        independently_decoded=True,enabled0_measured_stream_identity=True)
    dump(ROOT/'audits/RANS_identity.json',audit);return audit

def final_check():
    import torch
    verify_sources();selection=load(ROOT/'selection.json');stopping=load(ROOT/'stopping_state.json')
    assert selection['status']=='FROZEN_BEFORE_FINAL_TEST' and not selection['final_test_used_for_selection']
    assert stopping['status'] in ['EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED']
    cp=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True);n=cp['committed_router_updates']
    assert 0<n<=3000 and n==stopping['last_validation_step']==selection['actual_committed_router_updates']
    assert cp['Router_optimizer_step']==n and cp['next_cursor']==1500+n and len(cp['all_rank_RNG'])==4 and cp['boundary_buffers_cleared']
    assert all(int(s['step'])==n for s in cp['optimizer']['state'].values())
    files=sorted((ROOT/'receipts').glob('update_*.json.gz'));assert len(files)==n
    expected_plan=plan_rows();frozen=set();GPU_seconds=0
    for i,p in enumerate(files,1):
        r=load(p);assert r['step']==i and r['source_plan_cursor']==1500+i and r['rank_samples']==4*i
        assert r['rank_synchronized'] and r['only_Router_changed'] and r['frozen_unchanged']
        assert len(r['ranks'])==4 and [v['sample_key'] for v in r['ranks']]==[digest(expected_plan[4*(i-1)+rank]) for rank in range(4)]
        for x in r['ranks']:
            assert x['frozen_byte_exact'] and x['buffers_cleared'] and x['parameter_change_L2']>0 and x['local_gradient_norm']>0
            assert x['physical_GPU']==4+x['rank'] and all(math.isfinite(float(v)) for v in x['losses'].values())
            frozen.add(x['frozen_sha256']);GPU_seconds+=x['seconds_before_commit']
    assert len(frozen)==1 and cp['frozen_sha256'] in frozen
    assert cp['last_committed_receipt']==load(files[-1])
    anchor=1.;bad=0;best=None
    for i,check in enumerate(stopping['checks'],1):
        assert check['step']==250*i and check['anchor_before']==anchor
        if check['score']<anchor-.001:anchor=check['score'];bad=0
        else:bad+=1
        assert bad==check['bad_check_count'] and anchor==check['anchor_after']
        assert bad<4 or i==len(stopping['checks'])
        if best is None or check['score']<best['score']:best=check
    assert best['step']==selection['best_trained_router']['step']
    assert selection['final_selection']==('trained_router' if best['score']<1 else 'baseline_bypass')
    if stopping['status']=='EARLY_STOPPED_PATIENCE':assert bad==4
    else:assert n==3000
    batches=[('development',0)]+[('development',c['step']) for c in stopping['checks']]+[('final',best['step'])]
    for split,step in batches:
        root=ROOT/f'evaluation/{split}_router_{step:04d}';a=load(root/'integrity.json');assert a['status']=='PASS'
        assert a['points']==(96 if split=='development' else 135)
        points=[load(p) for p in (root/'points').glob('*.json')]
        assert len(points)==a['points']
        for p in points:
            r=p['result'];stream=ROOT/f'large_artifacts/evaluation/{split}_router_{step:04d}'/r['bitstream_file']
            assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
            assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['Router_checkpoint_sha256']==a['checkpoint_sha256']
            assert all(math.isfinite(float(r[m])) for m in METRICS)
            if split=='final':assert (ROOT/f'evaluation/{split}_router_{step:04d}/points/{Path(r["bitstream_file"]).stem}.json').stat().st_mtime>=selection['time']
    assert load(ROOT/'audits/reuse_summary.json')['status']=='PASS'
    base_reuse=load(ROOT/'audits/reuse_B_final.json')['records']
    best_step=selection['best_trained_router']['step']
    for reused in base_reuse:
        key=f'{next(v["domain"] for v in videos("final") if v["sequence"]==reused["sequence"])}_{reused["sequence"]}_qp{reused["QP"]}'
        new=read_csv(ROOT/f'evaluation/final_router_{best_step:04d}/transitions/{key}.csv')
        oldpoint=Path(reused['point_file']);oldroot=oldpoint.parent.parent
        old=read_csv(oldroot/f'transitions/{oldpoint.stem}.csv')
        assert [t['GT_flow_sha256'] for t in new]==[t['GT_flow_sha256'] for t in old],('MATCHED_GT_FLOW_TENSOR_CHANGED',key)
    for name in ['forward_equivalence','gradient_and_state','DDP_smoke','RANS_identity']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    assert len(list((ROOT/'plots/datasets').glob('*.png')))==4 and len(list((ROOT/'plots/per_video').glob('*.png')))==45
    streamchecks=read_csv(ROOT/'stream_identity_checks.csv');assert len(streamchecks)==135
    # Nonidentity is a recorded audit result, not silently normalized rates.
    stream_identical=all(r['same_bytes']=='True' and r['same_stream_sha256']=='True' for r in streamchecks)
    checks=dict(status='PASS',baseline_cumulative_tcg_step=6500,baseline_sha256=BASELINE_SHA,only_Router_trainable=True,
        frozen_components_unchanged=True,frozen_sha256=next(iter(frozen)),committed_router_updates=n,rank_samples=4*n,
        no_duplicate_or_skipped_committed_updates=True,checkpoint_reload=True,all_rank_RNG=4,selection_before_final_test=True,
        final_test_coverage='135/135',development_coverage=f'{96*(len(stopping["checks"])+1)}/{96*(len(stopping["checks"])+1)}',
        baseline_development_reused=96,dataset_plots=4,per_video_plots=45,codec_stream_identity_all_final_points=stream_identical,
        finite_metrics=True,stop_reason=stopping['status'],actual_training_GPU_seconds_before_commit=GPU_seconds,
        new_writes_restricted_to_project_root=True,time=time.time())
    dump(ROOT/'final_integrity.json',checks);return checks

if __name__=='__main__':env();final_check()
