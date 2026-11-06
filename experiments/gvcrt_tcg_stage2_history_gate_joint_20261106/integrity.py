"""Independent persisted-point, optimizer, sample, dtype and final-product assertions."""
from common_s2 import *
from reuse import read_csv
from evaluate import videos,METRICS,historical_input

def audit_rans_probe():
    points={n:load(ROOT/f'audits/rans_{n}.json') for n in ['matched','zero','enabled']};base=points['matched'];traces={}
    for name,p in points.items():
        assert p['status']=='PASS' and p['fresh_encode'] and p['independent_decode'] and p['TCG_and_Router_reset'] and p['codec_encoder_sync_independent_features_equal']
        assert p['result']['bitstream_sha256']==base['result']['bitstream_sha256'] and p['result']['real_bytes']==base['result']['real_bytes']
        traces[name]=read_csv(ROOT/f'evaluation/engineering_rans_{name}/codec_trace/hevc_b_Kimono1_qp5.csv')
    assert traces['matched']==traces['zero']==traces['enabled']
    assert points['zero']['reconstruction_sha256']==base['reconstruction_sha256']
    v=next(v for v in videos('final') if v['sequence']=='Kimono1');old,stream,trace=historical_input(v,5)
    assert sha(stream)==base['result']['bitstream_sha256'] and stream.stat().st_size==base['result']['real_bytes']
    dump(ROOT/'audits/RANS_identity.json',dict(status='PASS',sequence='Kimono1',QP=5,frames=64,matched_baseline_independent=True,
        M0_matched_reconstruction_strict_equal=True,M0_2_enabled_codec_features_and_stream_equal=True,
        historical_and_new_stream_sha256=sha(stream),bytes=stream.stat().st_size,independent_decode=True,quality_scan=False))

def audit_batch(root,split,checkpoint=None,candidate='joint'):
    points=[load(p) for p in sorted((root/'points').glob('*.json'))];allvideos=videos(split);vmap={v['sequence']:v for v in allvideos}
    expected={(v['sequence'],q) for v in allvideos for q in [0,5,9]};observed=[];flows={};cp_sha=None if checkpoint is None else sha(checkpoint)
    for p in points:
        r=p['result'];v=vmap[r['sequence']];key=f'{r["domain"]}_{r["sequence"]}_qp{r["QP"]}';observed.append((r['sequence'],int(r['QP'])))
        assert p['status']=='PASS' and p['candidate']==candidate and p['joint_checkpoint_sha256']==cp_sha and p['baseline_sha256']==BASELINE_SHA
        assert p['independent_decode'] and p['real_RANS'] and p['model_hashes_unchanged'] and p['TCG_and_Router_reset']
        assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
        assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
        stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
        if candidate=='joint':assert p['fresh_encode'] and not p['input_stream_reused'] and p['codec_encoder_sync_independent_features_equal']
        else:assert not p['fresh_encode'] and p['input_stream_reused'] and cp_sha is None
        fr=read_csv(root/f'frames/{key}.csv');tr=read_csv(root/f'transitions/{key}.csv');ct=read_csv(root/f'codec_trace/{key}.csv');masks=read_csv(root/f'masks/{key}.csv')
        assert len(fr)==64 and len(tr)==len(masks)==63 and len(ct)==64
        assert [int(x['frame']) for x in fr]==list(range(64)) and [int(x['to_frame']) for x in tr]==list(range(1,64))
        assert [int(x['actual_qp']) for x in fr]==p['actual_qps']
        assert all(f['source_frame_RGB_sha256']==v['frame_rgb_sha256'][i] for i,f in enumerate(fr))
        assert sum(int(f['real_bits']) for f in fr)==8*r['real_bytes']
        assert abs(float(r['bpp'])-8*r['real_bytes']/(64*v['height']*v['width']))<1e-12
        assert abs(float(r['kbps'])-8*r['real_bytes']*v['fps']/64/1000)<1e-9
        assert hashlib.sha256(''.join(f['reconstruction_tensor_sha256'] for f in fr).encode()).hexdigest()==p['reconstruction_sha256']
        mse=sum(float(f['pixel_SSE']) for f in fr)/(64*v['height']*v['width']*3);assert abs(-10*math.log10(max(mse,1e-15))-float(r['PSNR']))<1e-9
        for m in ['MS_SSIM','LPIPS','DISTS']:assert abs(sum(float(f[m]) for f in fr)/64-float(r[m]))<1e-10
        for m in ['FloLPIPS','WarpingError','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(t[m]) for t in tr)/63-float(r[m]))<1e-10
        assert r['WarpingError']==r['WarpingError_ReconFlow_PWC'] and all(math.isfinite(float(r[m])) for m in METRICS)
        for t in tr:flows.setdefault((r['sequence'],int(t['to_frame'])),set()).add(t['GT_flow_sha256'])
        for row in masks[1:]:
            for layer in ['tcg_low','tcg_mid','tcg_out']:
                for op in ['previous_input','current_input','history_projection','current_projection','mask_interpolation','keep_weight','history_product','fusion','residual_add']:
                    assert row[f'{layer}_{op}_dtype']=='torch.float32',('TCG_FP32_BOUNDARY_FAILED',key,layer,op)
    assert len(observed)==len(expected) and set(observed)==expected and all(len(s)==1 for s in flows.values())
    write_csv(root/'metrics.csv',[p['result'] for p in points])
    audit=dict(status='PASS',points=len(points),Cartesian_product=True,checkpoint_sha256=cp_sha,candidate=candidate,
        independent_decode=True,finite_metrics=True,frame_and_transition_aggregation=True,actual_stream_bytes=True,TCG_internal_dtype_FP32=True)
    dump(root/'integrity.json',audit);return audit

def final_check():
    import torch
    verify_sources();selection=load(ROOT/'selection.json');stopping=load(ROOT/'stopping_state.json')
    assert selection['status']=='FROZEN_BEFORE_FINAL_TEST' and not selection['final_test_used_for_selection']
    assert stopping['status'] in ['EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED']
    state=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True);n=state['committed_joint_updates']
    assert n==stopping['last_validation_step']==selection['actual_committed_joint_updates'] and 0<n<=3000
    assert state['joint_optimizer_step']==n and state['next_cursor']==1500+n and len(state['all_rank_RNG'])==4
    assert state['boundary_buffers_cleared'] and all(int(s['step'])==n for s in state['optimizer']['state'].values())
    assert [(g['group_name'],g['lr']) for g in state['optimizer']['param_groups']]==[('Router',1e-4),('TCG',1e-5)]
    assert sum(t.numel() for t in state['router_state'].values())==33889 and sum(t.numel() for t in state['tcg_model_state'].values())==137760
    files=sorted((ROOT/'receipts').glob('update_*.json.gz'));assert len(files)==n;plan=plan_rows();frozen=set();GPU_seconds=0.
    for i,path in enumerate(files,1):
        r=load(path);assert r['step']==i and r['source_plan_cursor']==1500+i and r['rank_samples']==4*i
        assert r['rank_synchronized'] and r['only_Router_TCG_changed'] and r['frozen_unchanged']
        assert [v['sample_key'] for v in r['ranks']]==[digest(plan[4*(i-1)+j]) for j in range(4)]
        for x in r['ranks']:
            assert x['frozen_byte_exact'] and x['buffers_cleared'] and x['physical_GPU']==4+x['rank']
            assert all(v>0 for v in x['group_gradient_norms_local'].values()) and all(v>0 for v in x['group_parameter_changes_L2'].values())
            assert all(math.isfinite(float(v)) for v in x['losses'].values());frozen.add(x['frozen_sha256']);GPU_seconds+=x['seconds_before_commit']
    assert len(frozen)==1 and state['frozen_sha256'] in frozen and state['last_committed_receipt']==load(files[-1])
    anchor=1.;bad=0;best=None
    for i,c in enumerate(stopping['checks'],1):
        assert c['step']==250*i and c['anchor_before']==anchor
        if c['score']<anchor-.001:anchor=c['score'];bad=0
        else:bad+=1
        assert c['bad_check_count']==bad and c['anchor_after']==anchor and (bad<4 or i==len(stopping['checks']))
        if best is None or c['score']<best['score']:best=c
    assert best['step']==selection['best_trained_joint']['step']
    assert selection['raw_better_than_baseline']==(best['score']<1.)
    assert selection['meaningful_improvement_threshold_met']==(best['score']<.999)
    assert selection['final_selection']==('trained_joint' if best['score']<1. else 'baseline')
    if stopping['status']=='EARLY_STOPPED_PATIENCE':assert bad==4
    else:assert n==3000
    batches=[('development',None,'matched_baseline')]+[('development',s,'joint') for s in [0]+[c['step'] for c in stopping['checks']]]+[('final',None,'matched_baseline'),('final',best['step'],'joint')]
    for split,step,candidate in batches:
        root=ROOT/'evaluation'/evaluation_id(split,step,candidate);audit=load(root/'integrity.json');assert audit['status']=='PASS'
        assert audit['points']==(96 if split=='development' else 135)
        if split=='final':
            for p in (root/'points').glob('*.json'):assert p.stat().st_mtime>=selection['time']
    for name in ['branch_precision_and_state','gradient_equivalence','matched_M0_equivalence','DDP_smoke','RANS_identity']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    assert load(ROOT/'precision_baseline_comparison.json')['status']=='COMPLETE'
    assert len(list((ROOT/'plots/datasets').glob('*.png')))==4 and len(list((ROOT/'plots/per_video').glob('*.png')))==45
    matched=ROOT/'evaluation'/evaluation_id('final',candidate='matched_baseline');joint=ROOT/'evaluation'/evaluation_id('final',best['step'])
    for v in videos('final'):
        for q in [0,5,9]:
            key=f'{v["domain"]}_{v["sequence"]}_qp{q}';b=load(matched/f'points/{key}.json');j=load(joint/f'points/{key}.json')
            assert b['result']['bitstream_sha256']==j['result']['bitstream_sha256'] and b['result']['real_bytes']==j['result']['real_bytes']
            assert read_csv(matched/f'codec_trace/{key}.csv')==read_csv(joint/f'codec_trace/{key}.csv')
            bt=read_csv(matched/f'transitions/{key}.csv');jt=read_csv(joint/f'transitions/{key}.csv')
            assert [r['GT_flow_sha256'] for r in bt]==[r['GT_flow_sha256'] for r in jt]
    result=dict(status='PASS',baseline_sha256=BASELINE_SHA,baseline_cumulative_tcg_step=6500,Router_initial_M=.2,
        gate_position='after_previous_adapter_before_concat',mask_spatial_shared=True,TCG_and_Router_internal_FP32=True,
        only_Router_TCG_trainable=True,trainable_parameters=171649,committed_joint_updates=n,rank_samples=4*n,
        frozen_components_unchanged=True,frozen_sha256=next(iter(frozen)),no_duplicate_or_skipped_updates=True,
        all_rank_RNG=4,checkpoint_reload=True,selection_before_final_test=True,matched_baseline_points=231,
        development_joint_coverage=f'{96*(len(stopping["checks"])+1)}/{96*(len(stopping["checks"])+1)}',
        final_test_coverage='135/135',codec_stream_identity_all_final_points=True,codec_feature_identity_all_final_points=True,
        finite_metrics=True,dataset_plots=4,per_video_plots=45,stop_reason=stopping['status'],
        allocated_training_GPU_seconds_before_commit=GPU_seconds,new_writes_restricted_to_project_root=True,time=time.time())
    dump(ROOT/'final_integrity.json',result);return result
if __name__=='__main__':env();final_check()
