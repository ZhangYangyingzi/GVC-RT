"""Persisted-frame, codec-stream, frozen-state, cursor and final-product checks."""
from common_s2 import *
from evaluate import videos,METRICS,historical_input,read_csv

def audit_batch(root,split,checkpoint=None,candidate='lora'):
    points=[load(p) for p in sorted((root/'points').glob('*.json'))];allvideos=videos(split);vmap={v['sequence']:v for v in allvideos}
    expected={(v['sequence'],q) for v in allvideos for q in [0,5,9]};observed=[];flows={};cp_sha=None if checkpoint is None else sha(checkpoint)
    for p in points:
        r=p['result'];v=vmap[r['sequence']];key=f'{r["domain"]}_{r["sequence"]}_qp{r["QP"]}';observed.append((r['sequence'],int(r['QP'])))
        assert p['status']=='PASS' and p['candidate']==candidate and p['lora_checkpoint_sha256']==cp_sha and p['baseline_sha256']==BASELINE_SHA
        assert p['independent_decode'] and p['real_RANS'] and p['model_hashes_unchanged'] and p['TCG_reset']
        assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
        assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
        stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
        assert p['pilot_sha256']==PILOT_SHA and not p['fresh_encode'] and p['input_stream_reused'] and p['historical_codec_feature_trace_checked']
        assert p['bitstream_matches_canonical'] and p['bpp_matches_canonical']
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
        if candidate=='lora':
            calls=[k for k in masks[-1] if k.endswith('_forward_calls')]
            assert len(calls)==5 and all(int(masks[-1][k])>=63 for k in calls),'LORA_BRANCH_BYPASSED'
            assert all(v=='torch.float32' for k,v in masks[-1].items() if k.endswith('_parameter_dtype'))
        for row in masks[1:]:
            for layer in ['tcg_low','tcg_mid','tcg_out']:
                for op in ['previous_input','current_input','history_projection','current_projection','mask_interpolation','keep_weight','history_product','fusion','residual_add']:
                    assert row[f'{layer}_{op}_dtype']=='torch.float32',('TCG_FP32_BOUNDARY_FAILED',key,layer,op)
    assert len(observed)==len(expected) and set(observed)==expected and all(len(s)==1 for s in flows.values())
    write_csv(root/'metrics.csv',[p['result'] for p in points])
    audit=dict(status='PASS',points=len(points),Cartesian_product=True,checkpoint_sha256=cp_sha,candidate=candidate,
        independent_decode=True,finite_metrics=True,frame_and_transition_aggregation=True,actual_stream_bytes=True,TCG_internal_dtype_FP32=True)
    dump(root/'integrity.json',audit);return audit

def paired_streams(split,step):
    baseline=ROOT/'evaluation'/evaluation_id(split,candidate='baseline')
    candidate=ROOT/'evaluation'/evaluation_id(split,step)
    rows=[]
    for video in videos(split):
        for qp in [0,5,9]:
            key=f'{video["domain"]}_{video["sequence"]}_qp{qp}'
            bp=load(baseline/f'points/{key}.json');cp=load(candidate/f'points/{key}.json')
            b,c=bp['result'],cp['result']
            assert b['real_bytes']==c['real_bytes'] and b['bpp']==c['bpp'] and b['bitstream_sha256']==c['bitstream_sha256'],'PAIRED_STREAM_MISMATCH'
            assert read_csv(baseline/f'codec_trace/{key}.csv')==read_csv(candidate/f'codec_trace/{key}.csv'),'PAIRED_CODEC_FEATURE_MISMATCH'
            bt=read_csv(baseline/f'transitions/{key}.csv');ct=read_csv(candidate/f'transitions/{key}.csv')
            assert [t['GT_flow_sha256'] for t in bt]==[t['GT_flow_sha256'] for t in ct]
            if step==0:
                assert bp['reconstruction_sha256']==cp['reconstruction_sha256'],'ZERO_LORA_FULL_WINDOW_OUTPUT_MISMATCH'
                assert all(b[m]==c[m] for m in METRICS),'ZERO_LORA_METRIC_MISMATCH'
            rows.append(dict(split=split,sequence=video['sequence'],QP=qp,LoRA_step=step,baseline_bytes=b['real_bytes'],candidate_bytes=c['real_bytes'],
                baseline_bpp=b['bpp'],candidate_bpp=c['bpp'],baseline_stream_sha256=b['bitstream_sha256'],candidate_stream_sha256=c['bitstream_sha256'],
                byte_equal=True,bpp_equal=True,bitstream_hash_equal=True,codec_feature_equal=True))
    write_csv(candidate/'stream_identity_checks.csv',rows)
    dump(candidate/'stream_identity_integrity.json',dict(status='PASS',split=split,points=len(rows),all_bpp_equal=True,all_stream_hash_equal=True,all_codec_features_equal=True))
    return rows

def final_check(persist=True):
    import torch
    verify_sources();selection=load(ROOT/'selection.json');control=load(ROOT/'stopping_state.json')
    assert selection['status']=='FROZEN_BEFORE_FINAL_TEST' and not selection['final_test_used_for_selection']
    assert control['status'] in ['EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED']
    state=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True);n=state['committed_optimizer_updates']
    assert n==control['last_validation_step']==selection['committed_optimizer_updates'] and 0<n<=3000
    assert state['optimizer_step']==n and state['next_cursor']==1500+n and len(state['all_rank_RNG'])==4
    assert state['baseline_sha256']==BASELINE_SHA and state['pilot_sha256']==PILOT_SHA and state['baseline_cumulative_TCG_step']==6500
    assert all(int(s['step'])==n for s in state['optimizer']['state'].values())
    assert [(g['group_name'],g['lr']) for g in state['optimizer']['param_groups']]==[('LoRA',5e-5)]
    assert set(k for k in state if 'router' in k.lower() or k=='tcg_model_state')==set()
    assert sum(t.numel() for t in state['lora_state'].values())==59392
    assert state['boundary_buffers_cleared']
    files=sorted((ROOT/'receipts').glob('update_*.json.gz'));assert len(files)==n
    plan=plan_rows();frozen=set();GPU_seconds=0.
    for i,path in enumerate(files,1):
        r=load(path);assert r['step']==i and r['source_plan_cursor']==1500+i and r['rank_samples']==4*i
        assert r['rank_synchronized'] and r['only_LoRA_changed'] and r['frozen_unchanged']
        assert [x['sample_key'] for x in r['ranks']]==[digest(plan[4*(i-1)+j]) for j in range(4)]
        for x in r['ranks']:
            assert x['frozen_byte_exact'] and x['buffers_cleared'] and x['physical_GPU']==4+x['rank']
            assert all(v>0 for v in x['group_gradient_norms_local'].values()) and all(v>0 for v in x['group_parameter_changes_L2'].values())
            assert all(math.isfinite(float(v)) for v in x['losses'].values())
            frozen.add(x['frozen_sha256']);GPU_seconds+=x['seconds_before_commit']
    assert len(frozen)==1 and state['frozen_sha256'] in frozen and state['last_committed_receipt']==load(files[-1])
    anchor=1.;bad=0;best=None
    for i,c in enumerate(control['checks'],1):
        assert c['step']==250*i and c['anchor_before']==anchor
        if c['score']<anchor-.001:anchor=c['score'];bad=0
        else:bad+=1
        assert c['bad_check_count']==bad and c['anchor_after']==anchor and (bad<4 or i==len(control['checks']))
        if best is None or c['score']<best['score']:best=c
    assert best['step']==selection['best_trained']['step'] and best['score']==selection['best_trained']['score']
    assert selection['final_selection']==('trained_LoRA' if best['score']<1. else 'baseline')
    if control['status']=='EARLY_STOPPED_PATIENCE':assert bad==4
    else:assert n==3000
    complete=[]
    batches=[('development',None,'baseline')]+[('development',step,'lora') for step in [0]+[c['step'] for c in control['checks']]]
    batches += [('final',None,'baseline'),('final',best['step'],'lora')]
    for split,step,candidate in batches:
        root=ROOT/'evaluation'/evaluation_id(split,step,candidate);a=load(root/'integrity.json')
        assert a['status']=='PASS' and a['points']==(96 if split=='development' else 135)
        if split=='final':assert all(p.stat().st_mtime>=selection['time'] for p in (root/'points').glob('*.json'))
        if candidate=='lora':complete+=paired_streams(split,step)
    for name in ['full_model_load_and_dispatch','real_loss_gradients','RANS_identity','DDP_smoke']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    assert len(list((ROOT/'plots/datasets').glob('*.png')))==4
    assert len(list((ROOT/'plots/per_video').glob('*.png')))==45
    result=dict(status='PASS',experiment_complete=True,committed_optimizer_updates=n,rank_samples=4*n,LoRA_parameters=59392,rank=4,
        baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,TCG_cumulative_step_frozen_at=6500,
        Router_instantiated=False,only_LoRA_changed=True,TCG_and_all_original_components_frozen=True,
        frozen_state_sha256=next(iter(frozen)),all_rank_RNG=4,checkpoint_reload=True,no_duplicate_skipped_updates=True,
        actual_LoRA_dispatch_verified=True,all_pairs_bpp_and_stream_hash_equal=True,all_pairs_codec_features_equal=True,
        final_test_completed_points=135,final_baseline_points=135,full_development_points=96*(2+len(control['checks'])),
        finite_metrics=True,selection_before_final_test=True,stop_reason=control['status'],
        shared_environment_training_GPU_seconds_before_commit=GPU_seconds,dataset_plots=4,per_video_plots=45,time=time.time())
    if persist:dump(ROOT/'final_integrity.json',result)
    return result
