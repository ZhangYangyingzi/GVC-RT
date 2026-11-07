"""Independent persisted metrics, stream, budget and shared-step assertions."""
from common_s2 import *
from evaluate import videos,METRICS,read_csv
def audit_batch(root,split,checkpoint=None,candidate='lora',group='A_control'):
    points=[load(p) for p in sorted((root/'points').glob('*.json'))];allvideos=videos(split);vmap={v['sequence']:v for v in allvideos}
    expected={(v['sequence'],q) for v in allvideos for q in [0,5,9]};observed=[];flows={};cp_sha=None if checkpoint is None else sha(checkpoint)
    for p in points:
        r=p['result'];v=vmap[r['sequence']];key=f'{r["domain"]}_{r["sequence"]}_qp{r["QP"]}';observed.append((r['sequence'],int(r['QP'])))
        assert p['status']=='PASS' and p['candidate']==candidate and p['lora_checkpoint_sha256']==cp_sha and p['baseline_sha256']==BASELINE_SHA and p['group']==group
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
            assert len(calls)==(6 if group=='B_expanded' else 5) and all(int(masks[-1][k])>=63 for k in calls),'LORA_BRANCH_BYPASSED'
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

def paired_streams(split,step,group):
    base=eval_dir(split,group='baseline');candidate=eval_dir(split,step,group);rows=[]
    for v in videos(split):
        for q in [0,5,9]:
            key=f'{v["domain"]}_{v["sequence"]}_qp{q}';b=load(base/f'points/{key}.json');c=load(candidate/f'points/{key}.json')
            br,cr=b['result'],c['result']
            assert br['real_bytes']==cr['real_bytes'] and br['bpp']==cr['bpp'] and br['bitstream_sha256']==cr['bitstream_sha256']
            assert read_csv(base/f'codec_trace/{key}.csv')==read_csv(candidate/f'codec_trace/{key}.csv')
            bt=read_csv(base/f'transitions/{key}.csv');ct=read_csv(candidate/f'transitions/{key}.csv')
            assert [x['GT_flow_sha256'] for x in bt]==[x['GT_flow_sha256'] for x in ct]
            rows.append(dict(group=group,split=split,step=step,sequence=v['sequence'],QP=q,baseline_bpp=br['bpp'],candidate_bpp=cr['bpp'],
                baseline_bytes=br['real_bytes'],candidate_bytes=cr['real_bytes'],stream_sha256=br['bitstream_sha256'],candidate_stream_sha256=cr['bitstream_sha256'],
                bpp_equal=True,bytes_equal=True,stream_hash_equal=True,codec_features_equal=True))
    write_csv(candidate/'stream_identity_checks.csv',rows)
    dump(candidate/'stream_identity_integrity.json',dict(status='PASS',points=len(rows),group=group,all_bpp_bytes_hash_codec_features_equal=True))
    return rows

def verify_training(group):
    import torch
    from train import rng_hash
    home=group_dir(group);s=torch.load(home/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
    control=load(home/'stopping_state.json');selection=load(home/'selection.json');n=s['committed_optimizer_updates']
    assert s['group']==group and s['optimizer_step']==n and s['next_cursor']==1500+n and len(s['all_logical_rank_RNG'])==4
    assert s['TCG_cumulative_step_frozen_at']==6500 and s['baseline_sha256']==BASELINE_SHA and s['pilot_sha256']==PILOT_SHA
    assert not s['engineering_only'] and s['boundary_buffers_cleared']
    assert n==control['last_validation_step']==selection['committed_optimizer_updates'] and 0<n<=3000
    assert [(g['group_name'],g['lr']) for g in s['optimizer']['param_groups']]==[('LoRA',5e-5)]
    assert all(int(v['step'])==n for v in s['optimizer']['state'].values())
    assert sum(t.numel() for t in s['lora_state'].values())==(59392 if group=='A_control' else load(ROOT/'protocol.json')['groups']['B_expanded']['parameters'])
    files=sorted((home/'receipts').glob('update_*.json.gz'));assert len(files)==n;plan=plan_rows();frozen=set();GPU_seconds=0.
    keys={}
    for i,p in enumerate(files,1):
        r=load(p);assert r['step']==i and r['group']==group and r['source_plan_cursor']==1500+i and r['cumulative_samples']==4*i
        assert r['gradient_normalization_denominator']==4 and r['global_clip_count']==r['optimizer_step_count']==1
        assert r['rank_synchronized'] and r['optimizer_synchronized'] and r['only_LoRA_changed'] and r['frozen_unchanged']
        assert r['reference_anchor_checkpoint_sha256'] is None,'FORMAL_UPDATES_MUST_NOT_RESET_TO_ENGINEERING_ANCHORS'
        samples=r['logical_samples'];assert [x['logical_rank'] for x in samples]==list(range(4))
        keys[i]=[x['sample_key'] for x in samples]
        assert keys[i]==[digest(plan[4*(i-1)+j]) for j in range(4)]
        assert sorted(x['domain'] for x in samples)==['ulong','ulong','uvg','uvg']
        for x in samples:
            assert x['frozen_byte_exact'] and x['buffers_cleared'] and all(math.isfinite(float(v)) for v in x['losses'].values())
        for rnk in r['physical_ranks']:
            assert rnk['clip_count']==rnk['optimizer_step_count']==1
            assert all(v>0 for v in rnk['mean_group_gradient_norms'].values()) and all(v>0 for v in rnk['group_parameter_changes_L2'].values())
            frozen.add(rnk['frozen_sha256']);GPU_seconds+=rnk['seconds_before_commit']
    assert len(frozen)==1 and s['frozen_sha256'] in frozen and s['last_committed_receipt']==load(files[-1])
    anchor=1.;bad=0;best=None
    for idx,c in enumerate(control['checks'],1):
        assert c['step']==250*idx and c['anchor_before']==anchor
        if c['score']<anchor-.0005:anchor=c['score'];bad=0
        else:bad+=1
        assert c['bad_check_count']==bad and c['anchor_after']==anchor and (bad<4 or idx==len(control['checks']))
        if best is None or c['score']<best['score']:best=c
        assert load(eval_dir('development',c['step'],group)/'integrity.json')['status']=='PASS'
    assert control['status'] in ['EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED']
    if control['status']=='EARLY_STOPPED_PATIENCE':assert bad==4
    else:assert n==3000
    assert selection['status']=='FROZEN_BEFORE_FINAL_TEST' and not selection['test_used_for_selection']
    assert selection['best_trained']['step']==best['step'] and selection['best_trained']['score']==best['score']
    assert selection['baseline_retained']==(best['score']>=1.)
    assert sha(selection['best_trained']['checkpoint'])==selection['best_trained']['checkpoint_sha256']
    cp0=torch.load(home/'checkpoints/step_0.pt',map_location='cpu',weights_only=True)
    assert cp0['committed_optimizer_updates']==0 and not cp0['optimizer']['state']
    return dict(committed_optimizer_updates=n,best_dev_step=best['step'],dev_score=best['score'],stop_reason=control['status'],
        checkpoint=selection['best_trained']['checkpoint'],checkpoint_sha256=selection['best_trained']['checkpoint_sha256'],
        baseline_retained=selection['baseline_retained'],shared_GPU_seconds_before_commit=GPU_seconds),keys,cp0

def final_check(persist=True):
    import torch
    from train import rng_hash
    verify_sources();summaries={};keys={};init={}
    for g in GROUPS:summaries[g],keys[g],init[g]=verify_training(g)
    assert all(torch.equal(t,init['B_expanded']['lora_state'][n]) for n,t in init['A_control']['lora_state'].items())
    assert all(rng_hash(init['A_control']['all_logical_rank_RNG'][str(i)])==rng_hash(init['B_expanded']['all_logical_rank_RNG'][str(i)]) for i in range(4))
    for i in range(1,min(len(keys['A_control']),len(keys['B_expanded']))+1):assert keys['A_control'][i]==keys['B_expanded'][i]
    components=[load(group_dir(g)/'baseline_provenance.json')['components'] for g in GROUPS];assert components[0]==components[1]
    complete=load(ROOT/'selection_manifest.json');assert complete['status']=='FROZEN_BOTH_BEFORE_FINAL_TEST'
    streamrows=[]
    for split in ['development','final']:
        assert load(eval_dir(split,group='baseline')/'integrity.json')['status']=='PASS'
    for g in GROUPS:
        checks=load(group_dir(g)/'stopping_state.json')['checks']
        for c in checks:streamrows+=paired_streams('development',c['step'],g)
        root=eval_dir('final',summaries[g]['best_dev_step'],g)
        assert load(root/'integrity.json')['points']==135
        assert all(p.stat().st_mtime>=complete['time'] for p in (root/'points').glob('*.json'))
        streamrows+=paired_streams('final',summaries[g]['best_dev_step'],g)
    write_csv(ROOT/'stream_identity_checks.csv',streamrows)
    for n in ['initialization_and_dispatch','real_loss_gradients_A_control','real_loss_gradients_B_expanded','RANS_coverage_identity']:
        assert load(ROOT/f'audits/{n}.json')['status']=='PASS'
    aggregation=load(ROOT/'audits/aggregation_comparison.json');assert aggregation['selected_mode'] in ['two_card_parallel','four_card_alternating_segments']
    if aggregation['selected_mode']=='two_card_parallel':assert aggregation['status']=='PASS'
    assert len(list((ROOT/'plots/datasets').glob('*.png')))==4
    timing=load(ROOT/'decoding_timing.json');assert timing['status']=='PASS' and len(timing['records'])==9
    common=sorted(set(c['step'] for c in load(group_dir('A_control')/'stopping_state.json')['checks'])&set(c['step'] for c in load(group_dir('B_expanded')/'stopping_state.json')['checks']))
    result=dict(status='PASS',experiment_complete=True,groups=summaries,common_validation_steps=common,
        final_points=dict(common_baseline=135,A_control=135,B_expanded=135),all_pairs_bpp_hash_codec_features_equal=True,
        frozen_original_components_unchanged=True,shared_initial_parameters_and_logical_RNG_equal=True,shared_update_sample_keys_equal=True,
        no_repeated_skipped_or_uncommitted_updates=True,TCG_cumulative_step_frozen_at=6500,Router_instantiated=False,
        dataset_plots=4,decoding_timing_records=9,existing_test_set_reused=True,selected_runtime=aggregation['selected_mode'],time=time.time())
    if persist:dump(ROOT/'final_integrity.json',result)
    return result
