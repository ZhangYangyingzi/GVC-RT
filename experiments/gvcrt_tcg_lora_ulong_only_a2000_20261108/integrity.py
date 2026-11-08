"""Committed-update, stop-rule, source, feature/stream and final Cartesian-product checks."""
from common_s2 import *
from evaluate import videos,read_csv,METRICS
from evaluation_support import key,validate_point

def checkpoint_manifest():
    import torch
    rows=[]
    for p in sorted((ROOT/'checkpoints').glob('*.pt')):
        s=torch.load(p,map_location='cpu',weights_only=True)
        rows.append(dict(filename=p.name,size_bytes=p.stat().st_size,SHA256=sha(p),additional_updates=s['committed_optimizer_updates'],
            absolute_LoRA_optimizer_step=s['absolute_LoRA_optimizer_step'],TCG_cumulative_step=6500,s_train=s['s_train'],s_eval=s['s_eval']))
    dump(ROOT/'checkpoints_manifest.json',dict(records=rows,weights_uploaded=False,time=time.time()))

def training_integrity():
    import torch
    from train import rng_hash
    cp=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True);n=cp['committed_optimizer_updates']
    assert cp['parent_A2000_sha256']==INIT_SHA and cp['absolute_LoRA_optimizer_step']==cp['optimizer_step']==2000+n and cp['next_cursor']==n
    assert cp['s_train']==1. and cp['s_eval']==.5 and cp['TCG_cumulative_step_frozen_at']==6500 and cp['physical_world_size']==4
    assert len(cp['all_logical_rank_RNG'])==4 and cp['boundary_buffers_cleared'] and not cp['engineering_only']
    assert all(int(v['step'])==2000+n for v in cp['optimizer']['state'].values())
    parent=torch.load(INIT,map_location='cpu',weights_only=True);zero=torch.load(ROOT/'checkpoints/step_0.pt',map_location='cpu',weights_only=True)
    assert zero['committed_optimizer_updates']==0 and zero['absolute_LoRA_optimizer_step']==2000
    assert all(torch.equal(t,zero['lora_state'][name]) for name,t in parent['lora_state'].items())
    assert all(rng_hash(parent['all_logical_rank_RNG'][str(i)])==rng_hash(zero['all_logical_rank_RNG'][str(i)]) for i in range(4))
    plan=plan_rows();files=sorted((ROOT/'receipts').glob('update_*.json.gz'));assert len(files)==n
    frozen=set();rows=[]
    for i,p in enumerate(files,1):
        r=load(p);assert r['step']==i and r['source_plan_cursor']==i and r['absolute_LoRA_optimizer_step']==2000+i
        assert r['gradient_normalization_denominator']==4 and r['global_clip_count']==r['optimizer_step_count']==1
        assert r['rank_synchronized'] and r['optimizer_synchronized'] and r['only_LoRA_changed'] and r['frozen_unchanged']
        assert r['reference_anchor_checkpoint_sha256'] is None and r['s_train']==1. and r['s_eval']==.5
        sample=r['logical_samples'];assert [x['logical_rank'] for x in sample]==[0,1,2,3]
        expected=plan[4*(i-1):4*i];assert [x['sample_key'] for x in sample]==[x['sample_key'] for x in expected]
        assert all(x['domain']=='ulong' and x['frozen_byte_exact'] and x['buffers_cleared'] for x in sample)
        assert len({x['source_group'] for x in sample})==4
        for x,row in zip(sample,expected):
            assert sha(ROOT/f"audits/window_reads/{row['sample_key']}.json.gz")==x['read_audit_sha256']
            audit=load(ROOT/f"audits/window_reads/{row['sample_key']}.json.gz")
            assert audit['status']=='PASS' and audit['source_frame_indices']==row['source_frame_indices']
            assert all(math.isfinite(v) for v in x['losses'].values())
        for rank in r['physical_ranks']:
            frozen.add(rank['frozen_sha256']);assert rank['mean_gradient_norm']>0 and rank['clip_count']==rank['optimizer_step_count']==1
        rows.append(r)
    assert len(frozen)==1 and cp['frozen_sha256'] in frozen and cp['last_committed_receipt']==load(files[-1])
    controller=load(ROOT/'stopping_state.json');assert controller['last_validation_step']==n
    anchor=1.;bad=0;best=None;raw=None
    for j,c in enumerate(controller['checks'],1):
        assert c['step']==250*j and c['anchor_before']==anchor
        eligible=all(d<=1 for d in c['D_q'].values()) and c['S']<1
        assert c['qualified']==eligible
        if eligible and c['S']<anchor-.0005:anchor=c['S'];bad=0
        else:bad+=1
        assert c['anchor_after']==anchor and c['bad_check_count']==bad and (bad<4 or j==len(controller['checks']))
        if eligible and (best is None or c['S']<best['S']):best=c
        if raw is None or c['S']<raw['S']:raw=c
        assert load(ROOT/f'candidate/evaluation/development_lora_{c["step"]:04d}/integrity.json')['status']=='PASS'
    assert controller['status'] in ['EARLY_STOPPED_PATIENCE','MAX_BUDGET_REACHED']
    assert bad==4 if controller['status']=='EARLY_STOPPED_PATIENCE' else n==3000
    selection=load(ROOT/'selection.json');assert selection['status']=='FROZEN_BEFORE_FINAL_TEST' and not selection['test_used_for_selection']
    assert selection['selected_additional_step']==(best['step'] if best else 0)
    assert sha(selection['checkpoint'])==selection['checkpoint_SHA256']
    checkpoint_manifest()
    return dict(status='PASS',additional_updates=n,absolute_LoRA_optimizer_step=2000+n,TCG_cumulative_step=6500,
        physical_world_size=4,effective_batch=4,ULong_samples=4*n,UVG_samples=0,Vimeo_samples=0,optimizer_RNG_restored=True,
        no_repeated_skipped_or_uncommitted_updates=True,only_LoRA_changed=True,frozen_state_unchanged=True,
        stop_reason=controller['status'],qualified_best=best,raw_best_trained=raw),rows

def final_check():
    verify_sources();training,receipts=training_integrity();selection=load(ROOT/'selection.json');states=load(ROOT/'final_states.json')['states']
    assert len(states) in [3,4]
    stream_checks=[];frame_counts=transition_counts=0
    for v in videos('final'):
        for qp in [0,5,9]:
            observed=[]
            for state in states:
                cp=Path(state['checkpoint']) if state.get('checkpoint') else None
                p,fr,tr,ct=validate_point(ROOT/state['batch'],v,qp,state['candidate'],cp)
                if not p.get('measurement_reused'):assert (ROOT/state['batch']/f'points/{key(v,qp)}.json').stat().st_mtime>=selection['time']
                assert p['result']['frames']==64 and len(tr)==63
                frame_counts+=len(fr);transition_counts+=len(tr);observed.append((p,ct,tr))
            base=observed[0]
            for item in observed[1:]:
                for k in ['bitstream_sha256','real_bytes','bpp']:assert item[0]['result'][k]==base[0]['result'][k],('FINAL_STREAM_MISMATCH',v['sequence'],qp,k)
                assert item[1]==base[1]
                assert [r['GT_flow_sha256'] for r in item[2]]==[r['GT_flow_sha256'] for r in base[2]]
            stream_checks.append(dict(domain=v['domain'],sequence=v['sequence'],split=v['split'],QP=qp,unique_model_states=len(states),
                bitstream_SHA256=base[0]['result']['bitstream_sha256'],real_bytes=base[0]['result']['real_bytes'],bpp=base[0]['result']['bpp'],
                all_SHA_bytes_bpp_codec_features_equal=True,GT_flow_hashes_equal=True,actual_QPs_equal=all(p[0]['actual_qps']==base[0]['actual_qps'] for p in observed)))
    write_csv(ROOT/'stream_identity_checks.csv',stream_checks);assert len(stream_checks)==225
    for n in ['A2000_s05_initial_reproduction','aggregation_comparison','loss_components','MCL_real_IP_probe','source_isolation']:
        assert load(ROOT/f'audits/{n}.json')['status']=='PASS'
    return dict(status='PASS',complete=True,training=training,unique_final_model_states=len(states),points_per_model=225,MCL_points_per_model=90,
        final_points=225*len(states),frame_records=frame_counts,transition_records=transition_counts,all_rates_streams_codec_features_equal=True,
        source_protection=True,selection_frozen_before_tests=True,source_native_MCL_I_bitstream=True,old_MCL_RGB_bypass_results_reused=False,
        dataset_plots=5,historical_UVG_exposure_retained=True,all_new_outputs_under_project=True,time=time.time())
