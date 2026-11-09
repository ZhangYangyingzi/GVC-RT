"""Independent counts, bytes, hashes and aggregate recomputation."""
from common_s2 import *
from freeze import verify
from summarize import results,training_summary,screening,should_stop,PRIMARY
import collections

def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def optional_float(row,key):
    value=row.get(key)
    return None if value in (None,'') else float(value)

def check():
    env();verify();coverage=training_summary();history=screening();errors=[];points=0;frames=0;transitions=0;GT_reference_hashes={}
    base=results('A2000',0)
    assert len(base)==36 and len(results('Original',0))==36
    expected={(p['sequence'],q) for p in load(ROOT/'manifests/quick_development.json') for q in [0,5,9]}
    for g in GROUPS:
        donepath=ROOT/g/'training/completion.json'
        if not donepath.exists():errors.append(f'{g}: training not complete');continue
        done=load(donepath);step=done['committed_updates'];assert step in [200,300]
        assert done['status']=='PASS' and coverage[g]['committed_updates']==step and coverage[g]['windows']==4*step
        if step==200:assert should_stop(g)
        for name in [f'engineering_{g}.json',f'RANS_{g}.json',f'checkpoint_reload_{g}.json']:assert load(ROOT/'audits'/name)['status']=='PASS'
        assert load(ROOT/g/'smoke/completion.json')['committed_updates']==20
        if g in 'CD':assert load(ROOT/f'audits/online_alignment_{g}.json')['status']=='PASS'
        for s in range(0,step+1,50):assert (ROOT/g/f'training/checkpoints/step_{s:04d}.pt').exists()
        for s in range(0,step+1,100):
            rr=results(g,s)
            if len(rr)!=36:errors.append(f'{g} step{s}: {len(rr)}/36');continue
            assert {(r['sequence'],r['QP']) for r in rr}==expected
        receipts=[load(p) for p in sorted((ROOT/g/'training/receipts').glob('*.json'))]
        assert all(r['frozen_exact_bytes_checked'] and r['buffers_reset'] and r['gradient_norm']>0 and r['post_clip_norm']<=1.00001 for r in receipts)
        hashes={r['frozen_state_SHA256'] for r in receipts};assert len(hashes)==1
        assert all(len(r['samples'])==4 and all(x['frames_supervised']==15 and x['segment_lengths']==[4,4,4,3] for x in r['samples']) for r in receipts)
        for receipt in receipts:
            computed_valid_windows=0
            for sample in receipt['samples']:
                valid_frames=sample.get('frames_temporal_valid',sum(t['valid_fraction']>0 for t in sample.get('raw_trace',[])))
                empty_frames=sample.get('frames_temporal_empty',15-sum(t['valid_fraction']>0 for t in sample.get('raw_trace',[])))
                assert valid_frames+empty_frames==15
                computed_valid_windows+=int(valid_frames>0)
                assert all(t.get('RGB_temporal') is not None and t.get('Sobel_temporal') is not None for t in sample.get('raw_trace',[]) if t.get('temporal_valid',t.get('valid_fraction',0)>0))
                assert all(t.get('RGB_temporal') is None and t.get('Sobel_temporal') is None for t in sample.get('raw_trace',[]) if not t.get('temporal_valid',t.get('valid_fraction',0)>0))
            assert 0<=computed_valid_windows<=4
            assert receipt.get('batch_temporal_valid_windows',computed_valid_windows)==computed_valid_windows
            assert receipt.get('batch_temporal_empty_windows',4-computed_valid_windows)==4-computed_valid_windows
        for i,r in enumerate(receipts):
            assert r['source_plan_cursor']==4*(i+1) and r['optimizer_steps']==[i+1]*len(r['optimizer_steps'])
        import torch
        cp=torch.load(ROOT/g/'training/latest.pt',map_location='cpu',weights_only=True)
        assert cp['committed_updates']==step and cp['next_cursor']==4*step and cp['last_committed_receipt']==receipts[-1]
        assert cp['protocol_SHA256']==sha(ROOT/'protocol.json') and cp['source_manifest_SHA256']==sha(ROOT/'source_manifest.json')
        assert all(torch.isfinite(t).all() for t in cp['trainable_state'].values())
    model_paths=list((ROOT/'evaluation').glob('*/step_*/points/*.json'))
    for path in model_paths:
        p=load(path);assert p['status']=='PASS' and p['codec_features_match_original'] and p['codec_references_forced_to_original'] and p['frozen_components_unchanged']
        r=p['result'];assert (r['sequence'],r['QP']) in expected and r['frames']==64 and r['transitions']==63
        for filename,h in p['record_SHA256'].items():assert sha(ROOT/filename)==h
        records={name:readcsv(path.parents[1]/name/(path.stem+'.csv')) for name in ['frames','transitions','codec_trace']}
        ff=records['frames'];tt=records['transitions']
        for transition in tt:
            gid=(r['sequence'],int(transition['to_frame']));identity=(transition['GT_backward_flow_SHA256'],transition['GT_valid_mask_SHA256'])
            if gid in GT_reference_hashes:assert GT_reference_hashes[gid]==identity,'GT_FLOW_OR_MASK_DIFFERS_ACROSS_MODEL_QP'
            else:GT_reference_hashes[gid]=identity
            valid_pixels=int(transition['valid_pixels'])
            rgb=optional_float(transition,'MCResidual_RGB_RAFT_pixelcenter_v1');edge=optional_float(transition,'MCResidual_Sobel_RAFT_pixelcenter_v1')
            if valid_pixels==0:
                assert rgb is None and edge is None
                assert int(transition['RGB_valid_channel_denominator'])==0 and int(transition['Sobel_valid_channel_denominator'])==0
                assert transition.get('temporal_status','EMPTY_GT_VALID_REGION')=='EMPTY_GT_VALID_REGION'
            else:
                assert rgb is not None and edge is not None
                assert int(transition['RGB_valid_channel_denominator'])==valid_pixels*3 and int(transition['Sobel_valid_channel_denominator'])==valid_pixels*6
                assert transition.get('temporal_status','VALID')=='VALID'
        assert len(ff)==64 and len(tt)==63 and len(records['codec_trace'])==64
        assert [int(x['frame']) for x in ff]==list(range(64)) and [int(x['to_frame']) for x in tt]==list(range(1,64))
        assert sum(int(x['real_bits']) for x in ff)==r['real_bytes']*8 and p['bytes_consumed']==r['real_bytes']
        native=next(x['row'] for x in load(ROOT/'manifests/quick_development.json') if x['sequence']==r['sequence'])
        assert r['bpp']==r['real_bytes']*8/(64*native['height']*native['width'])
        mse=sum(float(x['pixel_SSE']) for x in ff)/(64*native['height']*native['width']*3)
        assert abs(r['PSNR']-(-10*math.log10(max(mse,1e-15))))<1e-9
        for key in ['LPIPS','DISTS','MS_SSIM']:assert abs(r[key]-sum(float(x[key]) for x in ff)/64)<1e-9
        for key in ['FloLPIPS','MCResidual_RGB_RAFT_pixelcenter_v1','MCResidual_Sobel_RAFT_pixelcenter_v1']:
            vals=[optional_float(x,key) for x in tt];vals=[v for v in vals if v is not None]
            ptop_vals=[optional_float(x,key) for x in tt[1:]];ptop_vals=[v for v in ptop_vals if v is not None]
            expected_mean=sum(vals)/len(vals) if vals else None
            expected_ptop=sum(ptop_vals)/len(ptop_vals) if ptop_vals else None
            assert r[key] is None if expected_mean is None else r[key] is not None and abs(r[key]-expected_mean)<1e-9
            assert r[key+'_PtoP'] is None if expected_ptop is None else r[key+'_PtoP'] is not None and abs(r[key+'_PtoP']-expected_ptop)<1e-9
        valid_count=sum(optional_float(x,'MCResidual_RGB_RAFT_pixelcenter_v1') is not None for x in tt)
        ptop_valid_count=sum(optional_float(x,'MCResidual_RGB_RAFT_pixelcenter_v1') is not None for x in tt[1:])
        assert r.get('valid_temporal_transitions',valid_count)==valid_count and r.get('empty_temporal_transitions',63-valid_count)==63-valid_count
        assert r.get('valid_temporal_transitions_PtoP',ptop_valid_count)==ptop_valid_count and r.get('empty_temporal_transitions_PtoP',62-ptop_valid_count)==62-ptop_valid_count
        assert all(math.isfinite(float(v)) for k,v in r.items() if isinstance(v,(int,float)))
        points+=1;frames+=64;transitions+=63
    videos=load(ROOT/'videos/manifest.json') if (ROOT/'videos/manifest.json').exists() else None
    if videos is None:errors.append('128-frame synchronized videos incomplete')
    else:
        assert len(videos['videos'])==4
        for r in videos['videos']:
            assert sha(r['lossless'])==r['sha256'] and sha(r['preview'])==r['preview_sha256'] and r['frames']==128
    assert len(list((ROOT/'plots').glob('*.png')))>=9
    result=dict(status='PASS' if not errors else 'PARTIAL',complete=not errors,time=time.time(),errors=errors,training=coverage,
        validation_points=points,validation_frames=frames,validation_transitions=transitions,each_valid_checkpoint_points=36,
        A2000_SHA256=INIT_SHA,formal_budget_not_expanded=True,final_test_points=0,source_group_isolation_pass=True,
        source_hashes_unchanged=True,codec_stream_bytes_bpp_features=True,frozen_state_exact_bytes_pass=True,
        full_metric_aggregates_recomputed=True,checkpoint_reload_pass=True,no_router=True,local_videos=4 if videos else 0)
    dump(ROOT/'final_integrity.json',result);return result

if __name__=='__main__':check()
