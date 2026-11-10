"""Fail closed on missing coverage, parameter, stream, selection or aggregate facts."""
from common_s2 import *
from collections import Counter

def check_point(a):
    r=a['result'];manifest='development' if r['phase']=='development' else 'test';videos=load(ROOT/f'manifests/{manifest}_videos.json');v=next(v for v in videos if (v['dataset'],v['sequence'])==(r['dataset'],r['sequence']))
    frames_path=next(path for path in a['record_SHA256'] if '/frames/' in path);transitions_path=next(path for path in a['record_SHA256'] if '/transitions/' in path)
    frames=list(csv.DictReader((ROOT/frames_path).open()));transitions=list(csv.DictReader((ROOT/transitions_path).open()));assert len(frames)==64 and len(transitions)==63
    def avg(rows,key):
        values=[float(x[key]) for x in rows if x[key]!=''];return sum(values)/len(values) if values else None
    expected={k:avg(frames,k) for k in ['LPIPS_Alex','LPIPS_VGG','DISTS']};expected['PSNR']=-10*math.log10(max(sum(float(x['pixel_SSE']) for x in frames)/(64*v['height']*v['width']*3),1e-15));expected['FloLPIPS']=avg(transitions,'FloLPIPS')
    for k in ['RGB_MCResidual','Sobel_MCResidual']:
        expected[k]=avg(transitions,k);expected[k+'_PtoP']=avg(transitions[1:],k);expected[k+'_ItoP']=avg(transitions[:1],k)
    for k,value in expected.items():assert (r[k] is None and value is None) or (r[k] is not None and value is not None and abs(r[k]-value)<1e-12),('POINT_VS_FRAME_CSV_DISAGREES',r['model'],r['sequence'],r['QP'],k)
    assert sum(int(x['real_bits']) for x in frames)==r['real_bytes']*8 and r['bpp']==r['real_bytes']*8/(64*v['height']*v['width'])
    assert [x['source_RGB_SHA256'] for x in frames]==v['frame_rgb_sha256']
    return True

def main():
    env();import torch
    from results import all_points,tables,FINAL_MODELS
    from selection import points,complete
    from experiment_model import construct,restore
    checks={};sources=load(ROOT/'source_inputs.json')['readonly']
    for p,h in sources.items():assert sha(p)==h,('READONLY_SOURCE_CHANGED',p)
    checks['Teacher_codec_original_and_parent_sources_unchanged']=True
    manifest=load(ROOT/'source_manifest.json')
    assert sha(ROOT/'protocol.json')==manifest['protocol_sha256']
    for p,h in manifest['code'].items():assert sha(ROOT/p)==h,('NUMERICAL_CODE_CHANGED',p)
    checks['frozen_protocol_and_numerical_code']=True
    assert load(ROOT/'initialization_audit.json')['physical_removal_matches_gain0_exact'] and load(ROOT/'initialization_audit.json')['G0_G1_step0_same']
    checks['common_initialization_and_exact_gain0_equality']=True
    sel=load(ROOT/'selection.json');done=load(ROOT/'training_completion.json');assert done['selection_SHA256']==sha(ROOT/'selection.json') and sel['status']=='FROZEN'
    counts=done['updates'];assert len(set(counts.values()))==1 and next(iter(counts.values()))<=300
    plan=plan_rows();comparison=[]
    for g in GROUPS:
        n=counts[g];receipts=[load(ROOT/g/f'training/receipts/{s:04d}.json') for s in range(1,n+1)];assert len(receipts)==n
        for s,r in enumerate(receipts,1):
            assert r['global_optimizer_updates']==s and r['effective_batch']==4 and r['sample_keys']==[x['sample_key'] for x in plan[4*(s-1):4*s]] and r['source_cursor']==4*s
            assert {x['physical_GPU'] for x in r['ranks']}==set([0,1] if g==GROUPS[0] else [2,3]);assert math.isfinite(r['loss']) and r['post_clip_norm']<=1.0001
        last=torch.load(ROOT/g/'training/latest.pt',map_location='cpu',weights_only=True);assert last['committed_updates']==n and len(last['RNG_by_rank'])==2 and all(int(v['step'])==n for v in last['optimizer']['state'].values())
        assert last['common_step0_SHA256']==sha(ROOT/'checkpoints/B300_noLoRA_step0.pt') and not last['LoRA_present']
        for s in range(0,n+1,50):assert (ROOT/g/f'training/checkpoints/step_{s:04d}.pt').exists()
        choice=sel['groups'][g];assert 0<choice['step']<=n and sha(choice['checkpoint'])==choice['checkpoint_SHA256'];comparison.append([r['sample_keys'] for r in receipts])
        complete(points('development',g,0))
        for step in [50,100,200,300]:
            if step<=n:complete(points('development',g,step))
        c=construct(g,False,0);restore(c,torch.load(choice['checkpoint'],map_location='cpu',weights_only=True),False)
        assert not any('lora' in name.lower() or 'router' in name.lower() for name,m in c['pm'].named_modules())
        assert sum(p.numel() for name,p in c['named'])==938884 and not any(p.requires_grad for p in c['pm'].parameters())
        common=torch.load(ROOT/'checkpoints/B300_noLoRA_step0.pt',map_location='cpu',weights_only=True)['recon_state'];recon=c['pm'].recon_generation_net.state_dict()
        for name,t in recon.items():
            if 'recon_generation_net.'+name not in c['trainable_parameter_names']:assert torch.equal(t.cpu(),common[name].to(t.dtype)),('FROZEN_BACKBONE_CHANGED',name)
        assert all(p.dtype==torch.float32 for p in last['trainable_state'].values());del last,c;torch.cuda.empty_cache()
    assert comparison[0]==comparison[1];checks['real_updates_and_matching_plan']=True;checks['LoRA_absent_full_Bridge_TCG_only_backbone_frozen']=True;checks['qualified_or_NOT_QUALIFIED_selection_recorded']=True
    for name in ['teacher_audit.json','gradient_calibration.json','audits/preflight.json',*[f'audits/engineering_{g}.json' for g in GROUPS]]:assert load(ROOT/name)['status']=='PASS'
    checks['training_only_Teacher_gradient_and_reduction_checks']=True
    data=tables();expected=load(ROOT/'protocol.json')['final']['datasets'];actual=Counter((r['dataset'],r['model']) for r in data['final']);assert len(data['final'])==1125
    for ds,n in expected.items():
        for m in FINAL_MODELS:assert actual[(ds,m)]==n*3
    assert len(data['identity'])==225 and all(r['status']=='PASS' for r in data['identity']);checks['complete_1125_coverage_and_equal_RANS_features_sources']=True
    for a in all_points('final')+all_points('development'):
        assert a['status']=='PASS' and a['real_RANS'] and a['actual_I_frame_bits_included'] and a['input_feature_trace_checked'] and not a['reconstruction_uses_GT'] and not a['reconstruction_uses_Teacher']
        for p,h in a['record_SHA256'].items():assert sha(ROOT/p)==h
        check_point(a)
        if a['result']['model']=='Original':assert a['original_baseline_source']['commit']==ORIGINAL_COMMIT and a['original_baseline_source']['no_v6_TCG_LoRA_B300_weights_in_Original_codec']
    checks['independent_Original_and_real_I_codec_trajectory']=True
    checks['all_point_metrics_recomputed_from_64_frame_63_transition_records']=True
    checks['GT_flow_and_validity_identity_equal_between_five_states']=True
    agg=list(csv.DictReader((ROOT/'rd_points_by_dataset.csv').open()));assert len(agg)==5*5*3
    for r in agg:
        z=[p for p in data['final'] if p['dataset']==r['dataset'] and p['model']==r['model'] and p['QP']==int(r['QP'])]
        for k in ['bpp','PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP']:assert abs(float(r[k])-sum(p[k] for p in z)/len(z))<1e-12
    checks['aggregates_recomputed_from_points']=True
    plots=load(ROOT/'plots/manifest.json');assert plots['no_cross_dataset_curves'] and plots['no_BD_rate']
    for p in plots['plots']:assert sha(p['path'])==p['SHA256']
    for ds in expected:
        for k in ['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP']:assert (ROOT/'plots'/ds/(k+'.png')).is_file()
    checks['separate_dataset_and_per_video_RD_plots']=True
    media=load(ROOT/'video_manifest.json');assert media['status']=='PASS' and len(media['native_clips'])==24 and len(media['synchronized_unscaled_ROI'])==4
    for a in media['native_clips']+media['synchronized_unscaled_ROI']:assert sha(a['path'])==a['SHA256']
    checks['lossless_native_and_unscaled_ROI_media']=True
    dump(ROOT/'checkpoint_manifest.json',dict(common_step0=dict(path=str(ROOT/'checkpoints/B300_noLoRA_step0.pt'),SHA256=sha(ROOT/'checkpoints/B300_noLoRA_step0.pt')),selected={g:sel['groups'][g] for g in GROUPS},saved=[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size) for g in GROUPS for p in sorted((ROOT/g/'training/checkpoints').glob('*.pt'))]))
    component_reused=Counter(x for a in all_points('final') for x in a.get('components_reused',[]))
    final=dict(status='PASS',complete=True,checks=checks,updates=counts,stop_reason=done['stop_reason'],coverage=[dict(dataset=ds,model=m,complete=actual[(ds,m)],expected=n*3) for ds,n in expected.items() for m in FINAL_MODELS],new_or_reused=dict(complete_points_reused=0,component_reuse_counts=dict(component_reused),LPIPS_VGG_and_RAFT_MC_all_new=True),selection_SHA256=sha(ROOT/'selection.json'),publication_complete=False,time=time.time())
    dump(ROOT/'final_integrity.json',final)
if __name__=='__main__':main()
