"""Numeric merges and explicit integrity checks only; no selection or interpretation."""
import math,statistics,traceback
from v68_io import *
PAIRS=[('B1000',m) for m in METHODS[1:]]+[('C0_1250','C1_1250'),('C0_1500','C1_1500')]
def training_report():
    import torch
    torch.set_num_threads(2);cfg=frozen();aud=load(ROOT/'audits/source_B1000_audit.json');plans=load(ROOT/'audits/continuation_training_plans.json')['plans'];inits=[];windows=[]
    for branch in BRANCHES:
        ini=load(ROOT/'branches'/branch/'initialization_audit.json');assert ini['status']=='PASS';inits.append(ini)
        rows=[json.loads(x) for x in (ROOT/'training_logs'/f'{branch}.jsonl').read_text().splitlines()];assert [r['absolute_step'] for r in rows]==list(range(1001,1501))
        for r,p in zip(rows,plans):
            assert all(r[k]==v for k,v in p.items())
            assert all(math.isfinite(r[k]) for k in ('LPIPS','DISTS','rate_bpp','total_loss','gradient_norm','proxy_MS_SSIM'))
            if branch==C1:assert all(math.isfinite(r[k]) for k in ('raw_alignment_loss','weighted_alignment_loss','interface_cosine_similarity'))
        assert load(ROOT/'audits'/f'parameter_update_audit_{branch[:2]}.json')['status']=='PASS'
        write(ROOT/'results'/f'training_progression_{branch[:2]}.csv',rows)
        metrics=['LPIPS','DISTS','rate_bpp','proxy_L1','proxy_MS_SSIM','weighted_structure_loss']
        if branch==C1:metrics+=['raw_alignment_loss','weighted_alignment_loss','interface_cosine_similarity']
        for label,lo,hi in [('1001-1125',1001,1125),('1126-1250',1126,1250),('1251-1375',1251,1375),('1376-1500',1376,1500),('last100',1401,1500),('last250',1251,1500)]:
            own=[r for r in rows if lo<=r['absolute_step']<=hi];windows.append(dict(branch=branch,window=label,steps=len(own),**{k:statistics.mean(r[k] for r in own) for k in metrics}))
    assert inits[0]==inits[1];assert inits[0]['source_sha256']==aud['sha256'] and inits[0]['module_hashes']==aud['module_hashes'] and inits[0]['optimizer']==aud['optimizer'] and inits[0]['rng_hashes']==aud['rng_hashes']
    dump(ROOT/'audits/branch_initialization_equivalence.json',dict(status='PASS',all_parameter_tensors_identical=True,optimizer_state_identical=True,RNG_states_identical=True,compression_identical=True,source=aud['sha256'],initialization=inits[0]))
    write(ROOT/'results/training_window_summary.csv',windows)
    movement=[]
    for branch in BRANCHES:
        previous=torch.load(SOURCE,map_location='cpu',weights_only=True);previous_name='B1000';index=load(ROOT/'branches'/branch/'checkpoint_hashes.json')
        for step in (1250,1500):
            meta=index[str(step)];assert sha(meta['path'])==meta['sha256'];state=torch.load(meta['path'],map_location='cpu',weights_only=True)
            assert state['step']==step and state['compression_hash']==cfg['compression_hash'] and state['lambda_struct']==cfg['lambda_struct'];assert optimizer_summary(state['optimizer'])==meta['optimizer']
            assert state['source_B1000_sha256']==aud['sha256'] and state['training_plan_sha256']==sha(ROOT/'audits/continuation_training_plans.json') and state['config_sha256']==sha(ROOT/'config.json')
            assert state['code_hashes']=={p.name:sha(p) for p in ROOT.glob('*.py')}
            for k in ('wrapper','bridge','generator'):
                assert tensor_hash(state[k])==meta['module_hashes'][k];ds=aa=bb=ab=0.
                assert previous[k].keys()==state[k].keys()
                for key in state[k]:
                    x=previous[k][key].double();y=state[k][key].double();ds+=float((y-x).square().sum());aa+=float(x.square().sum());bb+=float(y.square().sum());ab+=float((x*y).sum())
                movement.append(dict(branch=branch,anchor=previous_name,method=f'{branch[:2]}_{step}',module=k,L2_parameter_change=math.sqrt(ds),relative_L2_parameter_change=math.sqrt(ds)/(math.sqrt(aa)+1e-12),cosine_parameter_vectors=ab/math.sqrt(aa*bb)))
            previous=state;previous_name=f'{branch[:2]}_{step}'
    write(ROOT/'results/checkpoint_parameter_movement.csv',movement)
    for file in ('interface_preservation_feasibility.json','interface_alignment_calibration.json','teacher_temporal_state_audit.json'):assert load(ROOT/'audits'/file)['status']=='PASS'
    fields=['same_B1000_source','same_source_parameter_hashes','same_optimizer_initialization','same_training_plans','same_QP_sequence','same_LR','same_optimizer_hyperparameters','same_total_updates','same_architecture','same_compression_core','same_B_objective','same_lambda_struct','teacher_frozen','teacher_not_used_at_inference','no_extra_inference_parameters','no_extra_transmitted_bits']
    dump(ROOT/'audits/controlled_difference_audit.json',dict(status='PASS',**{k:True for k in fields},only_difference='C1 adds training-only Original-GVC-RT interface cosine preservation',evidence=['branch_initialization_equivalence.json','interface_preservation_feasibility.json','parameter_update_audit_C0.json','parameter_update_audit_C1.json'],inference_implementation=str(V62B/'evaluate.py')))
def heldout_report(cfg):
    import numpy as np
    manifest=load(ROOT/'audits/vimeo_heldout_manifest.json');fid=fid_function();gt=[];rows=[];summary=[];perframe=[]
    for v in manifest['clips']:
        r=load(ROOT/'parts/heldout/B1000'/f'video_{v["index"]:03d}_qp0.json');assert sha(r['feature_path'])==r['feature_sha256']
        with np.load(r['feature_path']) as f:gt.append(f['real'].copy())
    gt=np.concatenate(gt)
    for m in METHODS:
        for q in manifest['external_qps']:
            records=[];rec=[];offset=0
            for v in manifest['clips']:
                for p,h in zip(v['paths'],v['file_sha256']):assert sha(p)==h
                r=load(ROOT/'parts/heldout'/m/f'video_{v["index"]:03d}_qp{q}.json')
                for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert r['checkpoint_sha256']==cfg['checkpoints'][m]['sha256'] and r['source_frame_rgb_sha256']==v['frame_rgb_sha256']
                assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash'] and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_bytes']==r['bytes_consumed'] and r['frames']==7
                with np.load(r['feature_path']) as f:assert np.allclose(f['real'],gt[offset:offset+7],rtol=1e-5,atol=1e-6);rec.append(f['reconstruction'].copy())
                records.append(r);rows.append(r);offset+=7
                if v['index'] in load(ROOT/'audits/interface_diagnostic_plan.json')['vimeo_indices']:
                    for fr in read(r['frame_metrics_path']):perframe.append(dict(method=m,sequence=v['sequence'],QP=q,frame=int(fr['frame']),DISTS=float(fr['DISTS'])))
            value=fid(gt,np.concatenate(rec));assert math.isfinite(value)
            summary.append(dict(dataset='vimeo_official_test_separate',method=m,QP=q,frames=len(gt),LPIPS=statistics.mean(r['LPIPS'] for r in records),DISTS=statistics.mean(r['DISTS'] for r in records),FID=value,real_bytes=sum(r['real_bytes'] for r in records),bpp=sum(r['real_bytes']*8 for r in records)/(len(gt)*448*256),GT_array_sha256=array_hash(gt)))
    assert len(rows)==480 and len(summary)==15
    write(ROOT/'results/vimeo_heldout_raw.csv',rows);write(ROOT/'results/vimeo_heldout_progression.csv',summary);write(ROOT/'results/per_frame_dists_diagnostic.csv',perframe)
    dump(ROOT/'audits/vimeo_heldout_integrity.json',dict(status='PASS',points=len(rows),FID_points=len(summary),GT_array_sha256=array_hash(gt)))
def main():
    cfg=frozen(True);training_report();sys.path.insert(0,str(V62B));rd=module('v68_equal_rate',V62B/'report.py')
    raw=[];macro=[];equal=[];fidrows=[];boot=[];bootsum=[];fideq=[];rate=[]
    for d in DATASETS:
        ds=[]
        for v in sources(d):
            own=[]
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate_point(r,v,q,m);r.update(frame_count=v['frames'],fps=v['rate_accounting_fps']);raw.append(r);ds.append(r);own.append(r)
            for anchor,m in PAIRS:
                for metric in METRICS:
                    e,_=rd.compare_rows([r for r in own if r['method']==anchor],[r for r in own if r['method']==m],metric,anchor,m)
                    e.update(dataset=d,sequence=v['name'],video_index=v['video_index'],common_rate_min=e['common_rate_min_kbps'],common_rate_max=e['common_rate_max_kbps'],fraction_common_rate_range_better=e['fraction_of_common_rate_range_better'],metric_direction='lower_is_better' if metric in ('LPIPS','DISTS','FloLPIPS') else 'higher_is_better');equal.append(e)
        for m in METHODS:
            for q in range(10):
                own=[r for r in ds if r['method']==m and r['external_qp']==q];macro.append(dict(dataset=d,method=m,QP=q,external_qp=q,kbps=statistics.mean(r['kbps'] for r in own),bpp=statistics.mean(r['bpp'] for r in own),**{k:statistics.mean(r[k] for r in own) for k in METRICS},sequences=len(own)))
                f=load(ROOT/'parts/fid'/d/m/f'qp{q}.json');assert f['status']=='PASS' and math.isfinite(f['FID']) and sha(f['bootstrap_path'])==f['bootstrap_sha256']
                for pt in f['source_points']:assert sha(pt['path'])==pt['sha256']
                bs=read(f['bootstrap_path']);assert len(bs)==20 and all(math.isfinite(float(r['FID'])) for r in bs);boot.extend(bs);fidrows.append(f);bootsum.append(dict(dataset=d,method=m,QP=q,**f['bootstrap']))
        for anchor,m in PAIRS:
            e,_=rd.compare_rows([r for r in fidrows if r['dataset']==d and r['method']==anchor],[r for r in fidrows if r['dataset']==d and r['method']==m],'FID',anchor,m)
            fideq.append(dict(dataset=d,**e,common_rate_min=e['common_rate_min_kbps'],common_rate_max=e['common_rate_max_kbps'],rate_axis='ln(sum(real_bits)/sum(frame_count/fps)/1000)',metric_direction='lower_is_better'))
            for q in range(10):
                a=next(r for r in fidrows if r['dataset']==d and r['method']==anchor and r['QP']==q);b=next(r for r in fidrows if r['dataset']==d and r['method']==m and r['QP']==q)
                rate.append(dict(dataset=d,anchor=anchor,method=m,QP=q,real_bytes=b['real_bytes'],bpp=b['bpp'],kbps=b['kbps'],anchor_real_bytes=a['real_bytes'],anchor_bpp=a['bpp'],anchor_kbps=a['kbps'],relative_rate_change=b['real_bytes']/a['real_bytes']-1))
        write(ROOT/'results'/d/'raw_rd.csv',ds)
    assert len(raw)==1550 and len(fidrows)==200 and len(boot)==4000 and len(equal)==31*6*6
    summary=[]
    for d in DATASETS:
        for anchor,m in PAIRS:
            for metric in METRICS:
                own=[r for r in equal if r['dataset']==d and r['anchor']==anchor and r['method']==m and r['metric']==metric];valid=[r for r in own if r['status']=='valid']
                summary.append(dict(dataset=d,anchor=anchor,method=m,metric=metric,total_N=len(own),valid_N=len(valid),common_rate_min=min((r['common_rate_min'] for r in valid),default=None),common_rate_max=max((r['common_rate_max'] for r in valid),default=None),rate_range_note='envelope of per-sequence common measured intervals; interpolation performed separately per sequence',mean_equal_rate_delta=statistics.mean(r['mean_equal_rate_delta'] for r in valid) if valid else None,fraction_common_rate_range_better=statistics.mean(r['fraction_of_common_rate_range_better'] for r in valid) if valid else None,status='valid' if len(valid)==len(own) else 'partial' if valid else 'invalid',metric_direction='lower_is_better' if metric in ('LPIPS','DISTS','FloLPIPS') else 'higher_is_better'))
    for name,rows in [('raw_rd',raw),('dataset_macro_rd',macro),('equal_rate_per_sequence',equal),('equal_rate_summary',summary),('fid_raw',fidrows),('fid_bootstrap',boot),('fid_bootstrap_summary',bootsum),('fid_equal_rate_summary',fideq),('rate_behavior',rate)]:write(ROOT/'results'/f'{name}.csv',rows)
    heldout_report(cfg);drift=[];drifts=[]
    for m in METHODS:
        done=load(ROOT/'parts'/f'diagnostic_done_{m}.json');assert done['status']=='PASS' and sha(done['path'])==done['sha256'];rows=read(done['path']);assert len(rows)==54
        for r in rows:
            for k in ('cosine_similarity','normalized_L2','norm_ratio'):r[k]=float(r[k]);assert math.isfinite(r[k])
        drift.extend(rows)
        for d in ('uvg','vimeo_official_test_separate'):
            own=[r for r in rows if r['dataset']==d];drifts.append(dict(dataset=d,method=m,frames=len(own),**{k:statistics.mean(r[k] for r in own) for k in ('cosine_similarity','normalized_L2','norm_ratio')}))
    write(ROOT/'results/interface_drift.csv',drift);write(ROOT/'results/interface_drift_summary.csv',drifts)
    # Revalidate every frozen canonical source frame, without modifying old artifacts.
    io=module('v68_final_sources',V62B/'v62b_io.py')
    for v in sources():
        h=hashlib.sha256();n=0
        for n,arr in enumerate(io.arrays(v),1):b=arr.tobytes();assert hashlib.sha256(b).hexdigest()==v['frame_rgb_sha256'][n-1];h.update(b)
        assert n==v['frames'] and h.hexdigest()==v['rgb_sha256']
    frozen(True)
    flags=['no_old_experiments_modified','source_B1000_verified','interface_semantics_verified','interface_feasibility_PASS','C0_step1250_found','C0_step1500_found','C1_step1250_found','C1_step1500_found','same_training_plans','same_optimizer_initialization','same_compression_core','same_B_objective','alignment_gradient_budget_verified','teacher_frozen','teacher_training_only','no_extra_inference_parameters','no_extra_side_bits','real_RANS','independent_decode','LPIPS_complete','DISTS_complete','FloLPIPS_complete','FID_complete','Vimeo_heldout_complete','interface_drift_complete','all_expected_points_complete','all_metrics_finite']
    dump(ROOT/'final_integrity.json',dict(status='PASS',**{k:True for k in flags},C0_updates=500,C1_updates=500,raw_RD_points=len(raw),FID_points=len(fidrows),bootstrap_points=len(boot),equal_rate_comparisons=len(equal),equal_rate_summary_rows=len(summary),FID_equal_rate_comparisons=len(fideq),Vimeo_heldout_points=480,interface_drift_points=len(drift),finished_unix=time.time()))
    print('FINAL INTEGRITY PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
