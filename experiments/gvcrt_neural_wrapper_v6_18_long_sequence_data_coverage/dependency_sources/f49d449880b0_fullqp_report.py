"""Raw calibration tables only: no selection, ranking, or conclusions."""
import argparse
import collections
import math
import statistics
from fullqp_common import *
from fullqp_evaluate import point_path, validate_point

def training_reports(final=False):
    summaries=[];trajectories=[];gradients=[];plans={};init={};incomplete=[];histograms={}
    for branch in BRANCHES:
        br=ROOT/'branches'/branch;log=br/'training_log.jsonl'
        if not log.exists():incomplete.append(branch);continue
        lines=log.read_text().splitlines();rows=[]
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                assert not final and i==len(lines)-1,'Corrupt completed log'
        assert [r['update'] for r in rows]==list(range(1,len(rows)+1))
        assert len(rows)<=3000
        if len(rows)<3000:incomplete.append(branch)
        plans[branch]=[{k:r[k] for k in ('update','external_qp','video','start','crop_x','crop_y')} for r in rows]
        init[branch]=load(br/'initialization_audit.json');assert init[branch]['fresh_optimizer'] and init[branch]['initial_optimizer_state_entries']==0
        hist=collections.Counter(r['external_qp'] for r in rows);histograms[branch]=dict(hist)
        for r in rows:
            assert all(math.isfinite(r[k]) for k in ('loss','LPIPS','DISTS','D','rate_bpp','beta_q','beta_rate','proxy_L1','lambda_proxy_proxy_L1','rate_to_distortion_ratio','gradient_norm'))
            assert r['all_finite'] and r['external_qp'] in range(10)
            assert math.isclose(r['beta_q'],beta_q(branch,r['external_qp']),rel_tol=1e-12)
            assert math.isclose(r['D'],r['LPIPS']+r['DISTS'],rel_tol=1e-6)
            assert math.isclose(r['loss'],r['D']+r['beta_rate']+r['lambda_proxy_proxy_L1'],rel_tol=1e-6)
            assert r['actual_qps']==load(ROOT/'qp_train_eval_semantics_audit.json')['rows'][r['external_qp']]['evaluation']
        keys=('LPIPS','DISTS','D','rate_bpp','beta_q','beta_rate','proxy_L1','lambda_proxy_proxy_L1','rate_to_distortion_ratio')
        def aggregate(own,q,lo,hi):
            return dict(branch=branch,external_qp=q,update_start=lo,update_end=hi,count=len(own),**{'mean_'+k:statistics.mean(r[k] for r in own) for k in keys},
                ratio_of_means=statistics.mean(r['beta_rate'] for r in own)/statistics.mean(r['D'] for r in own))
        for q in range(10):
            own=[r for r in rows if r['external_qp']==q]
            if own:summaries.append(aggregate(own,q,1,len(rows)))
            for lo,hi in zip(STEPS,STEPS[1:]):
                block=[r for r in own if lo<r['update']<=hi]
                if block:trajectories.append(aggregate(block,q,lo+1,min(hi,len(rows))))
        gp=br/'gradient_balance_diagnostics.csv'
        if gp.exists():gradients.extend(read(gp))
    if summaries:write(ROOT/'loss_component_statistics.csv',summaries)
    if trajectories:write(ROOT/'loss_component_trajectory.csv',trajectories)
    if gradients:write(ROOT/'gradient_balance_diagnostics.csv',gradients)
    if init:
        hashes=[r['model_hashes'] for r in init.values()]
        identical=all(h==hashes[0] for h in hashes);assert identical
        # Validate saved step0 states, not merely claimed audit strings.
        import torch
        actual={}
        for branch in init:
            record=load(ROOT/'branches'/branch/'checkpoint_hashes.json')['0'];cp=Path(record['path'])
            assert sha(cp)==record['sha256'];state=torch.load(cp,map_location='cpu',weights_only=True)
            actual[branch]=model_hashes(state);assert actual[branch]==init[branch]['model_hashes']
            assert len(state['optimizer']['state'])==0
            del state
        dump(ROOT/'initialization_audit.json',dict(status='PASS' if len(actual)==len(BRANCHES) else 'PENDING',
            identical_step0_model_weights=True,step0_model_hashes=actual,fresh_optimizer_all_measured=True,measured_branches=list(actual)))
    compared=min([len(p) for p in plans.values()],default=0)
    values=list(plans.values())
    assert all(p[:compared]==values[0][:compared] for p in values)
    complete=len(plans)==len(BRANCHES) and not incomplete
    dump(ROOT/'paired_sampling_audit.json',dict(status='PASS' if complete else 'PENDING',seed=SEED,
        compared_updates=compared,compared_branches=list(plans),all_compared_plans_equal=True,
        counts={b:len(p) for b,p in plans.items()},plan_sha256={b:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() for b,p in plans.items()},
        uniform_qp_rule='random.Random(seed).randrange(10) after shared sample_clip; identical per-step QP sequence across branches',
        histograms=histograms))
    if final:
        assert complete and compared==3000
        from scipy.stats import chisquare
        uniform=[]
        for b,h in histograms.items():
            counts=[h.get(q,0) for q in range(10)];stat,p=chisquare(counts)
            uniform.append(dict(branch=b,counts=counts,chi_square=float(stat),p_value=float(p),all_qps_observed=all(counts),
                uniform_sampling_implementation=True,distribution_warning=bool(p<.001)))
            assert all(counts)
        dump(ROOT/'training_qp_uniformity_audit.json',dict(status='PASS',rows=uniform,statistical_test='Diagnostic only; no resampling/reselection'))
        assert len(gradients)==len(BRANCHES)*len(STEPS)*2
        for r in gradients:
            assert all(math.isfinite(float(r[k])) for k in ('grad_D_norm','grad_R_norm','grad_beta_R_norm','cos_grad_D_R'))
            assert float(r['grad_D_norm'])>0 and float(r['grad_R_norm'])>0
            assert -1.000001<=float(r['cos_grad_D_R'])<=1.000001
        assert {(r['branch'],int(r['checkpoint_update']),int(r['external_qp'])) for r in gradients}=={(b,s,q) for b in BRANCHES for s in STEPS for q in (0,9)}
    return complete

def collect(branch,step,split):
    videos=load(ROOT/f'validation_{split}_manifest.json')['videos'];rows=[]
    digest='' if branch=='original' else load(ROOT/'branches'/branch/'checkpoint_hashes.json')[str(step)]['sha256']
    for v in videos:
        for q in QPS:
            r=load(point_path(branch,step,split,v['video_index'],q));validate_point(r,v,q,digest);rows.append(r)
    assert len(rows)==192
    return rows

def pool(rows):
    import numpy as np
    fid=module('fullqp_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
    result=[]
    for q in QPS:
        own=[r for r in rows if r['external_qp']==q];refs=[];recons=[]
        for r in own:
            with np.load(r['feature_path']) as f:
                assert f['real'].shape==f['reconstruction'].shape==(32,2048)
                assert np.isfinite(f['real']).all() and np.isfinite(f['reconstruction']).all()
                refs.append(f['real']);recons.append(f['reconstruction'])
        a,b=np.concatenate(refs),np.concatenate(recons);fid_value=float(fid.low_rank_fid(a,b));assert math.isfinite(fid_value)
        result.append(dict(branch=own[0]['branch'],checkpoint_update=own[0]['checkpoint_update'],validation_split=own[0]['validation_split'],
            external_qp=q,QP=q,kbps=statistics.mean(r['kbps'] for r in own),bpp=statistics.mean(r['bpp'] for r in own),
            real_bytes=sum(r['real_bytes'] for r in own),bits=sum(r['bits'] for r in own),
            **{m:statistics.mean(r[m] for r in own) for m in SPATIAL},FloLPIPS=statistics.mean(r['FloLPIPS'] for r in own),FID=fid_value,
            FID_num_samples=len(a),num_sequences=32,num_frames=1024,num_transitions=992))
    return result

def output_dir(branch,step,split):
    return ROOT/'results'/'fullqp'/branch/f'step_{step:04d}'/split

def report_one(branch,step,split):
    out=output_dir(branch,step,split);rows=collect(branch,step,split);summary=pool(rows)
    if branch!='original':
        for r in rows:
            base=load(point_path('original',0,split,r['video_index'],r['external_qp']))
            assert r['source_rgb_sha256']==base['source_rgb_sha256']
    write(out/'per_sequence_all_qp.csv',rows);write(out/'all_qp_summary.csv',summary)
    proxy=[]
    for i in range(32):
        own=[r for r in rows if r['video_index']==i]
        for k in ('proxy_RGB_L1','proxy_RGB_MSE','proxy_edge_difference'):assert max(r[k] for r in own)-min(r[k] for r in own)<1e-12
        proxy.append(own[0])
    proxy_summary=dict(branch=branch,checkpoint_update=step,validation_split=split,num_videos=32,
        **{k:statistics.mean(r[k] for r in proxy) for k in ('proxy_RGB_L1','proxy_RGB_MSE','proxy_edge_difference')})
    dump(out/'proxy_summary.json',proxy_summary)
    if branch!='original':
        anchor_path=output_dir('original',0,split)/'all_qp_summary.csv'
        assert anchor_path.exists();anchor=read(anchor_path)
        sys.path.insert(0,str(V52));from rd_analysis import compare
        eq=[];bd=[]
        for metric in (*SPATIAL,'FloLPIPS','FID'):
            # The shared comparator expects lower-is-better distortion.
            if metric in ('PSNR','SSIM','MS_SSIM'):
                aa=[dict(r,**{metric:-float(r[metric])}) for r in anchor];cc=[dict(r,**{metric:-float(r[metric])}) for r in summary]
            else:aa,cc=anchor,summary
            e,b=compare(aa,cc,metric,'original',branch)
            if metric in ('PSNR','SSIM','MS_SSIM') and e['status']=='valid':e['mean_equal_rate_delta']=-e['mean_equal_rate_delta']
            for z in (e,b):z.update(branch=branch,checkpoint_update=step,validation_split=split,quality_direction='higher' if metric in ('PSNR','SSIM','MS_SSIM') else 'lower')
            eq.append(e);bd.append(b)
        write(out/'equal_rate_summary.csv',eq);write(out/'bd_rate.csv',bd)
        reduction=statistics.mean(1-r['kbps']/float(next(a['kbps'] for a in anchor if int(a['QP'])==r['QP'])) for r in summary)
        trajectory=dict(branch=branch,scale=BRANCHES[branch],fixed_or_schedule='fixed' if BRANCHES[branch] is None else 'schedule',
            checkpoint_update=step,validation_split=split,mean_same_qp_bitrate_reduction=reduction,
            bitrate_reduction_unit='fraction; positive means reduction',proxy_RGB_L1=proxy_summary['proxy_RGB_L1'],proxy_RGB_MSE=proxy_summary['proxy_RGB_MSE'])
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            e=next(r for r in eq if r['metric']==metric);b=next(r for r in bd if r['metric']==metric)
            trajectory['equal_rate_'+metric+'_delta']=e['mean_equal_rate_delta'];trajectory['equal_rate_'+metric+'_status']=e['status']
            trajectory[metric+'_BD_rate']=b['BD_rate_percent'];trajectory[metric+'_BD_rate_status']=b['status'];trajectory[metric+'_BD_rate_reason']=b['reason']
        dump(out/'trajectory.json',trajectory)
    tracked=['all_qp_summary.csv','per_sequence_all_qp.csv','proxy_summary.json']+(['equal_rate_summary.csv','bd_rate.csv','trajectory.json'] if branch!='original' else [])
    dump(out/'report_integrity.json',dict(status='PASS',revision=REVISION,points=192,summary_rows=6,files={p:sha(out/p) for p in tracked}))
    print('REPORT PASS',branch,step,split,flush=True)

def consolidate():
    objective=[];proxy=[];candidate=[]
    for branch in BRANCHES:
        for step in STEPS:
            own={}
            for split in ('normal','hard'):
                out=output_dir(branch,step,split)
                if not (out/'report_integrity.json').exists():continue
                own[split]=load(out/'trajectory.json');objective.append(own[split]);proxy.append(load(out/'proxy_summary.json'))
            if not own:continue
            row=dict(branch=branch,checkpoint_update=step,scale=BRANCHES[branch],diagnostic_only=False,complete_normal_hard=len(own)==2)
            for split in ('normal','hard'):
                t=own.get(split,{})
                row[split+'_same_qp_bitrate_reduction']=t.get('mean_same_qp_bitrate_reduction','')
                for metric,label in (('LPIPS','lpips'),('DISTS','dists'),('FloLPIPS','flo')):
                    d=t.get('equal_rate_'+metric+'_delta','');row[split+'_equal_rate_'+metric+'_delta']=d
                    row[split+'_'+metric+'_BD_rate']=t.get(metric+'_BD_rate','')
                    row[split+'_'+label+'_better']=(float(d)<0) if d!='' else ''
            candidate.append(row)
    if objective:write(ROOT/'objective_trajectory.csv',objective)
    if proxy:write(ROOT/'proxy_transformation_summary.csv',proxy)
    if candidate:write(ROOT/'validation_candidate_table.csv',candidate)
    return len(objective)==len(BRANCHES)*len(STEPS)*2

def finalize():
    check_gate();assert_old_unchanged();training_reports(True);assert consolidate()
    for branch in ['original']+list(BRANCHES):
        for step in ([0] if branch=='original' else STEPS):
            if branch!='original':
                record=load(ROOT/'branches'/branch/'checkpoint_hashes.json')[str(step)]
                assert sha(record['path'])==record['sha256']
            for split in ('normal','hard'):
                collect(branch,step,split)
                out=output_dir(branch,step,split);integrity=load(out/'report_integrity.json');assert integrity['status']=='PASS'
                for file,digest in integrity['files'].items():assert sha(out/file)==digest
    files=['qp_train_eval_semantics_audit.json','gvc_lambda_schedule_audit.json','paired_sampling_audit.json','initialization_audit.json',
        'validation_normal_manifest.json','validation_hard_manifest.json','loss_component_statistics.csv','loss_component_trajectory.csv',
        'gradient_balance_diagnostics.csv','proxy_transformation_summary.csv','objective_trajectory.csv','validation_candidate_table.csv']
    assert all((ROOT/p).exists() for p in files)
    dump(ROOT/'final_integrity.json',dict(status='PASS',revision=REVISION,completed_unix=time.time(),
        old_source_hashes_unchanged=True,train_manifest_exact_reuse=True,no_UVG_VIRAT=True,no_final_test_used=True,
        normal_hard_train_test_disjoint=True,qp_semantics_equal=True,uniform_qp_sampling=True,fresh_optimizer_each_branch=True,
        identical_initialization=True,paired_branch_sampling=True,real_RANS_complete=True,independent_decode_complete=True,
        all_requested_checkpoints_evaluated=True,all_measured_metrics_finite=True,
        bd_rate_undefined_policy='Retain invalid status/reason; never fabricate or monotonicize measurements',
        required_branches=list(BRANCHES),checkpoints=STEPS,validation_points=(1+len(BRANCHES)*len(STEPS))*2*32*6,
        source_checkpoint_sha256=load(ROOT/'fullqp_config.json')['source_checkpoint_sha256'],artifacts={p:sha(ROOT/p) for p in files}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['original']+list(BRANCHES));p.add_argument('--step',type=int,default=0)
    p.add_argument('--split',choices=('normal','hard'));p.add_argument('--training',action='store_true');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.branch:assert a.split;report_one(a.branch,a.step,a.split)
    if a.training:training_reports()
    consolidate()
    if a.finalize:finalize()
if __name__=='__main__':main()
