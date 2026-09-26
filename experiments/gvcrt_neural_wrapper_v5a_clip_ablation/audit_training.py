import argparse
import collections
import math
import numpy as np
import torch
from v5_utils import ROOT,BRANCHES,load,dump,read,write,sha,checkpoint

AVERAGES=('LPIPS','DISTS','R_est_bpp','proxy_L1','PSNR','MS_SSIM')

def initial():
    audits={b:load(f'parts/init_{b}.json') for b in BRANCHES}
    a,c=audits.values();source=load('config.json')['source_checkpoint']
    checks=dict(initialized_from_same_v4_1_step20000=all(r['source_step']==20000 and r['source_checkpoint']==source and r['source_checkpoint_sha256']==sha(source) for r in audits.values()),
        optimizer_state_restored_both=all(r['optimizer_state_restored'] for r in audits.values()),
        optimizer_state_identical=a['optimizer_state_hash']==c['optimizer_state_hash'],
        compression_core_identical=a['hashes']['compression']==c['hashes']['compression'],
        compression_core_frozen=all(r['compression_trainable_parameters']==0 for r in audits.values()),
        saved_rng_restored_both=all(r['saved_rng_restored'] for r in audits.values()))
    for model in ('wrapper','bridge','generator'):
        checks[model+'_state_identical']=a['hashes'][model]==c['hashes'][model]
        checks[model+'_trainable']=all(r['trainable_parameters'][model]>0 for r in audits.values())
    for b,r in audits.items():
        checks[b+'_initialized_identically']=r['actual_resume_step']==0 and all(r[n+'_equal_to_checkpoint'] for n in ('wrapper','bridge','generator'))
    result=dict(checks,status='PASS' if all(checks.values()) else 'FAIL',branches=audits)
    dump('initialization_audit.json',result);assert result['status']=='PASS'
    normalize={};temporal={}
    for b,spec in BRANCHES.items():
        history=read(f'training_logs/training_{b}.csv');positions=read(f'training_logs/positions_{b}.csv')
        first=next(r for r in history if int(r['step'])==1);ps=[r for r in positions if int(r['step'])==1]
        denominator=spec['clip_length']-1;assert len(ps)==denominator
        differences={k:abs(float(first[k])-sum(float(r[k]) for r in ps)/denominator) for k in AVERAGES}
        differences['loss']=abs(float(first['loss'])-sum(float(r['frame_objective']) for r in ps)/denominator)
        assert max(differences.values())<1e-5
        normalize[b]=dict(clip_length=spec['clip_length'],denominator=denominator,first_real_update_mean_errors=differences,status='PASS')
        audit=load(f'parts/temporal_{b}.json');assert audit['status']=='PASS' and audit['temporal_gradient_through_DPB'] is False
        assert audit['BPTT_disabled'] and audit['first_frame_no_grad'] and audit['DPB_tensors_detached']
        assert audit['tests'] and all(r['previous_reconstruction_gradient']==r['previous_feature_gradient']=='None' for r in audit['tests'])
        temporal[b]=audit
    dump('loss_normalization_audit.json',dict(status='PASS',clip4_denominator=3,clip8_denominator=7,branches=normalize))
    dump('temporal_gradient_audit.json',dict(status='PASS',temporal_gradient_through_DPB=False,BPTT_disabled=True,
         first_frame_no_grad=True,method='actual autograd.grad(future loss, previous reconstruction/feature, allow_unused=True); DPB requires_grad and grad_fn checked every frame',branches=temporal))
    print('INITIALIZATION / NORMALIZATION / TEMPORAL AUDITS PASS',flush=True)

def final():
    initial();summaries=[];sampling={};initial_audit=load('initialization_audit.json')
    for b,spec in BRANCHES.items():
        done=load(f'parts/train_{b}_done.json');assert done['status']=='PASS'
        history=read(f'training_logs/training_{b}.csv');positions=read(f'training_logs/positions_{b}.csv')
        denominator=spec['clip_length']-1;updates=spec['updates']
        assert [int(r['step']) for r in history]==list(range(1,updates+1))
        assert len(positions)==21000 and len(history)*denominator==21000
        by_step=collections.defaultdict(list)
        for r in positions:by_step[int(r['step'])].append(r)
        for r in history:
            step=int(r['step']);ps=by_step[step]
            assert [int(x['position']) for x in ps]==list(range(1,denominator+1))
            assert int(r['num_p_frames'])==denominator and int(r['clip_length'])==spec['clip_length']
            assert int(r['last_frame_index'])-int(r['first_frame_index'])==denominator
            assert str(r['consecutive_frames']).lower()=='true'
            assert float(r['beta'])==load('config.json')['beta']
            for key in AVERAGES:assert abs(float(r[key])-sum(float(x[key]) for x in ps)/denominator)<1e-5
            assert abs(float(r['loss'])-sum(float(x['frame_objective']) for x in ps)/denominator)<1e-5
            assert all(math.isfinite(float(r[k])) for k in (*AVERAGES,'loss','wrapper_gradient_norm','bridge_gradient_norm','generator_gradient_norm'))
            for x in ps:
                objective=float(x['LPIPS'])+float(x['DISTS'])+float(r['beta'])*float(x['R_est_bpp'])+.01*float(x['proxy_L1'])
                assert abs(float(x['frame_objective'])-objective)<1e-5
        counts=collections.Counter(r['video'] for r in history);qp_counts=collections.Counter(int(r['qp']) for r in history)
        videos=[r['filename'] for r in load('train_manifest.json')['videos']]
        assert set(counts)<=set(videos) and set(qp_counts)<=set(range(4))
        crops={}
        for key in ('crop_x','crop_y','start'):
            values=[int(r[key]) for r in history]
            crops[key]=dict(min=min(values),max=max(values),mean=float(np.mean(values)),std=float(np.std(values)),histogram={str(k):v for k,v in sorted(collections.Counter(values).items())})
        assert all(0<=int(r['crop_x'])<=int(r['width'])-256 and 0<=int(r['crop_y'])<=int(r['height'])-256 for r in history)
        sampling[b]=dict(clip_length=spec['clip_length'],optimizer_update_count=updates,total_optimized_P_frames=updates*denominator,
            total_raw_decoded_frames=sum(int(r['raw_decoded_frames']) for r in history),accepted_raw_frames=updates*spec['clip_length'],
            raw_frame_definition='successful OpenCV read calls, including abandoned short samples; codec-internal seek frames not exposed by OpenCV',
            rejected_sample_attempts=sum(int(r['rejected_sample_attempts']) for r in history),
            per_video_sample_count={name:counts[name] for name in videos},per_QP_update_count={str(q):qp_counts[q] for q in range(4)},
            crop_and_start_distribution=crops,train_manifest_sha256=sha(ROOT/'train_manifest.json'))
        for position in range(1,denominator+1):
            ps=[r for r in positions if int(r['position'])==position]
            assert len(ps)==updates
            summaries.append(dict(branch=b,clip_length=spec['clip_length'],position=position,num_samples=len(ps),
                **{k:float(np.mean([float(r[k]) for r in ps])) for k in AVERAGES}))
        cp=torch.load(checkpoint(b),map_location='cpu',weights_only=True)
        assert cp['step']==updates and cp['global_step']==20000+updates and cp['optimized_P_frames']==21000
        assert len(cp['optimizer']['state'])==initial_audit['branches'][b]['optimizer_state_count']
        assert all(int(s['step'])==20000+updates and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in cp['optimizer']['state'].values())
        for g in cp['optimizer']['param_groups']:assert g['lr']==load('config.json')['optimizer'][g['name']+'_lr']
        assert done['compression_hash_unchanged'] and done['initial_hashes']['compression']==done['final_hashes']['compression']
        del cp
    write('training_position_summary.csv',summaries)
    dump('training_sampling_audit.json',dict(status='PASS',matched_optimized_P_frame_budget=21000,branches=sampling))
    audit=load('loss_normalization_audit.json');audit['all_completed_updates_checked']=True;dump('loss_normalization_audit.json',audit)
    dump('parts/training_audit_done.json',dict(status='PASS',updates={'clip4_control':7000,'clip8':3000},optimized_P_frames_per_branch=21000,
        loss_formula_verified_every_update=True,all_training_values_finite=True))
    print('FULL TRAINING AUDIT PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',action='store_true');args=p.parse_args()
    try:(initial if args.initial else final)()
    except Exception as exc:
        dump('parts/training_audit_failure.json',dict(status='FAIL',error=repr(exc)));raise
