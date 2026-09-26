import math
from v5_utils import ROOT,V41,BRANCHES,load,read,dump,sha,truth
from report import collect

def main():
    checks={}
    try:
        init=load('initialization_audit.json');norm=load('loss_normalization_audit.json');temporal=load('temporal_gradient_audit.json')
        sample=load('training_sampling_audit.json');training=load('parts/training_audit_done.json');cfg=load('config.json')
        for key in ('initialized_from_same_v4_1_step20000','clip4_control_initialized_identically','clip8_initialized_identically',
                    'optimizer_state_restored_both','compression_core_frozen','wrapper_trainable','bridge_trainable','generator_trainable'):
            checks[key]=init[key] is True
        checks['clip4_length']=cfg['branches']['clip4_control']['clip_length']==4
        checks['clip8_length']=cfg['branches']['clip8']['clip_length']==8
        checks['clip4_loss_denominator']=norm['clip4_denominator']==3
        checks['clip8_loss_denominator']=norm['clip8_denominator']==7
        checks['clip4_optimized_P_frames']=sample['branches']['clip4_control']['total_optimized_P_frames']==21000
        checks['clip8_optimized_P_frames']=sample['branches']['clip8']['total_optimized_P_frames']==21000
        checks['temporal_DPB_detach_unchanged']=temporal['status']=='PASS' and temporal['temporal_gradient_through_DPB'] is False
        checks['BPTT_disabled']=temporal['BPTT_disabled'] is True and all(r['DPB_tensors_detached'] for r in temporal['branches'].values())
        checks['first_frame_no_grad_unchanged']=temporal['first_frame_no_grad'] is True
        checks['loss_unchanged_except_clip_average_denominator']=norm['all_completed_updates_checked'] and training['loss_formula_verified_every_update']
        checks['beta_unchanged']=cfg['beta']==load(V41/'config.json')['beta']==17.302782275540444
        checks['compression_hash_unchanged']=all(load(f'parts/train_{b}_done.json')['compression_hash_unchanged'] for b in BRANCHES)
        checks['training_audit_pass']=training['status']==sample['status']==init['status']==norm['status']=='PASS'
        checks['previous_experiments_unchanged']=all(sha(p)==h for p,h in load('source_snapshot.json').items())
        checks['manifests_and_protocol_identical']=all(sha(ROOT/p)==sha(V41/p) for p in ('train_manifest.json','validation_manifest.json','test_manifest.json','metric_implementation_audit.json'))
        val,final=collect('validation'),collect('final')
        checks['validation_complete']=len(val)==96
        checks['final_test_complete']=len(final)==128
        checks['real_RANS_complete']=all(int(r['real_bytes'])==int(r['bytes_consumed'])>0 for r in val+final)
        checks['independent_decode_pass']=all(truth(r['independent_decode_pass']) and truth(r['metric_decode_pass']) and truth(r['position_decode_pass']) for r in val+final)
        for metric in ('LPIPS','DISTS','FloLPIPS'):checks[metric+'_complete']=all(math.isfinite(float(r[metric])) for r in val+final)
        vf,ff=read('fid_validation.csv'),read('fid_final.csv')
        checks['FID_complete']=len(vf)==len(ff)==16 and all(int(r['num_real_samples'])==int(r['num_reconstruction_samples'])==count and math.isfinite(float(r['FID'])) for rows,count in ((vf,192),(ff,512)) for r in rows)
        equal=read('validation_equal_rate_summary.csv')+read('final_equal_rate_summary.csv')
        bd=read('validation_perceptual_bd_rate.csv')+read('final_perceptual_bd_rate.csv')
        checks['equal_rate_analysis_complete']=len(equal)==32 and all(r['status']=='valid' and math.isfinite(float(r['mean_equal_rate_delta'])) or r['status']=='invalid' and bool(r['reason']) for r in equal)
        checks['BD_rate_complete_or_validly_skipped']=len(bd)==32 and all(r['status']=='valid' and math.isfinite(float(r['BD_rate_percent'])) or r['status']=='invalid' and bool(r['reason']) for r in bd)
        checks['no_final_test_checkpoint_selection']=load('parts/validation_done.json')['checkpoint_selection'] is False and load('parts/final_done.json')['checkpoint_selection'] is False
        checks['validation_finished_before_final']=load('parts/validation_done.json')['validation_sha256']==sha(ROOT/'checkpoint_validation.csv')
        positions=read('training_position_summary.csv');checks['training_positions_complete']=len(positions)==10 and {int(r['position']) for r in positions if r['branch']=='clip8'}==set(range(1,8))
        vp=read('validation_temporal_position_summary.csv');checks['validation_positions_complete']=len(vp)==4*5*(31+3)
        checks['plots_complete']=all((ROOT/'rd_curves'/f'{s}_Rate_{m}.{e}').stat().st_size>0 for s in ('validation','final') for m in ('LPIPS','DISTS','FloLPIPS','FID') for e in ('png','pdf'))
        result_gate=load('v5a_result_gate.json');expected={r['metric']:r for r in read('final_equal_rate_summary.csv') if r['anchor']=='clip4_control' and r['method']=='clip8'}
        correct=True
        for metric,key in [('FloLPIPS','clip8_temporal_improvement'),('LPIPS','clip8_lpips_nonworse'),('DISTS','clip8_dists_nonworse'),('FID','clip8_fid_nonworse')]:
            r=expected[metric];value=float(r['mean_equal_rate_delta']) if r['status']=='valid' else None
            flag=(value<0 if metric=='FloLPIPS' else value<=0) if value is not None else None
            correct=correct and result_gate[key]==flag
        checks['result_gate_matches_raw_numbers']=correct
        checks['all_values_finite']=training['all_training_values_finite'] and all(math.isfinite(float(r[k])) for r in val+final for k in ('kbps','bpp','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'))
        result=dict(checks,status='PASS' if all(checks.values()) else 'FAIL',failed_checks=[k for k,v in checks.items() if not v],
            actual_values=dict(clip4_length=4,clip8_length=8,clip4_loss_denominator=3,clip8_loss_denominator=7,
                clip4_optimizer_updates=7000,clip8_optimizer_updates=3000,clip4_optimized_P_frames=21000,clip8_optimized_P_frames=21000),
            missing_source_rng_states=init['branches']['clip4_control']['missing_rng_states'],
            integrity_status_is_not_a_research_success_gate=True)
    except Exception as exc:
        result=dict(checks,status='FAIL',error=repr(exc));dump('final_integrity.json',result);raise
    dump('final_integrity.json',result);assert result['status']=='PASS',result['failed_checks']
    print('FINAL INTEGRITY PASS',flush=True)

if __name__=='__main__':main()
