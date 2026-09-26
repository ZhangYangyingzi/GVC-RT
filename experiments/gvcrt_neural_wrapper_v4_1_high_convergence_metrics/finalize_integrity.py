import ast
import json
import math
import torch
from experiment_utils import ROOT,V4,sha,dump,read
from report import collect,load_json,truth

def normalized_statement(source, needle):
    tree=ast.parse(source)
    return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
            if isinstance(n,ast.stmt) and ast.get_source_segment(source,n)==needle]

def main():
    checks={}
    try:
        audit=load_json('resume_audit.json'); cfg=load_json('config.json'); source=load_json('source_snapshot.json')
        checks['original_v4_files_unchanged']=all(sha(path)==digest for path,digest in source.items())
        initial=ROOT/'checkpoints/beta_high/step_10000.pt'
        checks['initialized_from_v4_beta_high_step10000']=(audit['source_step']==audit['actual_resume_step']==10000 and
            sha(initial)==sha(cfg['source_checkpoint']) and all(audit[n+'_equal_before_resume'] for n in ('wrapper','bridge','generator')))
        checks['optimizer_state_restored']=audit['optimizer_state_restored'] is True and audit['optimizer_parameter_state_count']>0
        checks['saved_rng_states_restored']=all(audit[k] is True for k in ('sampling_rng_restored','torch_rng_restored','cuda_rng_restored'))
        for step in (15000,20000):
            path=ROOT/'checkpoints/beta_high'/f'step_{step}.pt'; cp=torch.load(path,map_location='cpu',weights_only=True)
            ok=cp['step']==step and cp['beta']==cfg['beta'] and cp['branch']=='beta_high'
            ok=ok and len(cp['optimizer']['state'])==audit['optimizer_parameter_state_count']
            ok=ok and all(int(s['step'])==step and torch.isfinite(s['exp_avg']).all().item() and
                         torch.isfinite(s['exp_avg_sq']).all().item() for s in cp['optimizer']['state'].values())
            checks[f'step{step}_completed']=bool(ok)
            del cp
        pa=load_json('parts/parameter_audit_step_10000.json')
        for name in ('wrapper','bridge','generator'): checks[name+'_trainable']=pa['trainable'][name]>0
        checks['compression_core_frozen']=pa['compression_trainable_parameters']==audit['compression_trainable_parameters']==0
        checks['compression_hash_unchanged']=all(load_json(f'parts/parameter_audit_step_{s}.json')['compression_hash']==audit['compression_hash'] for s in (15000,20000))
        history=read('training_logs/training_beta_high_extension.csv')
        checks['extension_training_history_complete']=[int(r['step']) for r in history]==list(range(10001,20001))
        numeric=('beta','qp','start','crop_x','crop_y','loss','LPIPS','DISTS','R_est_bpp','proxy_L1','PSNR','MS_SSIM',
                 'wrapper_gradient_norm','bridge_gradient_norm','generator_gradient_norm')
        checks['training_all_values_finite']=all(truth(r['all_finite']) and all(math.isfinite(float(r[k])) for k in numeric) for r in history)
        checks['loss_unchanged']=all(abs(float(r['loss'])-(float(r['LPIPS'])+float(r['DISTS'])+cfg['beta']*float(r['R_est_bpp'])+.01*float(r['proxy_L1'])))<1e-5 for r in history)
        train_source=(ROOT/'train_extension.py').read_text(); original_source=(V4/'train.py').read_text()
        detach="p_model.add_ref_frame(captured[-1].detach(),(reconstruction.detach()*2-1).half())"
        compact=lambda s: ''.join(s.split())
        checks['temporal_detach_unchanged']=compact(detach) in compact(train_source) and compact(detach) in compact(original_source)
        checks['clip_length_4']=cfg['clip_length']==4 and 'frames[1:]' in train_source
        val=collect('validation'); final=collect('final')
        checks['validation_6_videos_complete']=len(val)==120 and len({r['video_index'] for r in val})==6
        checks['validation_all_4_qps_complete']=all(len([r for r in val if r['method']==m and r['video_index']==i])==4 for m in {r['method'] for r in val} for i in range(6))
        checks['real_rans_validation_complete']=all(int(r['real_bytes'])==int(r['bytes_consumed'])>0 for r in val)
        checks['independent_decode_pass']=all(truth(r['independent_decode_pass']) and truth(r['metric_decode_pass']) for r in val+final)
        checks['final_8_ulong_complete']=len(final)==128 and len({r['video_index'] for r in final})==8 and all(r['num_frames']==64 for r in final)
        for metric in ('LPIPS','DISTS','FloLPIPS'): checks[metric+'_complete']=all(math.isfinite(float(r[metric])) for r in val+final)
        vf,ff=read('fid_validation.csv'),read('fid_final.csv')
        checks['FID_complete']=len(vf)==20 and len(ff)==16 and all(int(r['num_real_samples'])==int(r['num_reconstruction_samples'])==count and math.isfinite(float(r['FID'])) for rs,count in ((vf,192),(ff,512)) for r in rs)
        ers=read('equal_rate_perceptual_summary.csv')+read('final_equal_rate_perceptual_summary.csv')
        brs=read('perceptual_bd_rate.csv')+read('final_perceptual_bd_rate.csv')
        checks['equal_rate_analysis_complete']=len(ers)==21 and all(r['status']=='valid' and math.isfinite(float(r['mean_equal_rate_delta'])) or r['status']=='invalid' and bool(r['reason']) for r in ers)
        checks['BD_rate_analysis_complete_or_validly_skipped']=len(brs)==21 and all(r['status']=='valid' and math.isfinite(float(r['BD_rate_percent'])) or r['status']=='invalid' and bool(r['reason']) for r in brs)
        selection=load_json('selected_checkpoint.json')
        checks['checkpoint_selected_without_final_test']=(selection['status']=='PASS' and selection['used_final_test'] is False and
            selection['validation_sha256']==sha(ROOT/'checkpoint_validation.csv') and
            selection['selection_csv_sha256']==sha(ROOT/'checkpoint_selection.csv') and
            selection['checkpoint_sha256']==sha(selection['checkpoint']))
        choices=[r for r in read('checkpoint_selection.csv') if truth(r['eligible'])]
        best=max(choices,key=lambda r:(float(r['real_bitrate_reduction_percent']),-int(r['step'])))
        checks['selection_ranking_verified']=int(best['step'])==selection['step']
        checks['fixed_manifests_unchanged']=all(sha(ROOT/f'{s}_manifest.json')==sha(V4/f'{s}_manifest.json') for s in ('train','validation','test'))
        checks['metric_implementation_audit_pass']=load_json('metric_implementation_audit.json')['status']=='PASS'
        checks['rd_plots_complete']=all((ROOT/'rd_curves'/f'{s}_Rate_{m}.{ext}').stat().st_size>0 for s in ('validation','final') for m in ('LPIPS','DISTS','FloLPIPS','FID') for ext in ('png','pdf'))
        checks['all_values_finite']=checks['training_all_values_finite'] and all(math.isfinite(float(r[k])) for r in val+final for k in ('kbps','bpp','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'))
        result=dict(checks,missing_rng_states=audit['missing_rng_states'],
                    missing_rng_note='Original checkpoint did not save global Python or NumPy RNG; recorded as missing, never claimed restored.',
                    status='PASS' if all(checks.values()) else 'FAIL',failed_checks=[k for k,v in checks.items() if not v])
    except Exception as exc:
        result=dict(checks,status='FAIL',error=repr(exc))
        dump('final_integrity.json',result); raise
    dump('final_integrity.json',result)
    assert result['status']=='PASS',result['failed_checks']
    print('FINAL INTEGRITY PASS',flush=True)

if __name__=='__main__': main()
