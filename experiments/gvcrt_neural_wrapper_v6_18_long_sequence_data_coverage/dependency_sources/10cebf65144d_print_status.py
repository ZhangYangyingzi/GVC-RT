import json
from experiment_utils import ROOT,read

def main():
    load=lambda name:json.loads((ROOT/name).read_text())
    check=load('final_integrity.json'); resume=load('resume_audit.json'); selection=load('selected_checkpoint.json')
    status=lambda value:'PASS' if value else 'FAIL'
    print('experiment directory:',ROOT)
    print('resume checkpoint:',resume['source_checkpoint'])
    print('optimizer restore:',status(resume['optimizer_state_restored']))
    for step in (15000,20000): print(f'{step//1000}k complete:',status(check.get(f'step{step}_completed',False)))
    for row in read('convergence_summary.csv'):
        print('step:',row['step'])
        for key in ('mean_real_kbps','mean_LPIPS','mean_DISTS','mean_FloLPIPS','LPIPS_BD_rate','DISTS_BD_rate','FloLPIPS_BD_rate',
                    'mean_equal_rate_delta_LPIPS','mean_equal_rate_delta_DISTS','mean_equal_rate_delta_FloLPIPS'):
            metric=key.split('_BD_rate')[0]
            value=row[key] or ('invalid: '+row.get(metric+'_BD_reason','not available'))
            print(key+':',value)
    print('selected checkpoint step:',selection['step'])
    final=next(r for r in read('v4_1_vs_original_summary.csv') if r['qp']=='all')
    print('final selected vs Original mean rate change:',final['mean_rate_change_percent'])
    for metric in ('LPIPS','DISTS','FloLPIPS'):
        print('final '+metric+' BD-rate:',final[metric+'_BD_rate'] or 'invalid: '+final[metric+'_BD_reason'])
    for row in read('fid_final.csv'):
        if row['method'] in ('original','selected'): print('FID',row['method'],'QP'+row['qp']+':',row['FID'])
    for label,key in [('compression frozen','compression_core_frozen'),('independent decode','independent_decode_pass')]: print(label+':',status(check.get(key,False)))
    print('final_integrity:',check['status'])
    for path in ('convergence_summary.csv','checkpoint_selection.csv','perceptual_bd_rate.csv','equal_rate_perceptual_summary.csv','flolpips_final.csv','fid_final.csv','rd_curves'):
        print(path+':',ROOT/path)

if __name__=='__main__': main()
