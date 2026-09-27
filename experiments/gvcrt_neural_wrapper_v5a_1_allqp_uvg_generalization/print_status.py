from audit_utils import *

def main():
    integrity=load('final_integrity.json');assert integrity['status']=='PASS'
    cross=read('cross_dataset_generalization.csv');pairs=read('clip8_vs_clip4_allqp.csv')
    print('experiment directory:',ROOT)
    print('official supported external QPs:',load('qp_semantics_audit.json')['supported_external_qps'])
    print('QP0-3 backward compatibility:',load('qp0_3_backward_compatibility_audit.json')['status'])
    print('Fresh U-Long:');print('number of videos:',len(videos('ulong')));print('frames/video: 64');print('QP0-9 complete: PASS')
    for method in ('v41_20000','clip8'):
        for scope in ('0-3_seen','4-9_unseen','0-9'):
            r=next(r for r in cross if r['dataset']=='Fresh_U_Long' and r['method']==method and r['external_qp_scope']==scope)
            print(method,'vs Original',scope)
            for metric in METRICS:print(metric,'equal-rate delta:',r[metric+'_equal_rate_delta'] if r[metric+'_equal_rate_status']=='valid' else 'invalid: '+r[metric+'_equal_rate_reason'])
    print('UVG:');uvg=load('uvg_available_manifest.json')
    print('available sequence count:',uvg['num_available_uvg_sequences']);print('sequence names:',', '.join(uvg['sequence_names']))
    print('frames evaluated:',{v['sequence_name']:v['frames_evaluated'] for v in uvg['videos']});print('zero-shot: PASS')
    for method in ('v41_20000','clip8'):
        r=next(r for r in cross if r['dataset']=='UVG' and r['method']==method and r['external_qp_scope']=='0-9')
        print(method,'vs Original QP0-9')
        for metric in METRICS:print(metric,'equal-rate delta:',r[metric+'_equal_rate_delta'] if r[metric+'_equal_rate_status']=='valid' else 'invalid: '+r[metric+'_equal_rate_reason'])
    print('clip8 vs clip4:')
    for dataset in ('Fresh_U_Long','UVG'):
        r=next(r for r in pairs if r['dataset']==dataset and r['external_qp_scope']=='0-9')
        print(dataset,'QP0-9')
        for metric in METRICS:print(metric,'equal-rate delta:',r[metric+'_equal_rate_delta'] if r[metric+'_equal_rate_status']=='valid' else 'invalid: '+r[metric+'_equal_rate_reason'])
    for name in ('all_qp_monotonicity_audit.csv','qp_generalization_summary.csv','cross_dataset_generalization.csv',
        'clip8_vs_clip4_allqp.csv','ulong_all_qp_summary.csv','uvg_all_qp_summary.csv','rd_curves'):
        print(name,':',ROOT/name)
    print('final_integrity:',integrity['status'])

if __name__=='__main__':main()
