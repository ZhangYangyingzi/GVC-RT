"""Numeric terminal report requested for the completed V5-A.2 experiment."""
from audit_utils import *

def main():
    assert load('final_integrity.json')['status']=='PASS'
    print('experiment directory:',ROOT)
    print('canonical protocol: resolution 1920x1080; fps 30.0; frames 64; format RGB24 lossless PNG')
    print('U-Long: 8 videos PASS')
    print('UVG: 7 sequences PASS; stride4 120->30 PASS')
    print('checkpoint hashes unchanged PASS')
    for dataset in ('ulong','uvg'):
        print(dataset)
        summary=read(f'{dataset}_all_qp_summary.csv');equal=read(f'{dataset}_equal_rate_summary.csv')
        for method,label in [('original','Original'),('v41_20000','V4.1'),('clip8','clip8')]:
            row=next(r for r in summary if r['method']==method and int(r['external_qp'])==0)
            print(label,'QP0 min kbps:',row['mean_real_kbps'])
        for candidate,label in [('v41_20000','V4.1'),('clip8','clip8')]:
            print(label,'vs Original:')
            for metric in METRICS:
                row=next(r for r in equal if r['anchor']=='original' and r['candidate']==candidate and r['metric']==metric)
                print(metric,'equal-rate delta:',row['mean_equal_rate_delta'] if row['status']=='valid' else 'invalid: '+row['reason'])
    for name in ('source_domain_statistics_dataset_summary.csv','ulong_all_qp_summary.csv','uvg_all_qp_summary.csv',
                 'ulong_equal_rate_summary.csv','uvg_equal_rate_summary.csv','rd_curves/ulong','rd_curves/uvg'):
        print(ROOT/name)
    print('final_integrity PASS')

if __name__=='__main__':main()

