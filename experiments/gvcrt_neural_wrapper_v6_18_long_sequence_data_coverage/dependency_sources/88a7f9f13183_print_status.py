from parallel_utils import *
def main():
    audit=load(ROOT/'parallel_run_integrity.json')
    print('Experiment A directory:',A);print('Experiment B directory:',B)
    print('parallel execution',audit['status']);print('runtime overlap seconds',audit['overlap_runtime_seconds']);print('compute overlap seconds',audit['compute_overlap_seconds'])
    print('Experiment A: source count 8; resolution 854x480; rate-accounting fps 20')
    cp=load(A/'checkpoint_audit.json')['checkpoints']['clip8'];print('clip8 checkpoint:',cp['path']);print('checkpoint sha256:',cp['sha256'])
    gates=read(A/'strict_per_qp_gate.csv')
    for r in gates:
        print('QP',r['external_qp'],'Original kbps',r['Original_kbps'],'clip8 kbps',r['clip8_kbps'],'rate change %',r['rate_change_percent'])
        print(' '.join(m+' delta '+r[m+'_delta'] for m in METRICS),'strict pass','YES' if r['strict_all_pass']=='True' else 'NO')
    print('strict-pass QPs:',sum(r['strict_all_pass']=='True' for r in gates),'/ 10')
    for r in read(A/'bd_rate.csv'):print(r['metric'],'BD-rate',r['BD_rate_percent'] if r['status']=='valid' else 'invalid: '+r['reason'])
    for m in ('original','clip8'):
        row=next(r for r in read(A/'virat_480p20_all_qp_summary.csv') if r['method']==m and int(r['external_qp'])==0)
        print('lowest',m,'QP0 kbps:',row['kbps'])
    print('A final_integrity',load(A/'final_integrity.json')['status'])
    for name in ('visualizations','strict_per_qp_gate.csv','virat_480p20_all_qp_summary.csv','rd_curves'):print(A/name)
    print('Experiment B:')
    for r in read(B/'module_attribution_summary.csv'):
        if r['method']=='ORIGINAL':continue
        print(r['dataset'],r['method'],'QP',r['QP'],'kbps',r['kbps'],' '.join(m+' delta '+r[m+'_delta'] for m in METRICS))
    print(B/'proxy_domain_statistics.csv');print('B final_integrity',load(B/'final_integrity.json')['status'])
    print(ROOT/'parallel_run_integrity.json')
if __name__=='__main__':main()
