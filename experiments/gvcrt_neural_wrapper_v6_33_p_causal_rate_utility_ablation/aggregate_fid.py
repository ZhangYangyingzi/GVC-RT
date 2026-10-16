"""Exact inherited pooled FID helper, overlapped with independent GPU work."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', OMP_NUM_THREADS='4')
from runtime import *

def manifest():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'evaluation/points').glob('*.json'))}

def main():
    receipt=ROOT/'evaluation/fid_aggregation_integrity.json'
    helper=V32/'report32.py'
    before=manifest();assert len(before)==1840
    if receipt.exists():
        prior=load(receipt)
        if prior['status']=='PASS' and prior['point_receipt_sha256']==before and prior['source_helper_sha256']==sha(helper) and prior['pooled_csv_sha256']==sha(ROOT/'evaluation/pooled_fid.csv'):
            print('FID verified cache reused',flush=True);return
    points=[load(ROOT/p)['result'] for p in before]
    for r in points:assert sha(r['feature_path'])==r['feature_sha256']
    old=module('causal_exact_fid32',helper)
    pools=old.pooled(points)
    from evaluate import write
    write('evaluation/pooled_fid.csv',pools)
    assert before==manifest()
    dump('evaluation/fid_aggregation_integrity.json',dict(status='PASS',source_helper=str(helper),source_helper_sha256=sha(helper),point_receipt_sha256=before,feature_files_all_rehashed=True,pooled_csv_sha256=sha(ROOT/'evaluation/pooled_fid.csv'),curves=len(pools),implementation='Exact inherited report32.pooled; float64 pooled features, ddof1, native resolution official Inception features'))
    print('FID aggregation PASS',len(pools),flush=True)

if __name__=='__main__':main()
