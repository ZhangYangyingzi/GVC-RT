"""CPU protocol checks; no synthetic measurements enter experiment tables."""
import ast
import math
from v62b_io import *
from report import compare_rows
def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    cfg=check_frozen();assert cfg['expected_points']==930 and cfg['no_training'] and cfg['no_model_selection']
    assert cfg['external_qps']==list(range(10)) and cfg['gpus']==[4,5,6,7]
    manifest=load(ROOT/'source_manifest.json')['videos']
    assert {d:len(sources(d)) for d in DATASETS}==dict(ulong=8,uvg=7,virat720=8,virat480=8)
    for v in manifest:
        assert v['rate_accounting_fps']==(20.0 if v['dataset'].startswith('virat') else 30.0)
        assert len(v['source_frame_indices'])==len(v['frame_rgb_sha256'])==v['frames']
    # Comparator invariants using in-memory arrays, not experiment result files.
    for metric in METRICS:
        a=[dict(kbps=2**i,external_qp=i,**{metric:i+1 if metric in HIGHER else 1/(i+1)}) for i in range(10)]
        e,b=compare_rows(a,a,metric,'unit_anchor','unit_identity')
        assert e['status']==b['status']=='valid' and abs(e['mean_equal_rate_delta'])<1e-12 and abs(b['BD_rate_percent'])<1e-10
        c=[dict(r,kbps=r['kbps']*10000) for r in a];e,_=compare_rows(a,c,metric,'unit_anchor','unit_out_of_range');assert e['status']=='invalid'
        c=[dict(r) for r in a];c[5][metric]=c[0][metric];_,b=compare_rows(a,c,metric,'unit_anchor','unit_nonmonotone');assert b['status']=='invalid'
    dump(ROOT/'selftest_audit.json',dict(status='PASS',syntax=True,cohort_counts=True,fps_rules=True,frozen_candidates=True,
        unchanged_PCHIP_protocol=True,identity_zero_delta=True,no_extrapolation=True,nonmonotone_BD_invalid=True,synthetic_results_saved=False))
    print('SELFTEST PASS',flush=True)
if __name__=='__main__':main()
