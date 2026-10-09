"""CPU regression checks; synthetic points never enter result tables."""
import ast
import math
import numpy as np
from audit_utils import *
from rd_analysis import compare
from stream_audit import structure

def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    old=module('v51_rd_reference_test',V41/'rd_analysis.py')
    summary=read(V5/'final_qp_summary.csv')
    for r in summary:r['external_qp']=int(r['qp'])
    a=[r for r in summary if r['method']=='original']
    for method in METHODS[1:]:
        c=[r for r in summary if r['method']==method]
        for metric in METRICS:
            e,b=compare(a,c,metric,'original',method);oe,ob=old.compare(a,c,metric,'original',method)
            assert e==oe and b==ob,(method,metric)
    for n in (4,6,10):
        a=[dict(external_qp=i,kbps=float(100*1.2**i),LPIPS=1/(i+1)) for i in range(n)]
        c=[dict(r,kbps=r['kbps']*.8) for r in a]
        e,b=compare(a,c,'LPIPS','original','test')
        assert e['status']==b['status']=='valid' and abs(b['BD_rate_percent']+20)<1e-9
        assert e['mean_equal_rate_delta']<0 and e['grid_points']==100
        c=[dict(r) for r in a];c[1]['LPIPS']=1.1
        e,b=compare(a,c,'LPIPS','original','test')
        assert e['status']=='valid' and b['status']=='invalid' and b['reason']
        c=[dict(r) for r in a];c[1]['kbps']=c[0]['kbps']
        e,b=compare(a,c,'LPIPS','original','test');assert e['status']==b['status']=='invalid'
    reused=load(point_path('ulong','clip8',0,0))
    frames=structure(Path(reused['bitstream_path']),0,64)
    assert frames[0]['actual_qp']==0 and frames[1]['actual_qp']==2 and frames[8]['actual_qp']==0
    assert sum(r['real_bits'] for r in frames)==reused['real_bytes']*8
    dump('implementation_selftest.json',dict(status='PASS',syntax=True,original_four_point_regression=True,
        exact_20_percent_BD_test=True,scopes_tested=[4,6,10],nonmonotone_BD_invalid=True,duplicate_rate_invalid=True,
        actual_header_QP_cycle=True,synthetic_values_used_only_in_tests=True))
    print('IMPLEMENTATION SELFTEST PASS')

if __name__=='__main__':main()
