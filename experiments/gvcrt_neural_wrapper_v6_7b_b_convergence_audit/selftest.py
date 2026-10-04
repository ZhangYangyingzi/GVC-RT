"""CPU verification of FID formula, bootstrap strata and frozen protocol."""
from v67b_io import *
def main():
    import numpy as np
    frozen();assert load(ROOT/'audits/fid_protocol_audit.json')['status']=='PASS'
    fn=fid_function();rng=np.random.RandomState(8);x=rng.normal(size=(32,12));y=x+.25
    self_fid=fn(x,x);shift_fid=fn(x,y)
    assert abs(self_fid)<1e-9 and abs(shift_fid-12*.25**2)<1e-9
    for d in DATASETS:
        p=load(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json');assert len(p['indices'])==20
        for draw in p['indices']:
            assert len(draw)==sum(v['frames'] for v in sources(d));offset=0
            for v in sources(d):
                selected=draw[offset:offset+v['frames']];assert all(offset<=i<offset+v['frames'] for i in selected);offset+=v['frames']
    sys.path.insert(0,str(V62B));old=module('v67b_test_rd',V62B/'report.py')
    a=[dict(external_qp=i,kbps=float(i+1),FID=20-i) for i in range(10)];b=[dict(r,FID=r['FID']-1) for r in a]
    eq,_=old.compare_rows(a,b,'FID','anchor','candidate');assert eq['status']=='valid' and abs(eq['mean_equal_rate_delta']+1)<1e-9
    dump(ROOT/'audits/cpu_selftest.json',dict(status='PASS',FID_self=self_fid,FID_known_mean_shift=shift_fid,sequence_stratified_bootstrap_pass=True,FID_PCHIP_adapter_pass=True))
    print('CPU SELFTEST PASS',flush=True)
if __name__=='__main__':main()
