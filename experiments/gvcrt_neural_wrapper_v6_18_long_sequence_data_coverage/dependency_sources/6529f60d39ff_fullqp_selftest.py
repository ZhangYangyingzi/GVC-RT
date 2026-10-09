"""CPU regression checks. Synthetic arrays remain in memory only."""
import ast
import math
from fullqp_common import *
def main():
    cfg=check_gate();assert_old_unchanged();checks=[]
    for p in list(ROOT.glob('fullqp_*.py'))+[ROOT/'run_fullqp.py']:ast.parse(p.read_text())
    checks.append('all_source_files_parse')
    assert len(cfg['branches'])==5 and set(cfg['branches'])==set(BRANCHES)
    assert cfg['gpus']==[4,5,6,7] and cfg['updates']==3000 and cfg['checkpoint_steps']==STEPS
    for b,s in BRANCHES.items():
        mapping=read(ROOT/'branches'/b/'qp_lambda_beta_mapping.csv')
        assert len(mapping)==10
        for q,r in enumerate(mapping):
            assert int(r['external_qp'])==q
            assert math.isclose(float(r['lambda_q']),lambda_q(q),rel_tol=1e-12)
            assert math.isclose(float(r['beta_q']),beta_q(b,q),rel_tol=1e-12)
    checks.append('five_branch_mapping_tables_match_request')
    assert math.isclose(lambda_q(0),.08) and math.isclose(lambda_q(9),.9)
    audit=load(ROOT/'qp_train_eval_semantics_audit.json')
    assert len(audit['rows'])==10 and all(r['exact_match'] and r['training']==r['evaluation'] for r in audit['rows'])
    checks.append('all_ten_real_implementation_qp_sequences_match')
    for split in ('normal','hard'):
        m=load(ROOT/f'validation_{split}_manifest.json');assert m['count']==32
        for v in m['videos']:
            assert Path(v['path']).is_file() and sha(v['path'])==v['sha256']
            assert v['dataset']==split and v['frames']==32 and v['rate_fps']==v['fps']
    checks.append('64_extracted_validation_sources_verified')
    sys.path.insert(0,str(V52));from rd_analysis import compare
    a=[dict(kbps=2**i,LPIPS=1/(i+1),external_qp=q) for i,q in enumerate(QPS)]
    e,b=compare(a,a,'LPIPS','unit_anchor','unit_identity')
    assert e['status']==b['status']=='valid' and abs(e['mean_equal_rate_delta'])<1e-12 and abs(b['BD_rate_percent'])<1e-10
    c=[dict(r,kbps=r['kbps']*1000) for r in a];e,_=compare(a,c,'LPIPS','unit_anchor','unit_disjoint')
    assert e['status']=='invalid'
    c=[dict(r) for r in a];c[2]['LPIPS']=c[0]['LPIPS']*2
    _,b=compare(a,c,'LPIPS','unit_anchor','unit_nonmonotone');assert b['status']=='invalid'
    checks.append('shared_pchip_identity_no_extrapolation_no_monotonicization')
    dump(ROOT/'fullqp_selftest_audit.json',dict(status='PASS',checks=checks,synthetic_metrics_saved=False))
    print('SELFTEST PASS',checks,flush=True)
if __name__=='__main__':main()
