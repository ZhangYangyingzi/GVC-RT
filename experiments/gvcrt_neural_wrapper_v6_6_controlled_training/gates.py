"""Validate training and prepare immutable evaluation references."""
from v66_io import *
def main():
    cfg=frozen();plans=load(ROOT/'audits/training_sequence_equivalence.json')['plans'];cps={A:cfg['baseline']}
    baseline=[json.loads(x) for x in (V64/'training_logs/vimeo_only.jsonl').read_text().splitlines()]
    calibration=read(ROOT/'audits/structure_guard_calibration_samples.csv');assert len(calibration)==32
    import ast
    for r,ref in zip(calibration,baseline):assert ast.literal_eval(r['frame_sha256'])==ref['frame_sha256']
    for b in BRANCHES:
        assert load(ROOT/'branches'/b/'training_status.json')['status']=='PASS'
        assert load(ROOT/'audits'/f'parameter_update_audit_{b[0]}.json')['status']=='PASS'
        rows=[json.loads(x) for x in (ROOT/'training_logs'/f'{b}.jsonl').read_text().splitlines()];assert len(rows)==1000
        for plan,r,ref in zip(plans,rows,baseline):
            assert all(r[k]==z for k,z in plan.items())
            assert r['frame_sha256']==ref['frame_sha256'] and r['actual_qps']==ref['actual_qps']
        hashes=load(ROOT/'branches'/b/'checkpoint_hashes.json');assert set(hashes)=={'0','250','500','1000'}
        for cp in hashes.values():assert sha(cp['path'])==cp['sha256']
        cps[b]=hashes['1000']
    dump(ROOT/'checkpoint_hashes.json',cps);dump(ROOT/'evaluation/config.json',dict(checkpoints=cps))
    ev=eval_adapter();old=module('v66_old_io',V64/'v64_io.py');old_ev=old.eval_adapter()
    for d in DATASETS:
        for v in sources(d):
            for src,dst in [('original','original'),('vimeo_only',A)]:
                for q in range(10):
                    original=old.point(d,src,v['video_index'],q);r=load(original);old_ev.validate(r,v,q,src)
                    r.update(method=dst,reused=True,reused_from=str(original),reused_point_sha256=sha(original));ev.validate(r,v,q,dst);dump(point(d,dst,v['video_index'],q),r)
    audit=load(ROOT/'audits/training_sequence_equivalence.json');audit['formal_logs_pass']=True;dump(ROOT/'audits/training_sequence_equivalence.json',audit)
    dump(ROOT/'audits/training_integrity.json',dict(status='PASS',same_source_checkpoint=True,same_sample_sequence=True,same_qp_sequence=True,updates=1000,baseline_retrained=False))
    print('TRAINING AND BASELINE GATES PASS',flush=True)
if __name__=='__main__':main()
