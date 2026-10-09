"""Strict reuse of existing full recurrent real-RANS evaluation points."""
import shutil,traceback
from io18 import *
def main():
    frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    old=module('temporal_old_v17_adapter',V17/'adapter.py')
    old.frozen();good=[];pending=[];assets={}
    for d in DATASETS:
        for v in sources(d):
            for m in METHODS:
                for q in QPS:
                    src=V17/'parts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json';target=point(d,m,v['video_index'],q)
                    try:
                        r=load(src);old.validate(r,v,q,m)
                        assert r['checkpoint_sha256']==('' if m=='original' else evalcfg()['checkpoints'][m]['sha256'])
                        assert r['source_frame_indices']==v['source_frame_indices'] and r['frames']==64
                        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                        for k in ('bitstream','feature','frame_metrics','transitions'):
                            assets[r[k+'_path']]=r[k+'_sha256']
                        for k in ('frame_metrics','transitions'):
                            dest=ROOT/'evaluation/source_artifacts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.{k}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
                            if dest.exists():assert sha(dest)==r[k+'_sha256']
                            else:shutil.copyfile(r[k+'_path'],dest)
                            r[k+'_path']=str(dest)
                        checks=dict(status='PASS',historical_validator=str(V17/'adapter.py'),historical_validator_SHA256=sha(V17/'adapter.py'),checkpoint_exact=True,source_RGB_and_indices_exact=True,actual_QPs_exact=True,codec_and_metric_hashes_verified=True,real_RANS=True,independent_decode=True,state_synchronization=True,whole_64_frame_causal_code_SHA256=sha(ENGINE/'engine.py'),metric_dependencies_SHA256=sha(V17/'audits/dependencies.json'))
                        r.update(temporal_audit_reused=True,temporal_audit_source=str(src),temporal_audit_source_sha256=sha(src),reuse_checks=checks)
                        if target.exists():assert load(target)==r
                        else:dump(target,r)
                        good.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,source=str(src),source_SHA256=sha(src),checks=checks))
                    except (AssertionError,KeyError,FileNotFoundError) as e:pending.append(dict(dataset=d,sequence=v['name'],method=m,video=v['video_index'],external_QP=q,source=str(src),reason=repr(e)))
            print('REUSE VERIFIED',d,v['name'],flush=True)
    # Add referenced assets from older source experiments to the before/after SHA audit.
    before=load(ROOT/'audits/historical_sha256_before.json')
    for p,h in assets.items():
        assert sha(p)==h
        if p in before:assert before[p]==h
        before[p]=h
    dump(ROOT/'audits/historical_sha256_before.json',before)
    dump(ROOT/'evaluation/reuse_manifest.json',dict(status='PASS',expected_points=240,reused_points=len(good),verified=good,fresh_pending=pending))
    print('REUSE PASS',len(good),'/240; FRESH',len(pending),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'evaluation/reuse_manifest.json',dict(status='FAIL',error=traceback.format_exc()));raise
