"""Read-only validation of available points; explicitly partial, never final PASS."""
from preflight import *
from data import load
from bootstrap import BRANCHES
from evaluation_stage import expected_signatures,receipt_valid,task_name,METHODS
from evaluate import csvwrite
import csv,math

def main():
 import numpy as np
 manifest=load(R/'eval_manifest.json');signatures=expected_signatures();qs=load(R/'protocol.json')['q_grid']
 expected=[dict(dataset=d,video=i,branch=m,q=q) for d,vs in manifest.items() for i in range(len(vs)) for m in METHODS for q in qs]
 valid=[];missing=[];invalid=[];gt={};counts={};artifacts={}
 for t in expected:
  path=R/f'results/points/{task_name(t)}.json'
  if not path.exists():missing.append(t);continue
  try:
   assert receipt_valid(t,signatures),'receipt validation failed'
   x=load(path);r=x['result']
   with open(r['per_frame'],newline='') as f:ff=list(csv.DictReader(f))
   with open(r['per_transitions'],newline='') as f:tt=list(csv.DictReader(f))
   assert [int(v['frame']) for v in ff]==list(range(64))
   assert [(int(v['from_frame']),int(v['to_frame'])) for v in tt]==[(i,i+1) for i in range(63)]
   assert all(math.isfinite(float(v[k])) for v in ff for k in ('LPIPS','DISTS','L1'))
   assert all(math.isfinite(float(v['FloLPIPS'])) for v in tt)
   with np.load(r['feature']) as z:
    assert z['real'].shape==z['reconstruction'].shape==(64,2048)
    assert np.isfinite(z['real']).all() and np.isfinite(z['reconstruction']).all()
    h=hashlib.sha256(z['real'].tobytes()).hexdigest()
    assert gt.setdefault((t['dataset'],t['video']),h)==h,'GT feature mismatch'
   valid.append(dict(r,reused='reuse' in x));artifacts[str(path)]=sha(path)
   key=(t['dataset'],t['branch']);counts[key]=counts.get(key,0)+1
  except Exception:invalid.append(dict(task=t,error=traceback.format_exc()))
 coverage=[dict(dataset=d,method=m,completed=counts.get((d,m),0),expected=len(vs)*len(qs),status='complete' if counts.get((d,m),0)==len(vs)*len(qs) else 'partial') for d,vs in manifest.items() for m in METHODS]
 csvwrite(R/'results/partial_coverage.csv',coverage)
 csvwrite(R/'results/partial_per_video_all_q.csv',valid)
 save('audits/progress_integrity.json',dict(status='PARTIAL' if len(valid)!=len(expected) or invalid else 'AVAILABLE_POINTS_VALIDATED',not_final_integrity=True,time=time.time(),evaluation_completed=len(valid),evaluation_expected=len(expected),reused=sum(r['reused'] for r in valid),new=sum(not r['reused'] for r in valid),missing=missing,invalid=invalid,coverage=coverage,per_point_frames=64,per_point_transitions=63,GT_features_identical=True,GT_feature_hashes=[dict(dataset=d,video=v,sha256=h) for (d,v),h in gt.items()],receipt_sha256=artifacts,updates={b:load(R/f'branches/{b}/status.json')['updates'] for b in BRANCHES},final_report_and_publication_pending=True))
 print('AVAILABLE POINTS VALIDATED',len(valid),'/',len(expected),'invalid',len(invalid),flush=True)
 assert not invalid,invalid
if __name__=='__main__':main()
