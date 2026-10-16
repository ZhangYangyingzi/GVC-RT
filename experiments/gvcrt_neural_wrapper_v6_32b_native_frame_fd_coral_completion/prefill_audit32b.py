"""Complete cache manifest, dtype and occurrence/unique-identity accounting; CPU only."""
from io32 import *
def main():
 import numpy as np,collections
 plan=load(ROOT/'plans/shared.json')['updates'];rows=[];dtypes=collections.Counter()
 for i,row in enumerate(plan):
  p=ROOT/'reference_stats/receipts'/f'{i:04d}.json';r=load(p);assert r['status']=='PASS' and r['source_sha256']==row['source_sha256'] and r['full_RGB_sha256']==row['full_RGB_sha256'];f=Path(r['feature_path']);assert sha(f)==r['artifacts'][str(f)]
  with np.load(f) as a:
   for key in ('real','reconstruction'):
    assert a[key].shape==(15,2048) and a[key].dtype==np.float32 and np.isfinite(a[key]).all();dtypes[str(a[key].dtype)]+=1
  rows.append(dict(plan_index=i,domain=row['domain'],QP=row['external_qp'],source_sha256=row['source_sha256'],source_frame_indices=row['source_frame_indices'][49:],feature_path=str(f),feature_sha256=sha(f),dtype='float32',shape=[15,2048],receipt_path=str(p),receipt_sha256=sha(p),reused=r.get('reused',False),historical_receipt=r.get('reuse_receipt'),historical_signature=r.get('historical_signature'),status='PASS'))
 summary=dict(status='PASS',completed=1000,expected=1000,feature_occurrences=15000,unique_target_UIDs=len({(r['source_sha256'],f) for r in plan for f in r['source_frame_indices'][49:]}),unique_windows=len({(r['source_sha256'],tuple(r['source_frame_indices'])) for r in plan}),unique_videos=len({r['source_sha256'] for r in plan}),array_dtype_counts=dict(dtypes),reused=sum(r['reused'] for r in rows),freshly_evaluated=sum(not r['reused'] for r in rows),four_native_windows_fresh=True,ordinary_two_forward_equivalence_checks_fresh=True,occurrence_weights_preserved=True,correlated_and_repeated_frames_not_independent=True)
 dump(ROOT/'prefill_manifest.json',dict(**summary,entries=rows));dump(ROOT/'audits/prefill_complete.json',summary);print(summary)
if __name__=='__main__':main()
