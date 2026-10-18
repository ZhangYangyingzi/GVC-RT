"""Byte/identity isolation of the unchanged second exposure; no test-derived tuning."""
from io32 import *
import collections,concurrent.futures

def main():
 prev=load(PREV/'data_isolation_audit.json');rows=load(ROOT/'plans/continuation.json')['updates'];sources={r['source_row']['source_path']:r['source_row']['source_sha256'] for r in rows};tests=prev['test_sources']
 assert prev['status']=='PASS' and not prev['source_sha256_intersection'] and not prev['source_path_intersection']
 assert {(r['source_row']['source_path'],r['source_row']['source_sha256']) for r in rows}=={(r['source_path'],r['source_sha256']) for r in prev['training_sources']}
 allfiles=dict(sources)
 for r in tests:allfiles[r['source_path']]=r['source_sha256']
 def verify(kv):
  p,h=kv;actual=sha(p);assert actual==h,(p,actual,h);st=Path(p).stat();return dict(path=p,sha256=actual,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,allfiles.items()))
 counts=collections.Counter((r['source_row']['domain'],r['source_row']['external_qp']) for r in rows);assert len(counts)==20 and set(counts.values())=={50}
 dump(ROOT/'audits/data_isolation.json',dict(prev,status='PASS',inherited_audit_path=str(PREV/'data_isolation_audit.json'),inherited_audit_sha256=sha(PREV/'data_isolation_audit.json'),second_exposure=True,new_unique_training_videos=0,new_unique_windows=0,additional_updates=1000,domain_QP_counts=[dict(domain=d,QP=q,updates=n) for (d,q),n in sorted(counts.items())],verified_source_files=verified,source_file_hashes_recomputed=True,additional_calibration_windows=0,additional_prefill_windows=0,near_duplicate_coverage='Inherited historical exclusions and exact source/file/frame identities; no new exhaustive perceptual near-duplicate search or claim about unknown official pretraining exposure'))
 print('DATA ISOLATION PASS',len(verified),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  error(dict(stage='data_isolation',error=traceback.format_exc()));raise
