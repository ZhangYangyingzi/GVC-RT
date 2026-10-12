from io21 import *
from index_io import read_file_index,read_upload_manifest

def main():
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified'];idx=read_file_index();verified=[]
 for r in idx:
  if 'shards' in r:
   h=hashlib.sha256();size=0
   for s in r['shards']:
    p=ROOT/s['path'];assert sha(p)==s['sha256'] and p.stat().st_size==s['bytes']<=524288;h.update(p.read_bytes());size+=p.stat().st_size
   assert h.hexdigest()==r['sha256'] and size==r['bytes'];verified.append(dict(path=r['path'],bytes=size,sha256=r['sha256'],status='PASS'))
 dump(ROOT/'audits/text_reassembly_integrity.json',dict(status='PASS',files=verified,max_shard_bytes=524288))
 required=['config.json','protocol.json','diagnostic_plan.json','manifests/mcljcv.json','audits/checkpoint_integrity.json','audits/diagnostic_integrity.json','audits/evaluation_integrity.json','audits/overlap_integrity.json','diagnostics/discriminator_ability.csv','diagnostics/matched_gradient_windows.csv','diagnostics/training_gradient_distributions.csv','results/mcljcv/per_sequence_all_qp.csv','results/mcljcv/equal_rate_summary.csv','results/mcljcv/fid_decomposition_raw.csv','results/mcljcv/fid_decomposition_equal_rate.csv','REPORT.md','RESULTS_SUMMARY.md','final_integrity.json','file_index.json']
 uploaded=read_upload_manifest();assert all(p in uploaded or any(r['path']==p and 'shards' in r for r in idx) for p in required)
 j=load(ROOT/'final_integrity.json');j.update(status='PASS',publication_verified=True,publication_commit=pub['commit'],shards_verified=True,remote_key_files=required);dump(ROOT/'final_integrity.json',j)
 s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',publication_verified=True,initial_verified_publication=pub['commit']);dump(ROOT/'pipeline_status.json',s)
 env=os.environ.copy();env['V627_PUBLICATION_PHASE']='final_integrity_verified'
 try:subprocess.run([PYTHON,'-B','-u',str(ROOT/'publish_github.py')],env=env,cwd=REPO,check=True)
 except Exception:
  s=load(ROOT/'pipeline_status.json');s.update(status='BLOCKED',phase='final_record_publication_failed');dump(ROOT/'pipeline_status.json',s);j=load(ROOT/'final_integrity.json');j.update(status='BLOCKED',publication_verified=False);dump(ROOT/'final_integrity.json',j);raise
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified'];dump(ROOT/'audits/remote_final_verification.json',dict(status='PASS',final_commit=pub['commit'],required_files=required,all_remote_blobs_verified=True));print('REMOTE FINAL PASS',pub['commit'],flush=True)
if __name__=='__main__':main()
