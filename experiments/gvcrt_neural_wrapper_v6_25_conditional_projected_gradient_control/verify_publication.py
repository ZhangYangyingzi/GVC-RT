from io21 import *
def main():
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified'];idx=load(ROOT/'file_index.json');rows=[]
 for r in idx['files']:
  if 'shards' in r:
   h=hashlib.sha256();size=0
   for s in r['shards']:
    p=ROOT/s['path'];assert sha(p)==s['sha256'] and p.stat().st_size==s['bytes']<=524288;h.update(p.read_bytes());size+=p.stat().st_size
   assert h.hexdigest()==r['sha256'] and size==r['bytes'];rows.append(dict(path=r['path'],sha256=r['sha256'],bytes=size,status='PASS'))
 dump(ROOT/'audits/text_reassembly_integrity.json',dict(status='PASS',files=rows,max_shard_bytes=524288))
 required=['config.json','protocol.json','final_integrity.json','file_index.json','RESULTS_NUMBERS_ONLY.md','historical/fid_decomposition_raw.csv','historical/fid_decomposition_equal_rate.csv',*[f'branches/{b}/gradient_summary.csv' for b in BRANCHES],*[f'audits/{b}_pretraining.json' for b in BRANCHES],*[f'results/ip_-1/{d}/equal_rate_summary.csv' for d in DATASETS]]
 published=load(ROOT/'upload_manifest.json')['files'];assert all(p in published or any(f['path']==p and 'shards' in f for f in idx['files']) for p in required)
 final=load(ROOT/'final_integrity.json');final.update(status='PASS',publication_verified=True,publication_commit=pub['commit'],shards_verified=True,remote_key_files=required);dump(ROOT/'final_integrity.json',final)
 # Publish the verified integrity record normally. Remote blob verification is repeated.
 env=os.environ.copy();env['V625_PUBLICATION_PHASE']='final_integrity_verified';subprocess.run([PYTHON,'-B','-u',str(ROOT/'publish_github.py')],env=env,cwd=REPO,check=True)
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified'];dump(ROOT/'audits/remote_final_verification.json',dict(status='PASS',final_commit=pub['commit'],required_files=required,all_remote_blobs_verified=True))
if __name__=='__main__':main()
