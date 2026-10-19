"""Isolated ordinary publication of only this experiment directory."""
import json,hashlib,os,subprocess,sys,time,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1];REL=R.relative_to(REPO).as_posix();TMP=Path('/Huang_group/zyyz/home_dir/tmp/gvc-rt-v636-publish')
CRITICAL=['config.json','protocol.json','objective_definition.json','OBJECTIVE_COMPARISON.md','audits/reuse_pointwise_verification.json','branches/B_basicLoss_noP_500/gradient_summary.csv','results/all_rd_points.csv','results/pooled_fid.csv','results/equal_rate_summary.csv','REPORT.md','final_integrity.json','file_index.json']
def run(args,cwd=None,check=True):
 p=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
 if check and p.returncode:raise RuntimeError(f'{args} failed {p.returncode}: {p.stderr[-3000:]}')
 return p
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def light_files():
 excluded={'checkpoints','bitstreams','features','raw','__pycache__','.git'};out=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or any(x in excluded for x in p.parts):continue
  rel=p.relative_to(R).as_posix()
  if p.suffix in ('.pt','.pth','.ckpt','.bin','.npz','.npy','.mp4') or (p.suffix=='.png' and not (rel.startswith('plots/') or rel.startswith('figures/'))):continue
  if rel.endswith('.lock') or rel.startswith('logs/') and rel.endswith('.log'):continue
  if p.name in ('file_index.json','publication_status.json'):continue
  out.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
 return out
def main():
 if run(['git','-C',str(REPO),'status','--porcelain'],check=True).stdout is None:pass
 url=run(['git','-C',str(REPO),'remote','get-url','origin']).stdout.strip()
 assert 'ZhangYangyingzi/GVC-RT' in url,url
 payload=light_files();json.dump({'schema':'v636_file_index','root':REL,'files':payload,'binary_artifacts_excluded':'Checkpoints, bitstreams, features, video and tensor binary files remain server-side; see server_artifacts_index.json.'},open(R/'file_index.json','w'),indent=2);open(R/'file_index.json','a').write('\n')
 # Isolated worktree starts from the current remote main, preserving history.
 if TMP.exists():
  run(['git','-C',str(REPO),'worktree','remove','--force',str(TMP)],check=False)
 run(['git','-C',str(REPO),'fetch','origin','main'])
 remote=run(['git','-C',str(REPO),'rev-parse','origin/main']).stdout.strip()
 exists=run(['git','-C',str(REPO),'cat-file','-e',f'origin/main:{REL}/PROTOCOL.md'],check=False)
 if exists.returncode==0:raise RuntimeError('remote already contains v6.36 directory; refusing overwrite/merge')
 run(['git','-C',str(REPO),'worktree','add','-b','codex/v636-basic-full-'+str(int(time.time())),str(TMP),'origin/main'])
 dst=TMP/REL;dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copytree(R,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.pth','*.ckpt','*.bin','*.npz','*.npy','*.mp4','*.log','*.lock','__pycache__','checkpoints','bitstreams','features','raw'))
 # Ensure only intended directory is staged.
 run(['git','-C',str(TMP),'add','--all','--',REL])
 if run(['git','-C',str(TMP),'diff','--cached','--quiet'],check=False).returncode==0:raise RuntimeError('nothing to publish')
 run(['git','-C',str(TMP),'commit','-m','experiment: add v6.36 basic vs full loss matched control'])
 data_commit=run(['git','-C',str(TMP),'rev-parse','HEAD']).stdout.strip()
 push=run(['git','-C',str(TMP),'push','origin','HEAD:refs/heads/main'],check=False)
 if push.returncode:
  status={'status':'PUSH_FAILED','target':'origin/main','error':push.stderr[-4000:],'local_commit':data_commit,'remote_base':remote,'time':time.time()}
  json.dump(status,open(R/'publication_status.json','w'),indent=2);return 2
 # Read after push, then make a receipt commit recording the verified data commit.
 run(['git','-C',str(TMP),'fetch','origin','main'])
 seen=run(['git','-C',str(TMP),'rev-parse','origin/main']).stdout.strip();assert seen==data_commit
 for rel in CRITICAL:
  run(['git','-C',str(TMP),'cat-file','-e',f'origin/main:{REL}/{rel}'])
 status={'status':'PUBLISHED','target':'origin/main','data_commit':data_commit,'remote_head_verified':True,'remote_files_verified':CRITICAL,'file_count':len(payload),'verified_at_unix':time.time()}
 json.dump(status,open(TMP/REL/'publication_status.json','w'),indent=2);open(TMP/REL/'publication_status.json','a').write('\n')
 integ=json.load(open(TMP/REL/'final_integrity.json'));integ['publication_status']='PUBLISHED';integ['published_data_commit']=data_commit;json.dump(integ,open(TMP/REL/'final_integrity.json','w'),indent=2);open(TMP/REL/'final_integrity.json','a').write('\n')
 run(['git','-C',str(TMP),'add','--all','--',REL]);run(['git','-C',str(TMP),'commit','-m','experiment: record v6.36 publication verification'])
 receipt_commit=run(['git','-C',str(TMP),'rev-parse','HEAD']).stdout.strip();run(['git','-C',str(TMP),'push','origin','HEAD:refs/heads/main'])
 run(['git','-C',str(TMP),'fetch','origin','main']);final_remote=run(['git','-C',str(TMP),'rev-parse','origin/main']).stdout.strip();assert final_remote==receipt_commit
 for rel in CRITICAL+['publication_status.json','RESULTS_SUMMARY.md']:run(['git','-C',str(TMP),'cat-file','-e',f'origin/main:{REL}/{rel}'])
 # Copy the committed release receipts back into the shared working copy.
 for name in ('publication_status.json','final_integrity.json'):
  shutil.copy2(TMP/REL/name,R/name)
 receipt={'status':'PUBLISHED','target':'origin/main','data_commit':data_commit,'final_commit':receipt_commit,'remote_head_verified':True,'remote_paths_verified':CRITICAL+['publication_status.json','RESULTS_SUMMARY.md'],'file_count':len(payload),'verified_at_unix':time.time()}
 json.dump(receipt,open(R/'publication_status.json','w'),indent=2);open(R/'publication_status.json','a').write('\n')
 # This final local status mirrors the published receipt; report commit in the driver log.
 with open(R/'logs/publication.log','a') as f:f.write(json.dumps(receipt)+'\n')
 print('PUBLISHED',receipt_commit)
 return 0
if __name__=='__main__':sys.exit(main())
