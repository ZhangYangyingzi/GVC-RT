"""Normal origin publication of completed G11 only; remote ref and blob verification."""
from util import *
from runner import alive
import traceback
BRANCH='codex/g11-detail-temporal-pilot-20261113';REPO=ROOT.parents[1];PREFIX=str(ROOT.relative_to(REPO))+'/'
def git(*args):
 env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3');p=subprocess.run(['git',*args],cwd=REPO,env=env,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(dict(command=['git',*args],returncode=p.returncode,stderr=p.stderr[-4000:]))
 return p.stdout.strip()
def verify_remote(commit):
 actual=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert actual==commit;git('fetch','origin','refs/heads/'+BRANCH);assert git('rev-parse','FETCH_HEAD')==commit;blobs={}
 for name in ['REPORT.md','PROTOCOL.json','metrics/per_source.csv','metrics/all_per_frame.csv']:
  a=git('rev-parse',commit+':'+PREFIX+name);b=git('rev-parse','FETCH_HEAD:'+PREFIX+name);assert a==b;blobs[name]=b
 return dict(status='PASS',remote_commit=commit,verified_at=stamp(),key_blobs=blobs)
def stage():
 git('add','--',PREFIX.rstrip('/'));paths=git('diff','--cached','--name-only').splitlines();assert paths and all(p.startswith(PREFIX) for p in paths)
 for name in paths:
  p=REPO/name
  if p.exists():assert p.stat().st_size<20*2**20,('LARGE_FILE_STAGED',name)
  assert p.suffix not in ['.pt','.pth','.npz','.bin','.mkv','.mp4','.so']
 return paths

def main():
 assert load(ROOT/'final_integrity.json')['status']=='PASS';assert git('branch','--show-current')==BRANCH
 proc=ROOT/'receipts/finalize_process.json'
 while proc.exists() and alive(load(proc)['pid']):time.sleep(2)
 source=load(ROOT/'source_implementation_final.json')['code'];assert all(sha(ROOT/n)==h for n,h in source.items())
 local=load(ROOT/'summary.json');stage();message=ROOT/'temporary/git_message.txt';message.write_text('Run bounded G11 detail-temporal pilot and record factual results\n\nPreserve complete B300 plus D12-S900, train only G11 and native first receiver block, evaluate fixed six-source validation at committed milestones, and record calibration, freeze/state/codec checks and short-segment lossless media manifests. No final-test selection, extra group, or scientific conclusion. Large artifacts remain local.\n');git('commit','--file',str(message));results=git('rev-parse','HEAD')
 for attempt in range(3):
  try:git('push','origin','HEAD:refs/heads/'+BRANCH);break
  except Exception:
   if attempt==2:raise
   time.sleep(5)
 proof=verify_remote(results);record=dict(status='PUBLISHED_VERIFIED',branch=BRANCH,results_commit=results,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+results,remote_verified=True,verification=proof,large_artifacts_uploaded=False,large_artifacts_manifest=str(ROOT/'large_artifacts.json'))
 dump(ROOT/'publication_status.json',record);local.update(status='COMPLETE',complete=True,github=record,remaining=[]);dump(ROOT/'summary.json',local);stage();message.write_text('Record verified G11 publication\n\nRemote results commit and protocol/report/CSV blobs verified. No additional training or evaluation.\n');git('commit','--file',str(message));published=git('rev-parse','HEAD');git('push','origin','HEAD:refs/heads/'+BRANCH);last=verify_remote(published);assert not git('status','--porcelain','--',PREFIX.rstrip('/'));dump(ROOT/'logs/publication_remote_final.json',dict(status='PASS',results_commit=results,publication_commit=published,verification=last,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+published));print(json.dumps(dict(status='COMPLETE',remote_verified=True,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+published)))
if __name__=='__main__':
 with lock('publish.lock'):
  try:main()
  except BaseException:
   error=dict(status='LOCAL_COMPLETE_PUBLICATION_FAILED',remote_verified=False,branch=BRANCH,error=traceback.format_exc(),local_experiment_complete=True,large_artifacts_uploaded=False,time=stamp());dump(ROOT/'publication_status.json',error);summary=load(ROOT/'summary.json');summary.update(status='LOCAL_COMPLETE_PUBLICATION_FAILED',complete=False,github=error,remaining=['GitHub publication only; experiment artifacts are locally complete']);dump(ROOT/'summary.json',summary);raise
