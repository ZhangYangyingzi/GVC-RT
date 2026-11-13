"""Publish the completed local deliverable and verify remote commits and key blobs."""
from util import *
import argparse
BRANCH='codex/bridge-flowguided-deformable-ab-20261113'
REPO=ROOT.parents[1]
PREFIX=str(ROOT.relative_to(REPO))+'/'
def git(*args):return subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
def verify_remote(commit):
 remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==commit,('REMOTE_COMMIT_MISMATCH',remote,commit)
 git('fetch','origin','refs/heads/'+BRANCH)
 assert git('rev-parse','FETCH_HEAD')==commit
 blobs={}
 for name in ['REPORT.md','metrics/all_per_frame.csv','metrics/per_video_annotated.csv']:
  path=PREFIX+name;local=git('rev-parse',commit+':'+path);fetched=git('rev-parse','FETCH_HEAD:'+path);assert local==fetched;blobs[name]=fetched
 return dict(verified_at=stamp(),remote_branch=BRANCH,remote_commit=commit,blobs=blobs)
def stage():
 git('add','--',PREFIX.rstrip('/'));names=git('diff','--cached','--name-only').splitlines();assert names and all(n.startswith(PREFIX) for n in names)
 for name in names:
  p=REPO/name
  if p.exists():assert p.stat().st_size<20*2**20,('LARGE_FILE_STAGED',name)
 assert not any(Path(n).suffix in ['.pt','.npz','.bin','.mkv','.mp4','.so'] for n in names)
 return names
def commit(title,body):
 path=ROOT/'temporary/git_commit_message.txt';path.write_text(title+'\n\n'+body+'\n');git('commit','--file',str(path));return git('rev-parse','HEAD')
def main():
 assert load(ROOT/'final_integrity.json')['status']=='PASS';assert load(ROOT/'audits/native_validity_correction/complete.json')['status']=='PASS'
 assert git('branch','--show-current')==BRANCH
 assert len(list((ROOT/'evaluation/points').glob('*.json')))==150
 stage();results=commit('Run flow-guided deformable A/B pilot and record bounded results','A/B each commit 300 updates with fixed data and loss; evaluate 150 native64-frame points. Include baseline/state/freeze checks, native-domain MC correction audit, independent Original/B300 R-D, beta/blur diagnostics and local lossless video manifests. RWE remains explicitly unverified; no research conclusion. Large weights, caches and media remain local.')
 git('push','origin','HEAD:refs/heads/'+BRANCH);proof=verify_remote(results)
 publication=dict(status='PUBLISHED_VERIFIED',branch=BRANCH,results_commit=results,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+results,remote_verified=True,verification=proof,large_artifacts_uploaded=False,large_artifacts_manifest=str(ROOT/'large_artifacts.json'),scope='code, protocols, reports, CSV/JSON, small figures and artifact manifests only')
 dump(ROOT/'publication_status.json',publication)
 summary=load(ROOT/'summary.json');summary.update(status='COMPLETE',github=publication,remaining=['RWE_NOT_VERIFIED: allowed predefined fallback; no RWE numbers produced']);dump(ROOT/'summary.json',summary)
 status=load(ROOT/'status.json');status.update(status='COMPLETE',stage='COMPLETE',phase='COMPLETE',pid=None,children=[],heartbeat=time.time(),local_time=stamp(),github=publication,committed_updates={'A_zero_base':300,'B_flow_guided':300},evaluated_points=150,evaluated_frames=9600,caches_completed=384,evaluations_completed=150);dump(ROOT/'status.json',status)
 stage();record=commit('Record verified GitHub publication of completed A/B pilot','Record remote results-commit and report/CSV blob verification. No new experiments or training updates.');git('push','origin','HEAD:refs/heads/'+BRANCH);final=verify_remote(record);dump(ROOT/'logs/publication_remote_final.json',dict(status='PASS',results_commit=results,publication_commit=record,final_remote_verification=final,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+record))
 assert not git('status','--porcelain','--',PREFIX.rstrip('/'))
 print(json.dumps(dict(status='COMPLETE',publication_commit=record,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+record,remote_verified=True),indent=2))
if __name__=='__main__':
 with lock('publish.lock'):main()
