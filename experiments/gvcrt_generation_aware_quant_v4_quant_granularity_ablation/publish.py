"""Publish only reviewed new experiment files; write COMPLETE.json after remote verification."""
from util import *
import subprocess,fcntl
WORKTREE=P.parent.parent
BRANCH='experiment/gvcrt-quant-v4-granularity-ablation'
URL='https://github.com/ZhangYangyingzi/GVC-RT'
def git(*args):return subprocess.check_output(['git',*args],cwd=WORKTREE,text=True).strip()
def main():
 lock=(P/'publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(P/'delivery_status.json')['status']=='READY'
 for name in ('delivery_audit.json','figure_audit.json','preflight.json','cross_mode_audit.json'):assert load(P/name)['status']=='PASS'
 assert load(P/'delivery_audit.json')['signature']==signature()
 assert git('remote','get-url','origin')=='git@github-zyyz:ZhangYangyingzi/GVC-RT.git'
 prefix=str(P.relative_to(WORKTREE))+'/'
 changes=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'],cwd=WORKTREE).decode().split('\0')
 assert all(not s or s[3:].startswith(prefix) for s in changes)
 current=git('branch','--show-current')
 if not current:git('switch','-c',BRANCH)
 else:assert current==BRANCH
 dump(P/'publication_scope.json',dict(status='REVIEWED',destination=URL,branch=BRANCH,scope=prefix,user_authorization='Current request step8 explicitly directs branch creation, scoped git add, commit and GitHub push.',allowed='new experiment code, configuration, audit/index JSON, CSV tables, report Markdown and fixed figures/previews',excluded='weights, complete streams, full map arrays, logs and caches remain server only',force_push=False))
 git('add','--',prefix)
 files=git('ls-files','--',prefix).splitlines()
 assert files and all(n.startswith(prefix) and ('/server_artifacts/' not in n) and (Path(n).suffix in ('.py','.json','.csv','.png','.md') or Path(n).name=='.gitignore') for n in files)
 records=[dict(path=n,bytes=(WORKTREE/n).stat().st_size,sha256=sha(WORKTREE/n)) for n in files if Path(n).name!='delivery_manifest.json']
 dump(P/'delivery_manifest.json',dict(status='PASS',scope=prefix,files=records,server_artifacts='artifact_index.json',complete_receipt='COMPLETE.json is intentionally local; written only after push succeeds'))
 git('add','--',prefix)
 staged=git('diff','--cached','--name-only').splitlines();assert all(n.startswith(prefix) for n in staged)
 if staged:git('commit','-m','Evaluate v4 inference-only quantization granularity ablation')
 commit=git('rev-parse','HEAD')
 git('push','origin',f'HEAD:refs/heads/{BRANCH}')
 remote=git('ls-remote','--heads','origin',f'refs/heads/{BRANCH}').split()[0];assert commit==remote
 for r in records:
  blob=subprocess.check_output(['git','show',f'{commit}:{r["path"]}'],cwd=WORKTREE)
  assert hashlib.sha256(blob).hexdigest()==r['sha256']
 assert not git('status','--porcelain')
 dump(P/'COMPLETE.json',dict(status='COMPLETE',commit_SHA=commit,remote_SHA=remote,branch=BRANCH,remote_verified=True,tests=dict(preflight='PASS',full_audit='PASS',figures='PASS',streams='40/40',frames_per_stream=64,P_map_files=2520,FRAME_CHANNEL_exact=True,A_IDENTITY_native_stream_states_exact=True),training_updates=0,report_path=str(P/'REPORT.md'),report_url=f'{URL}/blob/{commit}/{prefix}REPORT.md',commit_url=f'{URL}/commit/{commit}',artifact_index=str(P/'artifact_index.json'),worktree_clean=True,remaining_blockers=[]))
 print('COMPLETE',commit,flush=True)
if __name__=='__main__':main()
