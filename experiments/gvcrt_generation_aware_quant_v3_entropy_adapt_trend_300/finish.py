"""Wait for fixed computation, audit, plot, publish only this experiment, verify remote."""
from util import *
import subprocess,shutil
WORKTREE=P.parent.parent
BRANCH='experiment/gvcrt-quant-v3-entropy-adapt-trend-300'
URL='https://github.com/ZhangYangyingzi/GVC-RT'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
def command(args,**kwargs):
 return subprocess.check_output(list(map(str,args)),cwd=WORKTREE,text=True,**kwargs).strip()
def git(*args):return command(['git',*args])
def committed_inventory():
 prefix=str(P.relative_to(WORKTREE))
 paths=git('ls-files','--',prefix).splitlines()
 return [dict(path=n,sha256=sha(WORKTREE/n),bytes=(WORKTREE/n).stat().st_size) for n in paths if Path(n).name not in ('delivery_manifest.json','publication_status.json','run_status.json','completion.json','final_integrity.json')]
def publish():
 assert git('branch','--show-current')==BRANCH
 assert git('remote','get-url','origin')=='git@github-zyyz:ZhangYangyingzi/GVC-RT.git'
 prefix=str(P.relative_to(WORKTREE))+'/'
 # The isolated tree is clean except this newly authorized experiment.
 status=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'],cwd=WORKTREE).decode().split('\0')
 assert all(not entry or entry[3:].startswith(prefix) for entry in status),status[:10]
 dump(P/'publication_status.json',dict(status='PUBLISHING',branch=BRANCH,remote='origin',force_push=False))
 git('add','--',prefix)
 staged=git('diff','--cached','--name-only').splitlines();assert staged and all(n.startswith(prefix) for n in staged)
 dump(P/'delivery_manifest.json',dict(status='PASS',source_commit='3f9fae83a8be672c4c4999e09b7c4b08bf4a8fb4',scope=prefix,public_files=committed_inventory(),large_server_artifacts='artifact_index.json',mutable_publication_receipts_excluded_from_file_hash_inventory=True))
 git('add','--',prefix)
 if git('diff','--cached','--name-only'):git('commit','-m','Add v3 matched quantization and entropy adaptation trend experiment')
 data_commit=git('rev-parse','HEAD')
 git('push','origin',f'HEAD:refs/heads/{BRANCH}')
 remote=git('ls-remote','--heads','origin',f'refs/heads/{BRANCH}').split()[0];assert remote==data_commit
 # Verify exact Git blob content against public inventory before recording publication.
 manifest=load(P/'delivery_manifest.json')
 for r in manifest['public_files']:
  blob=subprocess.check_output(['git','show',f'{data_commit}:{r["path"]}'],cwd=WORKTREE)
  assert hashlib.sha256(blob).hexdigest()==r['sha256']
 report=f'{URL}/blob/{data_commit}/{prefix}REPORT.md'
 receipt=dict(status='PUBLISHED',branch=BRANCH,data_commit=data_commit,data_commit_url=f'{URL}/commit/{data_commit}',report_url=report,remote_verified=True,remote_commit_at_verification=remote,public_blob_hashes_verified=len(manifest['public_files']),large_server_artifacts='artifact_index.json',force_push=False,remaining_blockers=[])
 dump(P/'publication_status.json',receipt)
 integ=load(P/'final_integrity.json');integ.update(publication_verified=True,publication_receipt='publication_status.json',report_url=report);dump(P/'final_integrity.json',integ)
 dump(P/'completion.json',dict(status='COMPLETE',new_updates={'N0':0,'N1':300,'Q0':0,'Q1':300},reused_updates={'N0':300,'N1':0,'Q0':300,'Q1':0},fixed_evaluations='64/64',integrity='PASS',publication=receipt,remaining_blockers=[]))
 dump(P/'run_status.json',dict(status='PUBLISHED',new_updates={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300},data_commit=data_commit,report_url=report,remaining_blockers=[]))
 git('add','--',prefix);assert all(n.startswith(prefix) for n in git('diff','--cached','--name-only').splitlines())
 git('commit','-m','Record verified v3 publication and completion receipts')
 receipt_commit=git('rev-parse','HEAD');git('push','origin',f'HEAD:refs/heads/{BRANCH}')
 remote=git('ls-remote','--heads','origin',f'refs/heads/{BRANCH}').split()[0];assert remote==receipt_commit
 assert not git('status','--porcelain')
 # External local receipt avoids a self-referential commit hash in committed content.
 dump(P/'server_artifacts/final_remote_verification.json',dict(status='PASS',branch=BRANCH,remote_head=remote,commit_url=f'{URL}/commit/{remote}',data_commit=data_commit,report_url=report,worktree_clean=True))
 print('PUBLISHED',remote,report,flush=True)
def main(prepare_only=False):
 lock=(P/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  st=load(P/'run_status.json');audit=P/'audits/reused_N0_extended_decode.json'
  if st['status']=='FAILED':raise RuntimeError(st)
  behavior=P/'audits/behavior_symbols.json'
  if st['status'] in ('COMPUTE_COMPLETE','COMPLETE','PUBLISHED') and audit.exists() and load(audit)['status']=='PASS' and behavior.exists() and load(behavior)['status']=='PASS':break
  time.sleep(10)
 env=dict(os.environ,PYTHONPATH=str(P)+os.pathsep+str(ORIGINAL),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run([PYTHON,'-B',str(P/'finalize.py')],cwd=WORKTREE,env=env,check=True)
 subprocess.run([PLOT_PYTHON,'-B',str(P/'plot_results.py')],cwd=WORKTREE,env=env,check=True)
 # Copies made only after all main tasks are finished, so committed logs are stable.
 logs=P/'logs';logs.mkdir(exist_ok=True)
 for f in (P/'server_artifacts/logs').glob('*.log'):shutil.copyfile(f,logs/f.name)
 for branch in ('N1','Q1'):shutil.copyfile(P/'server_artifacts'/branch/'optimizer_events.jsonl',logs/f'optimizer_events_{branch}.jsonl')
 dump(P/'run_status.json',dict(status='COMPLETE',new_updates={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300},signature=signature()))
 if prepare_only:
  print('PREPARED; audits, tables, plots and logs complete; no Git publication',flush=True)
 else:publish()
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 mode=parser.add_mutually_exclusive_group()
 mode.add_argument('--prepare-only',action='store_true',help='Prepare reviewable deliverables without committing or publishing')
 mode.add_argument('--publish-only',action='store_true',help='Publish already prepared and audited deliverables')
 args=parser.parse_args()
 try:
  if args.publish_only:
   lock=(P/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   assert load(P/'run_status.json')['status']=='COMPLETE'
   assert load(P/'final_integrity.json')['status']=='PASS'
   assert load(P/'figure_index.json')['status']=='PASS'
   publish()
  else:main(prepare_only=args.prepare_only)
 except BlockingIOError:
  print('FINISHER_ALREADY_ACTIVE; preserve active publication',flush=True)
