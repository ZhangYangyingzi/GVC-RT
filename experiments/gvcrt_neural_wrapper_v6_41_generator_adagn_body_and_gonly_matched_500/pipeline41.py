"""No-LLM terminal pipeline: wait, read-only eval, aggregate, publish normally."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
WORKTREE=Path('/Huang_group/zyyz/home_dir/tmp/gvcrt-v641-publish')
REL=Path('experiments')/HERE.name

def get(p):
 try:return json.loads((HERE/p).read_text())
 except FileNotFoundError:return {}
def put(p,obj):
 q=HERE/p;q.parent.mkdir(parents=True,exist_ok=True)
 t=q.with_name(q.name+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(q)
def command(args,cwd=REPO,env=None):
 print('COMMAND',args,flush=True)
 e=os.environ.copy();e.update(OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
 if env:e.update(env)
 subprocess.run(args,cwd=cwd,env=e,check=True)
def output(args,cwd=REPO):return subprocess.check_output(args,cwd=cwd,text=True).strip()

def publish():
 assert WORKTREE.is_dir()
 command(['git','fetch','origin','main'],cwd=WORKTREE)
 command(['git','switch','-C','v641-publication','origin/main'],cwd=WORKTREE)
 put('publication_status.json',dict(status='READY_TO_COMMIT',computed_integrity=get('final_integrity.json').get('status'),
  note='The remote commit verification is stored server-side after publishing to avoid a self-reference.'))
 command([PYTHON,str(HERE/'package41.py'),str(WORKTREE)])
 command(['git','add','-f','--',str(REL)],cwd=WORKTREE)
 staged=output(['git','diff','--cached','--name-only'],cwd=WORKTREE).splitlines()
 assert staged and all(p.startswith(str(REL)+'/') for p in staged)
 command(['git','commit','-m','Add v6.41 generator AdaGN and matched G-only control'],cwd=WORKTREE)
 for attempt in range(3):
  p=subprocess.run(['git','push','origin','HEAD:main'],cwd=WORKTREE)
  if p.returncode==0:break
  if attempt==2:raise RuntimeError('NORMAL_GIT_PUSH_FAILED_AFTER_THREE_ATTEMPTS')
  command(['git','fetch','origin','main'],cwd=WORKTREE)
  command(['git','rebase','origin/main'],cwd=WORKTREE)
 sha=output(['git','rev-parse','HEAD'],cwd=WORKTREE)
 remote=output(['git','ls-remote','origin','refs/heads/main'],cwd=WORKTREE).split()[0]
 assert sha==remote,(sha,remote)
 for name in ('REPORT.md','final_integrity.json','results/all_rd_points.csv','results/pooled_fid.csv','effective_config.json'):
  assert output(['git','cat-file','-e',f'{sha}:{REL/name}'],cwd=WORKTREE)==''
 put('publication_status.json',dict(status='PUBLISHED',commit=sha,remote_main=remote,
  github_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{sha}/{REL}',
  verified_files=['REPORT.md','final_integrity.json','results/all_rd_points.csv','results/pooled_fid.csv','effective_config.json'],
  verification_location='server-only receipt written after the normal push'))

def main():
 with (HERE/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   train=get('branches/G_fullLoss_Gonly_500/training_status.json')
   put('pipeline_status.json',dict(stage='WAIT_TRAIN',train=train.get('status','NOT_STARTED'),
    committed_updates=train.get('committed_updates',0),A=get('A_integrity.json').get('status','PENDING'),
    updated_unix=time.time()))
   if train.get('status')=='PASS' and train.get('committed_updates')==500:break
   if train.get('status') in ('FAILED','BLOCKED'):raise RuntimeError(f"TRAINING_TERMINAL_{train['status']}")
   if train.get('updated_unix') and time.time()-train['updated_unix']>1800:
    raise RuntimeError('TRAINING_STALE_OVER_30_MINUTES_REQUIRES_PROCESS_AUDIT')
   time.sleep(55)
  assert get('A_integrity.json').get('status')=='PASS'
  put('pipeline_status.json',dict(stage='EVALUATE_GONLY',updated_unix=time.time()))
  command([PYTHON,str(HERE/'run_gonly41.py')])
  assert get('run_gonly_status.json').get('status')=='PASS'
  put('pipeline_status.json',dict(stage='AGGREGATE',updated_unix=time.time()))
  command([PYTHON,str(HERE/'aggregate41.py')])
  command([PYTHON,str(HERE/'finalize41.py')])
  # The frozen codec environment intentionally has no matplotlib; figures use
  # the host Python with matplotlib and never load models or alter metrics.
  command(['python',str(HERE/'figures41.py')])
  assert get('final_integrity.json').get('status')=='PASS'
  put('pipeline_status.json',dict(stage='PUBLISH',updated_unix=time.time()))
  publish()
  put('pipeline_status.json',dict(stage='COMPLETE',status='PASS',commit=get('publication_status.json')['commit'],
   updated_unix=time.time()))
  print('PIPELINE41_PASS',get('publication_status.json')['commit'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  err=traceback.format_exc();put('pipeline_status.json',dict(stage='FAILED',status='FAILED',error=err,updated_unix=time.time()))
  with (HERE/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(stage='pipeline',traceback=err,time=time.time()))+'\n')
  raise
