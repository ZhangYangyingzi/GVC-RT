"""Wait for actual completion, index artifacts, and publish an allowlisted bundle."""
import csv,datetime,json,os,shutil,subprocess,sys,time,traceback
from common_expanded import *
PREFIX=ROOT.relative_to(REPO)
BRANCH='codex/tcg-stage1-expanded-20261103'
BASE='fbb55c444fde49052403c0fb975aa2c66750c87e'
def git(*args,cwd=REPO,check=True):
 env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20')
 p=subprocess.run(['git',*args],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
 with (ROOT/'logs/publication_full.log').open('a') as f:f.write(json.dumps(dict(argv=list(args),code=p.returncode,output=p.stdout))+'\n')
 if check and p.returncode:raise RuntimeError(f'git {args[0]} failed: '+p.stdout[-3000:])
 return p.stdout.strip()
def stage_check(wt,allowed):
 paths=git('diff','--cached','--name-only',cwd=wt).splitlines()
 assert paths and set(paths)<=set(allowed)
 for name in paths:
  p=wt/name
  assert not p.is_symlink() and p.stat().st_size<50*1024*1024
  assert p.name=='.gitignore' or p.suffix in ('.py','.json','.jsonl','.csv','.md','.log')
 return paths
def remote_verify(wt,commit):
 r=git('ls-remote','--heads','origin','refs/heads/'+BRANCH,cwd=wt).split()
 assert r and r[0]==commit
def artifacts():
 cps=[]
 for p in sorted((ROOT/'checkpoints').glob('*.pt')):
  cps.append(dict(filename=p.name,size_bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'checkpoints_manifest.json',dict(checkpoints=cps))
 rs=[dict(global_update=int(p.stem),file=p.name,size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'receipts').glob('*.json'))]
 dump(ROOT/'receipts_manifest.json',dict(committed_receipts=len(rs),receipts=rs))
 # Full logs remain in the execution worktree; only a bounded summary is public.
 logs=[]
 for p in sorted((ROOT/'logs').glob('*.log')):
  if p.name not in ('summary.log','publication_full.log'):
   logs.append(dict(file=p.name,size_bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'logs_manifest.json',dict(local_full_logs=logs))
 p=ROOT/'logs/torchrun.log'
 text=p.read_text(errors='replace') if p.exists() else ''
 (ROOT/'logs/summary.log').write_text('\n'.join(text.splitlines()[-120:])+'\n')
 m=load(ROOT/'source_manifest.json')
 m['publication_sources']={str((ROOT/'finalize.py').relative_to(REPO)):sha(ROOT/'finalize.py')}
 if (ROOT/'evaluate.py').exists():m['evaluation_sources']={str((ROOT/'evaluate.py').relative_to(REPO)):sha(ROOT/'evaluate.py')}
 dump(ROOT/'source_manifest.json',m)
 return cps
def publish():
 status=load(ROOT/'run_status.json')
 assert status['status'] in ('PASS','FAILED','BLOCKED')
 assert not git('ls-remote','--heads','origin','refs/heads/'+BRANCH),'Remote target exists; refusing to change it'
 # A second, detached publication worktree leaves execution artifacts untouched.
 wt=REPO.parent/('gvcrt_tcg_stage1_expanded_publish_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
 git('worktree','add','--detach',str(wt),BASE)
 dest=wt/PREFIX;assert not dest.exists();dest.mkdir(parents=True)
 state=dict(status='PREPARED',push_success=False,remote_verified=False,branch=BRANCH,
            training_status=status['training_status'],evaluation_status=status['evaluation_status'],
            evaluation_complete=status['evaluation_status']=='PASS')
 dump(ROOT/'publication_status.json',state)
 allowed=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  rel=p.relative_to(ROOT)
  if any(x in rel.parts for x in ('checkpoints','large_artifacts','__pycache__','receipts')):continue
  if p.name.endswith('.local.json') or p.name in ('paper_text.txt','publication_full.log'):continue
  if rel.parts[0]=='logs' and p.name!='summary.log':continue
  if not(p.name=='.gitignore' or p.suffix in ('.py','.json','.jsonl','.csv','.md','.log')):continue
  assert p.stat().st_size<50*1024*1024
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  if p.suffix=='.csv':target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
  allowed.append(str(target.relative_to(wt)))
 git('add','--',*allowed,cwd=wt)
 staged=stage_check(wt,allowed)
 audit=dict(status='PASS',staged_files=[dict(file=n,size_bytes=(wt/n).stat().st_size,sha256=sha(wt/n)) for n in staged],
            prohibited_files=[],scope='expanded experiment only; TCG source already in pinned baseline')
 dump(ROOT/'audits/staging_audit.json',audit)
 shutil.copy2(ROOT/'audits/staging_audit.json',dest/'audits/staging_audit.json')
 allowed.append(str(PREFIX/'audits/staging_audit.json'))
 git('add','--',str(PREFIX/'audits/staging_audit.json'),cwd=wt)
 stage_check(wt,allowed);git('diff','--cached','--check',cwd=wt)
 git('commit','-m','Add expanded causal TCG Stage 1 training and fixed validation artifacts',cwd=wt)
 data=git('rev-parse','HEAD',cwd=wt)
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=wt);remote_verify(wt,data)
 state.update(status='PUBLISHED',push_success=True,remote_verified=True,data_commit=data,
              data_commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+data,
              branch_url='https://github.com/ZhangYangyingzi/GVC-RT/tree/'+BRANCH)
 dump(ROOT/'publication_status.json',state);shutil.copy2(ROOT/'publication_status.json',dest/'publication_status.json')
 git('add','--',str(PREFIX/'publication_status.json'),cwd=wt);stage_check(wt,[str(PREFIX/'publication_status.json')])
 git('commit','-m','Record verified expanded TCG Stage 1 publication',cwd=wt)
 head=git('rev-parse','HEAD',cwd=wt)
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=wt);remote_verify(wt,head)
 assert set(git('diff','--name-only',BASE,head,cwd=wt).splitlines())==set(allowed)
 for name in allowed:
  blob=subprocess.check_output(['git','show',head+':'+name],cwd=wt)
  assert blob==(wt/name).read_bytes()
 state.update(commit=head,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+head,committed_files_verified=True)
 dump(ROOT/'publication_status.json',state)
 return state
def main():
 while True:
  status=load(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {}
  if status.get('status') in ('PASS','FAILED','BLOCKED'):break
  summary=load(ROOT/'training_summary.json') if (ROOT/'training_summary.json').exists() else {}
  print(json.dumps(dict(event='WAITING_FOR_ACTUAL_COMPLETION',stage=status.get('stage'),committed_global_updates=summary.get('committed_global_updates',0))),flush=True)
  time.sleep(30)
 cps=artifacts()
 try:pub=publish()
 except Exception as exc:
  pub=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
  pub.update(status='FAILED',push_success=False,error=str(exc),traceback=traceback.format_exc())
  dump(ROOT/'publication_status.json',pub);traceback.print_exc()
 status=load(ROOT/'run_status.json')
 result=dict(status='PASS' if status['status']=='PASS' and pub['status']=='PUBLISHED' else 'FAILED',
      experiment_directory=str(ROOT),training_status=status['training_status'],
      committed_global_updates=status['committed_global_updates'],GPU_mapping=[4,5,6,7],
      training_checkpoint_status=dict(saved=[x['filename'] for x in cps]),
      evaluation_status=status['evaluation_status'],report_path=str(ROOT/'REPORT.md'),
      GitHub_branch_URL=pub.get('branch_url'),verified_commit=pub.get('commit'),
      commit_URL=pub.get('commit_url'),failure_stage=status.get('failure_stage') if status['status']!='PASS' else None,
      log_path=str(ROOT/'logs'))
 if pub['status']!='PUBLISHED':result['failure_stage']='PUBLICATION'
 dump(ROOT/'last_status.json',result)
 dump(Path('/Huang_group/zyyz/home_dir/tmp/tcg_stage1_expanded_codex_last.txt'),result)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
