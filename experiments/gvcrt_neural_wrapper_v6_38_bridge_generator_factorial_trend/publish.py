"""Publish only this experiment via an isolated detached worktree; no dirty-tree mutation."""
from io32 import *
import subprocess,shutil,tempfile
PREFIX=ROOT.relative_to(REPO)
def git(*args,cwd=REPO,check=True):
 p=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 append(ROOT/'logs/publication.jsonl',dict(argv=['git',*args],cwd=str(cwd),exit_code=p.returncode,output=p.stdout,time=time.time()))
 if check and p.returncode:raise RuntimeError(p.stdout)
 return p
def copy_public(dest):
 ignored=shutil.ignore_patterns('*.pt','*.npz','*.npy','*.bin','*.lock','*.tmp','__pycache__','artifacts')
 shutil.copytree(ROOT,dest,ignore=ignored,dirs_exist_ok=True)
 for p in dest.rglob('*'):
  if p.is_file():assert p.stat().st_size<50*1024*1024,(p,'oversized public artifact')
def push(wt):
 # A remote fast-forward race is repaired before our sole push attempt.
 git('fetch','origin','main',cwd=wt);git('rebase','origin/main',cwd=wt)
 p=git('push','origin','HEAD:main',cwd=wt,check=False)
 if p.returncode:raise RuntimeError('PUSH_REJECTED: preserving local commit, no repeated push attempt. '+p.stdout)
 head=git('rev-parse','HEAD',cwd=wt).stdout.strip();remote=git('ls-remote','origin','refs/heads/main',cwd=wt).stdout.split()[0];assert remote==head
 paths=git('show','--pretty=format:','--name-only',head,cwd=wt).stdout.splitlines();assert all(not p or p.startswith(str(PREFIX)+'/') for p in paths)
 return head,paths
if __name__=='__main__':
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 if (ROOT/'publication_status.json').exists() and load(ROOT/'publication_status.json').get('push_success'):print('ALREADY_PUBLISHED');exit()
 wt=None
 try:
  git('fetch','origin','main');wt=Path(tempfile.mkdtemp(prefix='gvcrt638-publish-',dir='/Huang_group/zyyz/home_dir/tmp'));git('worktree','add','--detach',str(wt),'origin/main');dest=wt/PREFIX;assert not dest.exists()
  copy_public(dest);git('add','--',str(PREFIX),cwd=wt)
  paths=git('diff','--cached','--name-only',cwd=wt).stdout.splitlines();assert paths and all(p.startswith(str(PREFIX)+'/') for p in paths)
  git('commit','-m','Add v6.38 Bridge Generator factorial trend experiment',cwd=wt);data_sha,files=push(wt)
  dump(ROOT/'publication_status.json',dict(status='PUBLISHED',push_success=True,data_commit=data_sha,commit=data_sha,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{data_sha}',remote_verified=True,worktree=str(wt),published_files=files,time=time.time()))
  dump(ROOT/'run_status.json',dict(status='COMPLETE',updates={'O':0,'B':100,'G':100,'BG':100},RD_complete=24,RD_planned=24,blocked=[],data_commit=data_sha,time=time.time()))
  # This small audit commit records the already existing data commit, never its own future SHA.
  for n in ['publication_status.json','run_status.json']:shutil.copy2(ROOT/n,dest/n)
  git('add','--',str(PREFIX/'publication_status.json'),str(PREFIX/'run_status.json'),cwd=wt);git('commit','-m','Record verified v6.38 publication and completion',cwd=wt);audit_sha,_=push(wt)
  dump(ROOT/'publication_status.json',dict(status='PUBLISHED',push_success=True,data_commit=data_sha,commit=audit_sha,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{audit_sha}',directory_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{audit_sha}/{PREFIX}',remote_verified=True,worktree=str(wt),time=time.time()))
  print('PUBLISHED',audit_sha,flush=True)
 except Exception as e:
  head=None if wt is None else git('rev-parse','HEAD',cwd=wt,check=False).stdout.strip()
  dump(ROOT/'publication_status.json',dict(status='BLOCKED',push_success=False,error=str(e),local_commit=head,worktree=None if wt is None else str(wt),time=time.time()));raise
