"""Publish only this experiment via an isolated detached worktree; no dirty-tree mutation."""
from common42 import *
import subprocess,shutil,tempfile
PREFIX=ROOT.relative_to(REPO)
def git(*args,cwd=REPO,check=True):
 p=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 append(ROOT/'logs/publication.jsonl',dict(argv=['git',*args],cwd=str(cwd),exit_code=p.returncode,output=p.stdout,time=time.time()))
 if check and p.returncode:raise RuntimeError(p.stdout)
 return p
def copy_public(dest):
 ignored=shutil.ignore_patterns('*.pt','*.npz','*.npy','*.bin','*.lock','*.tmp','__pycache__','artifacts','server_media','server_previews','publication.jsonl')
 shutil.copytree(ROOT,dest,ignore=ignored,dirs_exist_ok=True)
 for p in dest.rglob('*'):
  if p.is_file():assert p.stat().st_size<50*1024*1024,(p,'oversized public artifact')
def verify_public_commit(wt,head):
 import hashlib,io,tarfile
 raw=subprocess.check_output(['git','archive',head,str(PREFIX)],cwd=wt)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
  actual={str(Path(m.name).relative_to(PREFIX)):hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
 expected=load(ROOT/'artifact_index.json')['public']
 missing=[x['relative_path'] for x in expected if x['relative_path'] not in actual]
 mismatched=[x['relative_path'] for x in expected if x['relative_path'] in actual and actual[x['relative_path']]!=x['sha256']]
 assert not missing and not mismatched,(missing,mismatched)
 dump(ROOT/'audits/publication_content_verification.json',dict(status='PASS',commit=head,public_artifacts_verified=len(expected),missing=[],hash_mismatches=[]))
 return len(expected)
def push(wt):
 for attempt in range(3):
  git('fetch','origin','main',cwd=wt);git('rebase','origin/main',cwd=wt)
  p=git('push','origin','HEAD:main',cwd=wt,check=False)
  if p.returncode==0:break
  if not any(x in p.stdout for x in ['non-fast-forward','fetch first','stale info']):raise RuntimeError('PUSH_FAILED; local commit preserved. '+p.stdout)
 else:raise RuntimeError('Remote advanced during all three non-force push attempts; local commit preserved.')
 head=git('rev-parse','HEAD',cwd=wt).stdout.strip()
 remote=git('ls-remote','origin','refs/heads/main',cwd=wt).stdout.split()[0]
 git('fetch','origin','main',cwd=wt)
 assert git('merge-base','--is-ancestor',head,'origin/main',cwd=wt,check=False).returncode==0
 paths=git('show','--pretty=format:','--name-only',head,cwd=wt).stdout.splitlines();assert all(not p or p.startswith(str(PREFIX)+'/') for p in paths)
 return head,paths
if __name__=='__main__':
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 if (ROOT/'publication_status.json').exists() and load(ROOT/'publication_status.json').get('push_success'):print('ALREADY_PUBLISHED');exit()
 wt=None
 try:
  git('fetch','origin','main')
  previous=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
  resume=bool(previous.get('worktree') and previous.get('local_commit') and Path(previous['worktree']).is_dir())
  if resume:
   wt=Path(previous['worktree']);head=git('rev-parse','HEAD',cwd=wt).stdout.strip()
   assert head==previous['local_commit']
   assert git('merge-base','--is-ancestor',head,'origin/main',cwd=wt,check=False).returncode!=0,'Never amend a published commit'
   dest=wt/PREFIX
  else:
   wt=Path(tempfile.mkdtemp(prefix='gvcrt642-publish-',dir='/Huang_group/zyyz/home_dir/tmp'));git('worktree','add','--detach','--no-checkout',str(wt),'origin/main');git('sparse-checkout','init','--cone',cwd=wt);git('sparse-checkout','set',str(PREFIX),cwd=wt);git('checkout','--detach','origin/main',cwd=wt);dest=wt/PREFIX;assert not dest.exists()
  copy_public(dest);git('add','--',str(PREFIX),cwd=wt)
  paths=git('diff','--cached','--name-only',cwd=wt).stdout.splitlines();assert paths and all(p.startswith(str(PREFIX)+'/') for p in paths)
  if resume:git('commit','--amend','--no-edit',cwd=wt)
  else:git('commit','-m','Add v6.42 fixed G_basic content and visual validation',cwd=wt)
  verify_public_commit(wt,git('rev-parse','HEAD',cwd=wt).stdout.strip());data_sha,files=push(wt);verify_public_commit(wt,data_sha)
  dump(ROOT/'publication_status.json',dict(status='PUBLISHED',push_success=True,data_commit=data_sha,commit=data_sha,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{data_sha}',remote_verified=True,worktree=str(wt),published_files=files,time=time.time()))
  dump(ROOT/'run_status.json',dict(status='COMPLETE',optimizer_updates=0,supplementary_points_complete=24,supplementary_points_planned=24,static_groups=32,video_groups=8,blocked=[],data_commit=data_sha,time=time.time()))
  # This small audit commit records the already existing data commit, never its own future SHA.
  for n in ['publication_status.json','run_status.json','audits/publication_content_verification.json']:shutil.copy2(ROOT/n,dest/n)
  git('add','--',str(PREFIX/'publication_status.json'),str(PREFIX/'run_status.json'),str(PREFIX/'audits/publication_content_verification.json'),cwd=wt);git('commit','-m','Record verified v6.42 publication and completion',cwd=wt);audit_sha,_=push(wt);verify_public_commit(wt,audit_sha)
  dump(ROOT/'publication_status.json',dict(status='PUBLISHED',push_success=True,data_commit=data_sha,commit=audit_sha,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{audit_sha}',directory_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{audit_sha}/{PREFIX}',remote_verified=True,worktree=str(wt),time=time.time()))
  print('PUBLISHED',audit_sha,flush=True)
 except Exception as e:
  head=None if wt is None else git('rev-parse','HEAD',cwd=wt,check=False).stdout.strip()
  dump(ROOT/'publication_status.json',dict(status='BLOCKED',push_success=False,error=str(e),local_commit=head,worktree=None if wt is None else str(wt),time=time.time()));raise
