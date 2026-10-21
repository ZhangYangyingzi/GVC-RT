"""Local staging and verification. Network fetch/push are explicit external Git commands."""
from common43 import *
import argparse,subprocess,shutil,tempfile,tarfile,io
PREFIX=ROOT.relative_to(REPO)
EXCLUDED={'__pycache__','.git','publication_status.json','artifact_index.json','publication_verification.json'}
def git(wt,*args):return subprocess.check_output(['git',*args],cwd=wt,text=True).strip()
def public_files():
 return [p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts) and p.suffix not in ('.lock','.tmp','.pyc')]
def index():
 files=public_files();assert all(p.stat().st_size<45*1024*1024 for p in files)
 dump(ROOT/'artifact_index.json',dict(public=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],server_only_models=load(ROOT/'model_manifest.json')['checkpoint'],inputs='See data_manifest.json and source_manifest.json; source RGB remains on server.'))
def verify(wt,head):
 raw=subprocess.check_output(['git','archive',head,str(PREFIX)],cwd=wt)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
  actual={str(Path(m.name).relative_to(PREFIX)):hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf.getmembers() if m.isfile()}
 expected=load(ROOT/'artifact_index.json')['public'];bad=[r['path'] for r in expected if actual.get(r['path'])!=r['sha256']];assert not bad,bad
 dump(ROOT/'publication_verification.json',dict(status='PASS',commit=head,verified_files=len(expected),mismatched=[]))
def main(phase,worktree):
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 if phase=='stage':
  index();wt=Path(tempfile.mkdtemp(prefix='gvcrt643-publish-',dir='/Huang_group/zyyz/home_dir/tmp'))
  git(REPO,'worktree','add','--detach','--no-checkout',str(wt),'origin/main');git(wt,'sparse-checkout','init','--cone');git(wt,'sparse-checkout','set',str(PREFIX));git(wt,'checkout','--detach','origin/main')
  dest=wt/PREFIX;assert not dest.exists();dest.mkdir(parents=True)
  for p in public_files()+[ROOT/'artifact_index.json']:
   rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  git(wt,'add','-f','--',str(PREFIX));paths=git(wt,'diff','--cached','--name-only').splitlines();assert paths and all(x.startswith(str(PREFIX)+'/') for x in paths)
  git(wt,'commit','-m','Add v6.43 fixed G_basic frame QP budget utility experiment')
  head=git(wt,'rev-parse','HEAD');verify(wt,head)
  dump(ROOT/'publication_status.json',dict(status='STAGED',worktree=str(wt),commit=head,push_success=False));print(wt,head)
 else:
  wt=Path(worktree);head=git(wt,'rev-parse','HEAD');git(wt,'merge-base','--is-ancestor',head,'origin/main');verify(wt,head)
  status=load(ROOT/'publication_status.json');status.update(status='PUBLISHED',push_success=True,remote_verified=True,commit=head,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}')
  if phase=='receipt':
   status['data_commit']=head;dump(ROOT/'publication_status.json',status)
   for name in ('publication_status.json','publication_verification.json'):shutil.copy2(ROOT/name,wt/PREFIX/name)
   git(wt,'add','-f','--',str(PREFIX/'publication_status.json'),str(PREFIX/'publication_verification.json'));git(wt,'commit','-m','Record verified v6.43 publication')
  else:dump(ROOT/'publication_status.json',status)
  print(git(wt,'rev-parse','HEAD'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['stage','receipt','verify']);p.add_argument('--worktree');a=p.parse_args();main(a.phase,a.worktree)
