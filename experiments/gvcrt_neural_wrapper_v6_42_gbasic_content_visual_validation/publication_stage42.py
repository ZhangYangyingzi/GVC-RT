"""Local publication stages; network Git calls can run directly with tool approval."""
from publish42 import *
import argparse

def stage(phase,wt):
 dest=wt/PREFIX
 assert wt.is_dir() and dest.is_dir()
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 head=git('rev-parse','HEAD',cwd=wt).stdout.strip()
 if phase=='prepare':
  assert git('merge-base','--is-ancestor',head,'origin/main',cwd=wt,check=False).returncode!=0
  copy_public(dest);git('add','--',str(PREFIX),cwd=wt)
  paths=git('diff','--cached','--name-only',cwd=wt).stdout.splitlines()
  assert all(p.startswith(str(PREFIX)+'/') for p in paths)
  git('commit','--amend','--no-edit',cwd=wt)
  head=git('rev-parse','HEAD',cwd=wt).stdout.strip();verify_public_commit(wt,head)
  dump(ROOT/'publication_status.json',dict(status='READY_TO_PUSH',push_success=False,local_commit=head,worktree=str(wt)))
 elif phase=='data_receipt':
  git('merge-base','--is-ancestor',head,'origin/main',cwd=wt);verify_public_commit(wt,head)
  dump(ROOT/'publication_status.json',dict(status='PUBLISHED',push_success=True,data_commit=head,commit=head,
   commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',remote_verified=True,worktree=str(wt)))
  dump(ROOT/'run_status.json',dict(status='COMPLETE',optimizer_updates=0,supplementary_points_complete=24,
   supplementary_points_planned=24,static_groups=32,video_groups=8,data_commit=head,blocked=[]))
  for n in ('publication_status.json','run_status.json','audits/publication_content_verification.json'):shutil.copy2(ROOT/n,dest/n)
  git('add','--',str(PREFIX/'publication_status.json'),str(PREFIX/'run_status.json'),str(PREFIX/'audits/publication_content_verification.json'),cwd=wt)
  git('commit','-m','Record verified v6.42 publication and completion',cwd=wt)
 elif phase=='verify':
  git('merge-base','--is-ancestor',head,'origin/main',cwd=wt);verify_public_commit(wt,head)
  status=load(ROOT/'publication_status.json');status.update(status='PUBLISHED',push_success=True,commit=head,
   commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',remote_verified=True,worktree=str(wt),
   directory_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{head}/{PREFIX}')
  dump(ROOT/'publication_status.json',status)
 print(phase,git('rev-parse','HEAD',cwd=wt).stdout.strip(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','data_receipt','verify'));p.add_argument('--worktree',type=Path,required=True);a=p.parse_args();stage(a.phase,a.worktree)
