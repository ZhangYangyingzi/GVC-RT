from io32 import *
import subprocess,shutil,tempfile,fnmatch

def main():
 final=load(ROOT/'final_integrity.json');assert final['status']=='PASS' and final['new_effective_updates']['D_basicDISTS_noP_500']==500
 log=ROOT/'logs/publication.log'
 def git(args,cwd=REPO,check=True):
  p=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  with log.open('a') as f:f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
  if check and p.returncode:raise RuntimeError(p.stdout)
  return p
 def result(args,cwd=REPO):return git(args,cwd).stdout.strip()
 git(['fetch','origin','main']);folder=Path(tempfile.mkdtemp(prefix='gvcrt637-publish-',dir='/Huang_group/zyyz/home_dir/tmp'));git(['worktree','add','--detach',str(folder),'origin/main']);target=folder/'experiments'/ROOT.name;assert not target.exists()
 def ignore(path,names):
  banned={'artifacts','cache','features','bitstreams','checkpoints','__pycache__'};out=[n for n in names if n in banned or any(fnmatch.fnmatch(n,p) for p in ('*.pt','*.npz','*.npy','*.bin','*.tmp','*.lock'))]
  if Path(path)==ROOT/'plans':out+=['shared.json']
  return out
 shutil.copytree(ROOT,target,ignore=ignore)
 oversized=[str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() and p.stat().st_size>50*1024**2];assert not oversized,oversized
 rel=str(target.relative_to(folder));git(['add','--',rel],folder);staged=result(['diff','--cached','--name-only'],folder).splitlines();assert staged and all(p.startswith(rel+'/') for p in staged)
 git(['commit','-m','Add v6.37 loss bridge generator mechanism experiment'],folder)
 def push():
  for _ in range(4):
   p=git(['push','origin','HEAD:refs/heads/main'],folder,check=False)
   if p.returncode==0:break
   if not any(x in p.stdout for x in ('fetch first','non-fast-forward','behind')):raise RuntimeError(p.stdout)
   git(['pull','--rebase','origin','main'],folder)
  else:raise RuntimeError('main advanced during all safe publication attempts')
  commit=result(['rev-parse','HEAD'],folder);remote=result(['ls-remote','--heads','origin','main']).split()[0];assert commit==remote;return commit
 data=push()
 status=dict(status='PUBLISHED',push_success=True,data_commit_SHA=data,verified_remote_main=data,repository='ZhangYangyingzi/GVC-RT',GitHub_directory='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/experiments/'+ROOT.name,verification_time=time.time(),audit_note='This audit records the preceding data commit, not its own not-yet-created SHA.',isolated_worktree=str(folder),original_dirty_worktree_untouched=True)
 dump(ROOT/'publication_status.json',status);final['publication_status']='PUBLISHED_DATA_COMMIT_VERIFIED';dump(ROOT/'final_integrity.json',final)
 for n in ('publication_status.json','final_integrity.json'):shutil.copyfile(ROOT/n,target/n)
 git(['add','--',rel+'/publication_status.json',rel+'/final_integrity.json'],folder);git(['commit','-m','Record v6.37 publication verification'],folder);audit=push()
 dump(ROOT/'publication_status.json',dict(status,audit_commit_SHA=audit,final_remote_main=audit,final_remote_verified_time=time.time(),GitHub_directory_at_commit=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{audit}/experiments/{ROOT.name}'))
 print('PUSH PASS',data,audit,flush=True)
if __name__=='__main__':
 import traceback
 try:main()
 except Exception:
  dump(ROOT/'publication_status.json',dict(status='FAILED',push_success=False,traceback=traceback.format_exc()));raise
