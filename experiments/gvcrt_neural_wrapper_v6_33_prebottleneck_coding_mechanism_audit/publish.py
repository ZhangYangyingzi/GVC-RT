"""Only this experiment, normal commit/current-branch push; no history repair."""
import subprocess,gzip,shutil,time,traceback
from audit_io import *
MESSAGE='Add v6.33 pre-bottleneck coding mechanism audit'
def run(*args):return subprocess.run(['git',*args],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),timeout=600)
def main():
 assert load(ROOT/'integrity.json')['status']=='PASS'
 branch=load(ROOT/'protocol.json')['branch'];assert run('branch','--show-current').stdout.strip()==branch
 remote=run('remote','get-url','--push','origin').stdout.strip();assert 'ZhangYangyingzi/GVC-RT' in remote and 'semcomm' not in remote
 extra=[]
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.csv','.json','.txt','.jsonl') and p.stat().st_size>40_000_000:
   z=p.with_suffix(p.suffix+'.gz')
   with p.open('rb') as f,gzip.open(z,'wb') as o:shutil.copyfileobj(f,o)
   extra.append(dict(original=p.name,sha256=sha(p),gzip=z.name,gzip_sha256=sha(z)))
 if extra:
  with (ROOT/'.gitignore').open('a') as f:
   for r in extra:f.write('/'+r['original']+'\n')
 dump('publication_artifacts.json',dict(large_text_gzip=extra,server_only=['parts/','cache/','sanity/','logs/'],receipts='point_receipts.jsonl.gz'))
 rel=str(ROOT.relative_to(REPO));add=run('add','--',rel);assert add.returncode==0,add.stdout
 # --only commits only authorized paths even if unrelated changes were already staged.
 commit=run('commit','--only','-m',MESSAGE,'--',rel);assert commit.returncode==0,commit.stdout
 head=run('rev-parse','HEAD').stdout.strip();dump('publication_status.json',dict(status='COMMITTED_PENDING_PUSH',commit=head,branch=branch,remote=remote))
 push=run('push','origin',f'HEAD:refs/heads/{branch}');(ROOT/'logs/push.log').write_text(push.stdout)
 if push.returncode:
  dump('publication_status.json',dict(status='PUSH_FAILED_LOCAL_RESULTS_PRESERVED',commit=head,branch=branch,remote=remote,error=push.stdout,no_force=True));return False
 remote_commit=run('ls-remote','origin',f'refs/heads/{branch}');assert remote_commit.returncode==0
 tip=remote_commit.stdout.split()[0];assert tip==head,'Remote moved after push; receipt verification requires review'
 # Fetch the just-pushed remote ref and verify that every committed experiment blob is readable.
 fetch=run('fetch','origin',branch);assert fetch.returncode==0,fetch.stdout
 tree=run('ls-tree','-r',f'origin/{branch}','--',rel);assert tree.returncode==0 and tree.stdout
 for line in tree.stdout.splitlines():
  oid=line.split()[2];check=run('cat-file','-e',oid);assert check.returncode==0
 dump('publication_status.json',dict(status='PUBLISHED',commit=head,remote_commit=tip,branch=branch,remote=remote,remote_blobs_verified=True,link=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',finished=time.time()));return True
if __name__=='__main__':
 try:main()
 except Exception:
  old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {};old.update(status='FAILED',traceback=traceback.format_exc());dump('publication_status.json',old);raise
