"""Publish corrected numerical bookkeeping without replacing intervening remote edits."""
import shutil,traceback
from publish_github import run,files_for_upload
from io21 import *
def main():
    f=load(ROOT/'final_integrity.json');assert f['status']=='PASS'
    assert all(x['expected']==x['completed'] for x in f['counts'].values())
    prior=load(ROOT/'publication_status.json');assert prior['status']=='PASS'
    dump(ROOT/'audits/initial_publication.json',prior)
    runtime=load(ROOT/'pipeline_status.json');runtime.update(status='PASS',phase='complete',publication_pending=False);runtime.pop('publication',None);dump(ROOT/'pipeline_status.json',runtime)
    files=files_for_upload();relative=ROOT.relative_to(REPO);paths=[str(relative/p.relative_to(ROOT)) for p in files]
    for attempt in range(3):
        run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main'])
        oldtree=run(['git','ls-tree','-r',prior['commit'],'--',str(relative)])
        tree=run(['git','ls-tree','-r',base,'--',str(relative)])
        assert tree==oldtree,'BLOCKED: intervening remote edits in experiment directory'
        work=REPO.parent/'GVC-RT-publication-worktrees'/f'v622_correction_{time.time_ns()}'
        run(['git','worktree','add','--detach',str(work),base])
        for p,name in zip(files,paths):
            target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        for off in range(0,len(paths),100):run(['git','add','-f','--',*paths[off:off+100]],work)
        staged=run(['git','diff','--cached','--name-only'],work).splitlines();assert staged and all(x.startswith(str(relative)+'/') for x in staged)
        run(['git','commit','-m','Fix v6.22 dataset counts and metric summary bookkeeping'],work);commit=run(['git','rev-parse','HEAD'],work)
        try:output=run(['git','push','origin','HEAD:main'],work)
        except subprocess.CalledProcessError as e:
            current=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
            if current!=base and attempt<2:continue
            raise RuntimeError('normal push failed: '+e.output)
        run(['git','fetch','origin','main'],work);run(['git','merge-base','--is-ancestor',commit,'origin/main'],work)
        tree=run(['git','ls-tree','-r','origin/main','--',str(relative)],work)
        objects={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
        for p,name in zip(files,paths):
            b=(work/name).read_bytes();h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert objects[name]==h
        current=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
        result=dict(status='PASS',commit=commit,remote_main=current,base=base,files=len(paths),all_remote_blobs_verified=True,worktree=str(work),link='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/'+str(relative),push_output=output,finished_unix=time.time())
        dump(ROOT/'publication_status.json',result);runtime.update(publication=result);dump(ROOT/'pipeline_status.json',runtime)
        print('PUBLICATION PASS',commit,flush=True);return
if __name__=='__main__':
    try:main()
    except Exception:
        old=load(ROOT/'publication_status.json');old.update(status='FAIL',error=traceback.format_exc());dump(ROOT/'publication_status.json',old);raise
