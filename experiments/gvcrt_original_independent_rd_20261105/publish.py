"""Publish an explicit small-result allowlist with remote and blob verification."""
from common import *
import shutil
import subprocess
import traceback
import time


def publish():
    branch='codex/tcg-stage1-data-expansion-ab-20261103'
    worktree=PROJECT/'GVC-RT_worktrees/tcg_online_v3_publication_1793765286'
    relative_root=ROOT.relative_to(PROJECT)
    state=dict(status='PREPARING',branch=branch,remote_verified=False)
    dump(ROOT/'publication.json',state)
    os.environ.setdefault('GIT_SSH_COMMAND','ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3')
    def git(*args):
        r=subprocess.run(['git',*args],cwd=worktree,text=True,capture_output=True,timeout=600)
        with (ROOT/'logs/publication.log').open('a') as log:
            log.write(json.dumps(dict(command=['git',*args],returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n')
        if r.returncode:raise RuntimeError(f'git {args[0]} failed ({r.returncode}): {r.stderr[-4000:]}')
        return r.stdout.strip()
    try:
        assert load(ROOT/'final_integrity.json')['status']=='PASS'
        dirty=git('status','--porcelain','-z')
        if dirty:
            for item in dirty.split('\0'):
                if item:assert item[3:].startswith(str(relative_root)+'/'),('Unrelated publication changes',item)
        remote=git('ls-remote','origin',f'refs/heads/{branch}').split()[0]
        git('fetch','origin',f'refs/heads/{branch}')
        git('merge-base','--is-ancestor',remote,'HEAD')
        chosen=[]
        for path in sorted(ROOT.rglob('*')):
            relative=path.relative_to(ROOT)
            if path.is_symlink() or not path.is_file() or any(p in ('cache','tmp','logs','bitstreams','__pycache__','progress') for p in relative.parts):continue
            if path.suffix not in ('.py','.json','.csv','.md','.png') and path.name!='.gitignore':continue
            if path.name in ('publication.json','SHA256_MANIFEST.json'):continue
            assert path.stat().st_size<50*1024*1024,path
            chosen.append(path)
        dump(ROOT/'SHA256_MANIFEST.json',dict(files=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p),size_bytes=p.stat().st_size) for p in chosen],excludes='self and mutable publication metadata; media/weights/streams/cache/logs excluded'))
        chosen.append(ROOT/'SHA256_MANIFEST.json')
        allowed=[]
        for source in chosen:
            target=worktree/relative_root/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            assert sha(source)==sha(target)
            allowed.append(str(target.relative_to(worktree)))
        for start in range(0,len(allowed),100):git('add','--sparse','--',*allowed[start:start+100])
        staged=git('diff','--cached','--name-only','-z').split('\0')
        staged=[name for name in staged if name]
        assert set(staged)<=set(allowed)
        assert all((worktree/name).stat().st_size<50*1024*1024 for name in staged)
        git('diff','--cached','--check')
        dump(ROOT/'staging_audit.json',dict(status='PASS',explicit_allowlist=True,files=[dict(file=name,size_bytes=(worktree/name).stat().st_size) for name in staged],remote_base=remote,no_media_weights_streams_cache=True))
        audit_target=worktree/relative_root/'staging_audit.json'
        shutil.copyfile(ROOT/'staging_audit.json',audit_target)
        git('add','--sparse','--',str(audit_target.relative_to(worktree)))
        git('diff','--cached','--check')
        if git('diff','--cached','--name-only'):git('commit','-m','Add independently measured original GVC-RT curves to TCG dataset comparisons')
        data_commit=git('rev-parse','HEAD')
        git('push','origin',f'HEAD:refs/heads/{branch}')
        assert git('ls-remote','origin',f'refs/heads/{branch}').split()[0]==data_commit
        git('fetch','origin',f'refs/heads/{branch}')
        for name in allowed:
            blob=subprocess.check_output(['git','show',f'{data_commit}:{name}'],cwd=worktree)
            assert hashlib.sha256(blob).hexdigest()==sha(worktree/name)
            assert git('rev-parse',f'{data_commit}:{name}')==git('rev-parse',f'FETCH_HEAD:{name}')
        state.update(status='PUBLISHED',data_commit=data_commit,commit=data_commit,verified_files=len(allowed),remote_verified=True,core_blobs_verified=True)
        dump(ROOT/'publication.json',state)
        dump(ROOT/'run_status.json',dict(status='PASS',stage='complete',completed_points=135,expected_points=135,evaluation_status='PASS',plotting_status='PASS',publication_status='PUBLISHED',verified_data_commit=data_commit,updated_unix=time.time()))
        for name in ('publication.json','run_status.json'):
            target=worktree/relative_root/name
            shutil.copyfile(ROOT/name,target)
            git('add','--sparse','--',str(target.relative_to(worktree)))
        git('diff','--cached','--check')
        if git('diff','--cached','--name-only'):git('commit','-m','Record verified original GVC-RT evaluation and publication completion')
        head=git('rev-parse','HEAD')
        git('push','origin',f'HEAD:refs/heads/{branch}')
        assert git('ls-remote','origin',f'refs/heads/{branch}').split()[0]==head
        for name in ('REPORT.md','metrics_original.csv','original_vs_parent0.csv','final_integrity.json',*[str(p.relative_to(ROOT)) for p in (ROOT/'plots').rglob('*.png')]):
            blob=subprocess.check_output(['git','show',f'{head}:{relative_root}/{name}'],cwd=worktree)
            assert hashlib.sha256(blob).hexdigest()==sha(ROOT/name)
        state.update(commit=head,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',remote_verified=True,core_blobs_verified=True)
        dump(ROOT/'publication.json',state)
        status=load(ROOT/'run_status.json');status.update(verified_commit=head);dump(ROOT/'run_status.json',status)
        print(json.dumps(state),flush=True)
    except Exception:
        state.update(status='FAILED',remote_verified=False,error=traceback.format_exc())
        dump(ROOT/'publication.json',state)
        raise


if __name__=='__main__':
    publish()
