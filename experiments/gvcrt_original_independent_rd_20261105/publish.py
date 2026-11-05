"""Publish only small results/code to the already authorized experiment branch."""
from common import *
import shutil
import subprocess


def publish():
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    worktree=PROJECT/'GVC-RT_worktrees/tcg_online_v3_publication_1793765286'
    branch='codex/tcg-stage1-data-expansion-ab-20261103'
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=worktree,text=True).strip()
    assert not git('status','--porcelain'),'Publication worktree has unrelated changes'
    remote=git('ls-remote','origin',f'refs/heads/{branch}').split()[0]
    git('fetch','origin',branch)
    assert subprocess.run(['git','merge-base','--is-ancestor',remote,'HEAD'],cwd=worktree).returncode==0,'Remote advanced; reconcile without force pushing'
    chosen=[]
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if not path.is_file() or any(p in ('cache','tmp','logs','bitstreams','__pycache__','progress') for p in relative.parts):continue
        if path.suffix not in ('.py','.json','.csv','.md','.png') and path.name!='.gitignore':continue
        if path.name=='publication.json':continue
        assert path.stat().st_size<5*1024*1024,path
        target=worktree/ROOT.relative_to(PROJECT)/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);assert sha(path)==sha(target)
        chosen.append(str(target.relative_to(worktree)))
    git('add','--sparse','--',str(ROOT.relative_to(PROJECT)))
    git('diff','--cached','--check')
    if git('diff','--cached','--name-only'):
        git('commit','-m','Add independently measured original GVC-RT curves to TCG dataset comparisons')
    commit=git('rev-parse','HEAD')
    try:
        git('push','origin',f'HEAD:refs/heads/{branch}')
        assert git('ls-remote','origin',f'refs/heads/{branch}').split()[0]==commit
        git('fetch','origin',branch)
        for name in chosen:
            assert git('rev-parse',f'{commit}:{name}')==git('rev-parse',f'FETCH_HEAD:{name}')
        dump(ROOT/'publication.json',dict(status='PUBLISHED',commit=commit,branch=branch,verified_files=len(chosen),remote_verified=True))
    except Exception as exc:
        dump(ROOT/'publication.json',dict(status='LOCAL_COMMIT_PUBLICATION_PENDING',commit=commit,branch=branch,remote_verified=False,error=str(exc)))
        raise


if __name__=='__main__':
    publish()
