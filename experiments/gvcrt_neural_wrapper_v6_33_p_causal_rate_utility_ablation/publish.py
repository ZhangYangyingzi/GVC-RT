"""Publish only the authorized experiment from an isolated latest-main worktree."""
from audit import *
import subprocess
import shutil
import tempfile
import time

def main():
    assert load(ROOT/'status.json')['status']=='PASS'
    log=ROOT/'logs/publication.log'
    def git(args,cwd=REPO):
        p=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with log.open('a') as f:f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
        if p.returncode:raise RuntimeError(p.stdout)
        return p.stdout.strip()
    git(['fetch','origin','main'])
    folder=Path(tempfile.mkdtemp(prefix='gvcrt633-publish-'))
    git(['worktree','add','--detach',str(folder),'origin/main'])
    target=folder/'experiments'/ROOT.name
    assert not target.exists(),'An existing remote experiment must be reviewed before replacing it'
    shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('*.pt','*.bin','*.npz','*.tmp','__pycache__','*.lock','bitstreams','features','cache'))
    relative=str(target.relative_to(folder))
    git(['status','--short'],folder)
    git(['add','--',relative],folder)
    git(['commit','-m','Add v6.33 P causal and rate-utility ablation'],folder)
    for attempt in range(3):
        p=subprocess.run(['git','push','origin','HEAD:refs/heads/main'],cwd=folder,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with log.open('a') as f:f.write(p.stdout+'\n')
        if p.returncode==0:break
        if 'fetch first' not in p.stdout and 'non-fast-forward' not in p.stdout and 'behind' not in p.stdout:raise RuntimeError(p.stdout)
        git(['pull','--rebase','origin','main'],folder)
    else:raise RuntimeError('Remote kept advancing during three safe publication attempts')
    commit=git(['rev-parse','HEAD'],folder)
    remote=git(['ls-remote','--heads','origin','main']).split()[0]
    assert commit==remote,('Remote verification',commit,remote)
    dump('publication_status.json',dict(status='PASS',push_success=True,commit=commit,remote_main=remote,worktree=str(folder),repository='ZhangYangyingzi/GVC-RT',time=time.time(),original_dirty_worktree_untouched=True))
    print('PUSH PASS',commit,flush=True)

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump('publication_status.json',dict(status='FAILED',push_success=False,traceback=traceback.format_exc()));raise
