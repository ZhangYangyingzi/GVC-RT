"""Publish only this experiment from an isolated latest-main worktree."""
from base import *
import argparse
import shutil
import tempfile
import traceback


def main(checkpoint=False):
    state=load(ROOT/'run_status.json')
    if not checkpoint:assert state['status'] in ('PASS','PARTIAL')
    log=ROOT/'logs/publication.log'
    def git(args,cwd=REPO,check=True):
        p=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with log.open('a') as f:f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
        if check and p.returncode:raise RuntimeError(p.stdout)
        return p
    git(['fetch','origin','main'])
    folder=Path(tempfile.mkdtemp(prefix='gvcrt-oracle-publish-'))
    git(['worktree','add','--no-checkout','--detach',str(folder),'origin/main'])
    # --no-checkout does not initialize an index. Seed it from HEAD before
    # applying sparse patterns, so absent unrelated files remain tracked.
    git(['read-tree','HEAD'],folder)
    git(['sparse-checkout','init','--cone'],folder)
    git(['sparse-checkout','set','experiments/'+ROOT.name],folder)
    target=folder/'experiments'/ROOT.name
    prior=ROOT/'publication_status.json'
    if target.exists():
        assert prior.exists(),'Remote experiment exists without a local publication receipt; review required'
        previous=load(prior)
        assert previous.get('push_success') and previous.get('commit')
        git(['merge-base','--is-ancestor',previous['commit'],'origin/main'])
        assert load(target/'experiment_manifest.json')['name']==ROOT.name
    ignored={'artifacts','__pycache__','.git','bitstreams','features','cache'}
    published=[]
    for source in sorted(ROOT.rglob('*')):
        relative=source.relative_to(ROOT)
        if any(part in ignored for part in relative.parts) or source.suffix in ('.pt','.pth','.bin','.npz','.tmp','.lock','.pyc') or not source.is_file():continue
        assert not source.is_symlink(),source
        assert source.stat().st_size<90*1024**2,('Unexpected large Git artifact',source)
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);published.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=sha(source)))
    (target/'manifests/publication_inventory.json').write_text(json.dumps(dict(checkpoint_only=checkpoint,experiment_state=state,files=published),indent=2)+'\n')
    relative=str(target.relative_to(folder))
    git(['add','--',relative],folder)
    changed=git(['diff','--cached','--name-only','--no-renames'],folder).stdout.splitlines()
    assert changed and all(p.startswith(relative+'/') for p in changed)
    git(['commit','-m','Add GVC-RT recoverability oracle feasibility experiment'+(' (checkpoint; running)' if checkpoint else '')],folder)
    def push():
        for _ in range(3):
            p=git(['push','origin','HEAD:refs/heads/main'],folder,check=False)
            if p.returncode==0:return
            if not any(s in p.stdout for s in ('fetch first','non-fast-forward','behind')):raise RuntimeError(p.stdout)
            git(['pull','--rebase','origin','main'],folder)
        raise RuntimeError('Remote advanced during three safe publication attempts')
    push()
    data_commit=git(['rev-parse','HEAD'],folder).stdout.strip()
    remote=git(['ls-remote','--heads','origin','main']).stdout.split()[0]
    assert git(['merge-base','--is-ancestor',data_commit,remote],folder,check=False).returncode==0
    receipt=dict(status='CHECKPOINT_PUSHED_RUNNING' if checkpoint else 'PASS',push_success=True,commit=data_commit,data_commit=data_commit,remote_main=remote,worktree=str(folder),repository='ZhangYangyingzi/GVC-RT',time=time.time(),experiment_state=state,original_dirty_worktree_untouched=True,excluded='datasets, weights, caches, raw large binary artifacts; all server artifacts are indexed at completion')
    dump('publication_status.json',receipt)
    # Commit a receipt of the already verified data push. Its own SHA is
    # recorded locally after the second push, avoiding a self-hash cycle.
    shutil.copy2(ROOT/'publication_status.json',target/'publication_status.json')
    shutil.copy2(log,target/'logs/publication.log')
    git(['add','--',relative+'/publication_status.json',relative+'/logs/publication.log'],folder)
    git(['commit','-m','Record recoverability oracle '+('checkpoint' if checkpoint else 'results')+' publication'],folder)
    push()
    commit=git(['rev-parse','HEAD'],folder).stdout.strip()
    remote=git(['ls-remote','--heads','origin','main']).stdout.split()[0]
    assert git(['merge-base','--is-ancestor',commit,remote],folder,check=False).returncode==0
    receipt.update(commit=commit,remote_main=remote,receipt_commit=commit)
    dump('publication_status.json',receipt)
    print('PUSH PASS',commit,'checkpoint' if checkpoint else 'final results',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',action='store_true');args=ap.parse_args()
    try:main(args.checkpoint)
    except Exception:
        dump('logs/publication_failure.json',dict(status='FAILED',push_success=False,traceback=traceback.format_exc(),time=time.time()));raise
