"""Explicit task-only light artifacts; ordinary branch push and blob verification."""
from common_s2 import *

def main():
    env();branch='codex/tcg-flicker-pilot-abcd-20261109'
    def git(*args):return subprocess.run(['git',*args],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout.strip()
    record=dict(status='PREPARED',branch=branch,remote_verified=False,core_blobs_verified=False,time=time.time())
    try:
        assert git('branch','--show-current')==branch and not git('diff','--cached','--name-only'),'UNRELATED_STAGED_FILES'
        remote=git('ls-remote','--heads','origin','refs/heads/'+branch)
        if remote:git('merge-base','--is-ancestor',remote.split()[0],'HEAD')
        files=[]
        for p in ROOT.rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            if any(part in ['cache','tmp','logs','locks','GT_roi_selection','__pycache__','checkpoints'] for part in p.relative_to(ROOT).parts):continue
            if p.suffix not in ['.py','.md','.csv','.json','.png','.diff']:continue
            assert p.stat().st_size<50*2**20
            files.append(str(p.relative_to(WORKTREE)))
        dump(ROOT/'publication_status.json',record)
        receipt=str((ROOT/'publication_status.json').relative_to(WORKTREE))
        if receipt not in files:files.append(receipt)
        for i in range(0,len(files),100):git('add','--sparse','--force','--',*files[i:i+100])
        assert set(git('diff','--cached','--name-only').splitlines())<=set(files)
        git('diff','--cached','--check');git('commit','-m','Fixed B300 vs original: person-dog and Cactus QP0/5 measurements and crop diagnostics')
        commit=git('rev-parse','HEAD');git('push','origin','HEAD:refs/heads/'+branch);assert git('ls-remote','--heads','origin','refs/heads/'+branch).split()[0]==commit
        checks=[]
        for file in files:
            raw=subprocess.check_output(['git','show',commit+':'+file],cwd=WORKTREE);assert hashlib.sha256(raw).hexdigest()==sha(WORKTREE/file)
            if file!=receipt:checks.append(dict(file=file,blob=git('rev-parse',commit+':'+file),SHA256=sha(WORKTREE/file)))
        record.update(status='PUBLISHED',core_commit=commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}',remote_verified=True,core_blobs_verified=True,allowlist=files,core_files=checks,verified_at=time.time());dump(ROOT/'publication_status.json',record)
        git('add','--sparse','--force','--',receipt);git('diff','--cached','--check');git('commit','-m','Verify person-dog Cactus comparison publication receipt')
        head=git('rev-parse','HEAD');git('push','origin','HEAD:refs/heads/'+branch);assert git('ls-remote','--heads','origin','refs/heads/'+branch).split()[0]==head
        for r in checks:assert git('rev-parse',head+':'+r['file'])==r['blob']
        dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',branch=branch,commit=head,core_commit=commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',remote_verified=True,core_blobs_verified=True,time=time.time()))
        return True
    except Exception as error:
        record.update(status='FAILED',exception=repr(error),stdout=getattr(error,'stdout',None),stderr=getattr(error,'stderr',None));dump(ROOT/'publication_status.json',record);return False

if __name__=='__main__':main()
