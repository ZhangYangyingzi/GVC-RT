"""Task-only explicit allowlist, normal commit/push and remote blob checks."""
from common_s2 import *
import argparse,traceback
BRANCH='codex/tcg-flicker-pilot-abcd-20261109'

def git(args,**kwargs):return subprocess.run(['git',*args],cwd=WORKTREE,check=True,text=True,capture_output=True,**kwargs).stdout.strip()

def allowlist():
    result=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(ROOT);parts=rel.parts
        if any(x in parts for x in ['cache','tmp','locks','__pycache__','checkpoints','smoke']):continue
        if 'receipts' in parts and parts[0] in GROUPS:continue
        if parts[0]=='logs' and p.name!='bounded_summary.json':continue
        if p.name in ['publication_remote_receipt.json'] or p.suffix in ['.pt','.pth','.ckpt','.bin','.log','.lock','.mkv','.mp4','.mov','.yuv','.so']:continue
        if p.suffix not in ['.py','.json','.csv','.md','.png','.gz'] and p.name!='.gitignore':continue
        assert p.stat().st_size<50*2**20,('PUBLICATION_FILE_TOO_LARGE',str(rel))
        result.append(str(p.relative_to(WORKTREE)))
    return sorted(result)

def package_receipts():
    for g in GROUPS:
        files=sorted((ROOT/g/'training/receipts').glob('*.json'))
        for start in range(0,len(files),100):
            block=files[start:start+100]
            dump(ROOT/f'receipts/{g}_{int(block[0].stem):04d}_{int(block[-1].stem):04d}.json.gz',[load(p) for p in block])
    summaries={p.name:p.read_text(errors='replace').splitlines()[-35:] for p in (ROOT/'logs').glob('*.log')}
    dump(ROOT/'logs/bounded_summary.json',summaries)

def core_paths():
    names=['protocol.json','source_manifest.json','REPORT.md','final_integrity.json','development_metrics.csv','screening_history.json','training_history.csv']
    return [str((ROOT/n).relative_to(WORKTREE)) for n in names if (ROOT/n).exists()]

def publish(phase):
    env();package_receipts();log=ROOT/'logs'/f'publication_{phase}_{int(time.time())}.log'
    try:
        assert git(['branch','--show-current'])==BRANCH
        remote_before=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])
        if remote_before:
            old=remote_before.split()[0]
            # Refuse divergent remote updates; no force push or automatic merge.
            git(['merge-base','--is-ancestor',old,'HEAD'])
        record=dict(status='PREPARED',phase=phase,branch=BRANCH,remote_verified=False,core_blobs_verified=False,files=None,time=time.time())
        dump(ROOT/'publication_status.json',record)
        files=allowlist();record['files']=files;dump(ROOT/'publication_status.json',record)
        with log.open('w') as f:f.write(json.dumps(dict(allowlist=files,remote_before=remote_before),indent=2)+'\n')
        # No broad staging command. Each argument is a validated task file.
        for i in range(0,len(files),150):git(['add','--sparse','--force','--',*files[i:i+150]])
        staged=git(['diff','--cached','--name-only']).splitlines();prefix=str(ROOT.relative_to(WORKTREE))+'/'
        assert staged and all(n.startswith(prefix) and n in files for n in staged),'UNRELATED_STAGED_FILE'
        for n in staged:
            p=WORKTREE/n;assert p.stat().st_size<50*2**20 and p.suffix not in ['.pt','.pth','.bin','.mp4','.mkv','.ckpt']
        git(['diff','--cached','--check'])
        git(['commit','-m',f'TCG flicker pilot ABCD: {phase} protocol and factual records'])
        core_commit=git(['rev-parse','HEAD']);pushed=git(['push','origin','HEAD:refs/heads/'+BRANCH])
        observed=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]);assert observed.split()[0]==core_commit
        checks=[]
        for p in core_paths():
            blob=subprocess.check_output(['git','show',core_commit+':'+p],cwd=WORKTREE)
            measured=hashlib.sha256(blob).hexdigest();assert measured==sha(WORKTREE/p)
            checks.append(dict(file=p,SHA256=measured,git_blob=git(['rev-parse',core_commit+':'+p])))
        record.update(status='PUBLISHED',core_commit=core_commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{core_commit}',remote_verified=True,core_blobs_verified=True,core_files=checks,push_output=pushed,verified_at=time.time())
        dump(ROOT/'publication_status.json',record)
        status_path=str((ROOT/'publication_status.json').relative_to(WORKTREE));git(['add','--sparse','--force','--',status_path]);git(['diff','--cached','--check']);git(['commit','-m',f'TCG flicker pilot ABCD: verified {phase} publication receipt'])
        receipt_commit=git(['rev-parse','HEAD']);git(['push','origin','HEAD:refs/heads/'+BRANCH]);remote=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]);assert remote.split()[0]==receipt_commit
        for r in checks:assert git(['rev-parse',receipt_commit+':'+r['file']])==r['git_blob']
        dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',phase=phase,branch=BRANCH,commit=receipt_commit,core_commit=core_commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{receipt_commit}',remote_verified=True,core_blobs_verified=True,time=time.time()))
        return True
    except Exception as error:
        traceback_text=traceback.format_exc()
        with log.open('a') as f:f.write(traceback_text+'\n');f.write(getattr(error,'stdout','') or '');f.write(getattr(error,'stderr','') or '')
        dump(ROOT/'publication_status.json',dict(status='FAILED',phase=phase,branch=BRANCH,remote_verified=False,core_blobs_verified=False,exception=repr(error),log=str(log),
            local_HEAD=git(['rev-parse','HEAD']),resume_command=f'{PYTHON} -B -u {ROOT}/publish.py --phase {phase}',time=time.time()))
        return False

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='results');a=p.parse_args();publish(a.phase)
