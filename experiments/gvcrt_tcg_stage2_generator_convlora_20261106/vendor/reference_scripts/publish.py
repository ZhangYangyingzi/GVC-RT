"""Explicit task-only allowlist, normal new branch commits and verified fast-forward push."""
from common_s2 import *
import argparse
BRANCH='codex/tcg-stage2-history-gate-joint-20261106'
URL='https://github.com/ZhangYangyingzi/GVC-RT'

def git(args,check=True):
    result=subprocess.run(['git',*args],cwd=WORKTREE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with (ROOT/'logs/publication.log').open('a') as f:f.write('$ git '+' '.join(args)+'\n'+result.stdout+'\n')
    if check and result.returncode:raise RuntimeError('GIT_COMMAND_FAILED: '+' '.join(args)+'\n'+result.stdout[-4000:])
    return result.stdout.strip(),result.returncode

def candidates():
    excluded={'cache','tmp','large_artifacts','checkpoints','logs','.git','__pycache__'}
    allowed={'.py','.json','.gz','.csv','.md','.png','.txt'}
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if str(rel)=='publication_allowlist.json':continue
        if any(x in excluded for x in rel.parts) or p.is_symlink() or not p.is_file():continue
        if p.suffix not in allowed and p.name!='.gitignore':continue
        assert p.stat().st_size<50*2**20,('PUBLICATION_FILE_TOO_LARGE',str(rel))
        assert not any(x in rel.suffixes for x in ['.pt','.pth','.ckpt','.mp4','.bin','.pdf','.yuv','.npy','.npz'])
        assert not any(x in p.name.lower() for x in ['secret','credential','private_key'])
        # All archived receiver/TCG code is source, never symlinked pretrained weights.
        files.append(p)
    return files

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['code','results'],default='results');args=ap.parse_args();env()
    with lock('publication.lock'):
        assert git(['branch','--show-current'])[0]==BRANCH
        assert 'ZhangYangyingzi/GVC-RT' in git(['remote','get-url','origin'])[0]
        try:
            remote=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])[0]
            if remote:
                remote_head=remote.split()[0]
                git(['fetch','origin',BRANCH])
                _,code=git(['merge-base','--is-ancestor',remote_head,'HEAD'],check=False)
                assert code==0,'REMOTE_ADVANCED_OR_DIVERGED; no overwrite or force push allowed'
            files=candidates();allow=[str(p.relative_to(WORKTREE)) for p in files]
            dump(ROOT/'publication_allowlist.json',dict(phase=args.phase,manifest_self_file='publication_allowlist.json',manifest_self_hash_excluded=True,files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
            files.append(ROOT/'publication_allowlist.json');allow=[str(p.relative_to(WORKTREE)) for p in files]
            # Refuse pre-existing unrelated staged changes instead of unstaging or modifying them.
            staged,_=git(['diff','--cached','--name-only','-z']);assert not staged or all(x in allow for x in staged.split('\0') if x)
            for start in range(0,len(allow),100):git(['add','--',*allow[start:start+100]])
            git(['diff','--cached','--check'])
            staged,_=git(['diff','--cached','--name-only','-z']);names=[x for x in staged.split('\0') if x]
            assert all(x in allow for x in names)
            for name in names:
                p=WORKTREE/name;assert p.is_relative_to(ROOT) and p.stat().st_size<50*2**20
            if names:git(['commit','-m','Add Stage2 internal-history-gate joint '+('implementation and fixed protocol' if args.phase=='code' else 'measured results and integrity audits')])
            head=git(['rev-parse','HEAD'])[0];git(['push','origin',f'HEAD:refs/heads/{BRANCH}'])
            verified=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])[0].split()[0];assert verified==head
            core=['protocol.json','model_context.py','runtime_tree/src/models/spatial_tcg_router.py','source_manifest.json']
            if args.phase=='results':core+=['REPORT.md','final_integrity.json','metrics_joint.csv','metrics_comparison.csv','selection.json']
            checks=[]
            for name in core:
                path=ROOT/name;repo_path=str(path.relative_to(WORKTREE));blob=git(['rev-parse',f'{head}:{repo_path}'])[0]
                local=git(['hash-object',str(path)])[0];remote_blob=git(['rev-parse',f'refs/remotes/origin/{BRANCH}:{repo_path}'])[0]
                assert blob==local==remote_blob;checks.append(dict(file=name,blob=blob))
            publication=dict(status='PUBLISHED',phase=args.phase,branch=BRANCH,commit=head,commit_url=URL+'/commit/'+head,remote_verified=True,core_blobs_verified=True,core_blobs=checks,time=time.time())
            dump(ROOT/'publication_status.json',publication)
            # Record verified publication in a follow-up commit; no self-referential commit hash.
            git(['add','--',str((ROOT/'publication_status.json').relative_to(WORKTREE))]);git(['diff','--cached','--check'])
            _,code=git(['diff','--cached','--quiet'],check=False)
            if code:git(['commit','-m','Record verified Stage2 '+args.phase+' publication'])
            audit_head=git(['rev-parse','HEAD'])[0];git(['push','origin',f'HEAD:refs/heads/{BRANCH}'])
            assert git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])[0].split()[0]==audit_head
            dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',published_artifact_commit=head,branch_HEAD=audit_head,branch=BRANCH,branch_url=URL+'/tree/'+BRANCH,HEAD_commit_url=URL+'/commit/'+audit_head,remote_verified=True,core_blobs_verified=True,time=time.time()))
        except Exception as exc:
            dump(ROOT/'publication_status.json',dict(status='FAILED',phase=args.phase,branch=BRANCH,error=str(exc),remote_verified=False,time=time.time(),resume_command=f'{PYTHON} -B -u {ROOT}/publish.py --phase {args.phase}'))
            raise
if __name__=='__main__':main()
