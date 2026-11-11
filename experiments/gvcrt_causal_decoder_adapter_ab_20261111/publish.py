"""Explicit experiment-only Git publication. No weights/caches/media or force push."""
from common_s2 import *
import argparse


def git(*args,check=True):
    return subprocess.run(['git',*args],cwd=WORKTREE,env=os.environ,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=check)


def main(phase):
    env();assert git('branch','--show-current').stdout.strip()==BRANCH
    relative=str(ROOT.relative_to(WORKTREE));manifest=load(ROOT/'source_manifest.json')
    manifest['code']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    manifest['final_code_snapshot_time']=time.time();dump(ROOT/'source_manifest.json',manifest)
    publication=dict(status='PREPARED',phase=phase,branch=BRANCH,remote='origin',
        includes='experiment code, protocol, manifests, facts, CSV, bounded plots and training monitor PNG',
        excludes='datasets, model weights, caches, large FFV1 videos, unrelated workspace files',force_push=False,merge_main=False,time=time.time())
    dump(ROOT/'publication_status.json',publication)
    git('add','--',relative)
    staged=git('diff','--cached','--name-only').stdout.splitlines()
    assert staged and all(f.startswith(relative+'/') for f in staged)
    blocked=[]
    for f in staged:
        p=WORKTREE/f
        if p.suffix.lower() in ['.pt','.pth','.bin','.mkv','.mp4','.ckpt'] or '/cache/' in f or '/media/' in f or p.stat().st_size>10*2**20:blocked.append(f)
    assert not blocked,('PROHIBITED_PUBLICATION_ARTIFACTS',blocked)
    git('commit','-m',f'Causal decoder adapter A/B: {phase} execution facts')
    payload=git('rev-parse','HEAD').stdout.strip()
    pushed=git('push','-u','origin',BRANCH,check=False)
    (ROOT/'logs/publication_push.log').write_text(pushed.stdout)
    if pushed.returncode:
        dump(ROOT/'publication_status.json',dict(publication,status='PUSH_FAILED',local_commit=payload,push_output=pushed.stdout));raise RuntimeError('GIT_PUSH_FAILED')
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).stdout.split()[0];assert remote==payload
    publication.update(status='REMOTE_VERIFIED',payload_commit=payload,verified_remote_commit=remote,
        payload_commit_URL='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+payload,
        branch_URL='https://github.com/ZhangYangyingzi/GVC-RT/tree/'+BRANCH,time=time.time())
    dump(ROOT/'publication_status.json',publication)
    git('add','--',str((ROOT/'publication_status.json').relative_to(WORKTREE)))
    git('commit','-m','Record verified remote publication of causal adapter results')
    receipt_commit=git('rev-parse','HEAD').stdout.strip();git('push','origin',BRANCH)
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).stdout.split()[0];assert remote==receipt_commit
    dump(ROOT/'publication_remote_receipt.json',dict(status='REMOTE_VERIFIED',phase=phase,branch=BRANCH,
        payload_commit=payload,commit=receipt_commit,remote_commit=remote,
        commit_URL='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+remote,
        publication_status_record_in_this_commit_verifies_payload_commit=payload,time=time.time()))
    print('REMOTE_VERIFIED',remote,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final_results');a=p.parse_args();main(a.phase)
