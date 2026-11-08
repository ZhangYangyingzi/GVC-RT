"""Only this experiment's allowlist, sparse-safe staging, normal push, remote/blob checks."""
from common_s2 import *
BRANCH='codex/tcg-stage2-convlora-coverage-ab-20261107'
URL='https://github.com/ZhangYangyingzi/GVC-RT'

def git(args,check=True):
    p=subprocess.run(['git',*args],cwd=WORKTREE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with (ROOT/'logs/publication.log').open('a') as f:f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
    if check and p.returncode:raise RuntimeError('GIT_FAILED: '+' '.join(args)+'\n'+p.stdout[-4000:])
    return p.stdout.strip(),p.returncode

def verify_remote(head,core):
    assert git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])[0].split()[0]==head
    records=[]
    for name in core:
        rel=str((ROOT/name).relative_to(WORKTREE));local=git(['hash-object',str(ROOT/name)])[0]
        committed=git(['rev-parse',head+':'+rel])[0];remote=git(['rev-parse','refs/remotes/origin/'+BRANCH+':'+rel])[0]
        assert local==committed==remote;records.append(dict(file=name,blob=local))
    return records

def main():
    env();verify_sources();assert load(ROOT/'final_integrity.json')['status']=='PASS'
    with lock('publication.lock'):
        try:
            assert git(['branch','--show-current'])[0]==BRANCH
            assert 'ZhangYangyingzi/GVC-RT' in git(['remote','get-url','origin'])[0]
            remote=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])[0]
            if remote:
                h=remote.split()[0];git(['fetch','origin',BRANCH]);assert git(['merge-base','--is-ancestor',h,'HEAD'],False)[1]==0,'REMOTE_ADVANCED_OR_DIVERGED_NO_FORCE'
            files=[]
            for p in sorted(ROOT.rglob('*')):
                if not p.is_file() or p.is_symlink():continue
                rel=p.relative_to(ROOT)
                if any(x in rel.parts for x in ['logs','cache','tmp','large_artifacts','__pycache__']):continue
                if p.suffix not in ['.py','.json','.csv','.md','.png','.gz']:continue
                assert p.stat().st_size<50*2**20
                assert not any(x in p.suffixes for x in ['.pt','.pth','.ckpt','.mp4','.bin','.pdf','.npy','.npz'])
                files.append(p)
            names=[str(p.relative_to(WORKTREE)) for p in files]
            staged=git(['diff','--cached','--name-only','-z'])[0]
            assert not staged or all(n in names for n in staged.split('\0') if n),'UNRELATED_STAGED_FILES'
            dump(ROOT/'publication_allowlist.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files if p.name!='publication_allowlist.json'],self_hash_excluded=True,git_add_sparse=True))
            names=list(dict.fromkeys(names+[str((ROOT/'publication_allowlist.json').relative_to(WORKTREE))]))
            for i in range(0,len(names),80):git(['add','--sparse','--',*names[i:i+80]])
            git(['diff','--cached','--check']);staged=git(['diff','--cached','--name-only','-z'])[0]
            assert all(n in names and (WORKTREE/n).is_relative_to(ROOT) for n in staged.split('\0') if n)
            if staged:git(['commit','-m','Complete fixed A2000 half-gain QP0 and QP5 development evaluation'])
            head=git(['rev-parse','HEAD'])[0];git(['push','origin','HEAD:refs/heads/'+BRANCH])
            core=['protocol.json','sources.json','convlora.py','evaluate_core.py','REPORT.md','metrics_s05_qp05.csv','metrics_s05_all_qps.csv','endpoint_metrics.csv',
                  'metrics_combined.csv','paired_deltas.csv','summary_by_qp.csv','summary.json','stream_identity_checks.csv','final_integrity.json']
            blobs=verify_remote(head,core)
            dump(ROOT/'publication_status.json',dict(status='PUBLISHED',branch=BRANCH,artifact_commit=head,commit_url=URL+'/commit/'+head,remote_verified=True,core_blobs_verified=True,core_blobs=blobs,time=time.time()))
            state=load(ROOT/'status.json');state.update(status='COMPLETE',complete=True,current_stage='complete',runner_running=False,active_worker_count=0,
                local_evaluation_complete=True,formal_completed_points=64,QP0_completed=32,QP5_completed=32,QP9_reused=32,half_total=96,combined_total=288,
                integrity_status='PASS',publication_status='PUBLISHED',time=time.time());dump(ROOT/'status.json',state)
            dump(ROOT/'active_processes.json',dict(stage='COMPLETE',children=[],time=time.time()))
            meta=[str((ROOT/n).relative_to(WORKTREE)) for n in ['publication_status.json','status.json','active_processes.json']]
            git(['add','--sparse','--',*meta]);git(['diff','--cached','--check'])
            assert all(n in meta for n in git(['diff','--cached','--name-only','-z'])[0].split('\0') if n)
            if git(['diff','--cached','--quiet'],False)[1]:git(['commit','-m','Record completed QP05 gain diagnostic and verified publication'])
            final=git(['rev-parse','HEAD'])[0];git(['push','origin','HEAD:refs/heads/'+BRANCH]);final_blobs=verify_remote(final,core+['status.json','publication_status.json'])
            dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',branch=BRANCH,artifact_commit=head,remote_HEAD=final,commit_url=URL+'/commit/'+final,
                remote_verified=True,core_blobs_verified=True,core_blobs=final_blobs,completion_status_on_remote='COMPLETE',time=time.time()))
            print('QP05_REMOTE_HEAD_AND_CORE_BLOBS_VERIFIED',final)
        except Exception as exc:
            dump(ROOT/'publication_status.json',dict(status='FAILED',branch=BRANCH,error=str(exc),remote_verified=False,local_commit=git(['rev-parse','HEAD'],False)[0],
                local_evaluation_complete=True,resume_command=f'{PYTHON} -B -u {ROOT}/publish.py',time=time.time()))
            raise

if __name__=='__main__':main()
