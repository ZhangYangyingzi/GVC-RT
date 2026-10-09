"""Commit only this experiment and push the current branch without forcing."""
import traceback
from io18 import *
MESSAGE='experiment: add v6.18 long-horizon error propagation audit'
def git(args):return subprocess.check_output(['git',*args],cwd=REPO,text=True,stderr=subprocess.STDOUT,timeout=300).strip()
def main():
    assert load(ROOT/'final_integrity.json')['status']=='PASS';relative=str(ROOT.relative_to(REPO))
    status=git(['status','--short']);(ROOT/'logs/git_status_before_publication.txt').write_text(status+'\n')
    historical=load(ROOT/'audits/historical_sha256_before.json')
    # Recheck all historical text/code/configuration after final audit, before staging.
    for p,h in historical.items():
        if Path(p).suffix in ('.py','.json','.csv','.jsonl','.md','.txt','.log'):assert sha(p)==h,('historical text changed before publication',p)
    branch=git(['branch','--show-current']);assert branch,'Cannot publish detached HEAD'
    remote=git(['config','--get',f'branch.{branch}.remote']);merge=git(['config','--get',f'branch.{branch}.merge']);assert remote not in ('','.') and merge.startswith('refs/heads/')
    remote_branch=merge.removeprefix('refs/heads/');upstream=git(['config','--get',f'remote.{remote}.url'])
    assert 'ZhangYangyingzi/GVC-RT' in upstream,upstream
    prev=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
    if prev.get('commit') and git(['rev-parse','HEAD'])==prev['commit']:commit=prev['commit']
    else:
        staged_before=git(['diff','--cached','--name-only']).splitlines()
        dump(ROOT/'audits/git_staging_scope.json',dict(status='PASS',current_branch=branch,remote=remote,remote_branch=remote_branch,previously_staged_other_paths=[p for p in staged_before if not Path(p).is_relative_to(Path(relative))],commit_scope=relative,commit_mode='git commit --only -- experiment directory',excluded=['weights','optimizer','bitstreams','features','cache','raw videos','temporary files']))
        git(['add','--',relative])
        staged_experiment=git(['diff','--cached','--name-only','--',relative]).splitlines();assert staged_experiment
        forbidden={'checkpoints','features','bitstreams','cache','visualizations','reconstructed_videos','__pycache__'}
        for p in staged_experiment:
            rel=Path(p).relative_to(Path(relative));assert not forbidden.intersection(rel.parts) and rel.suffix not in ('.pt','.pth','.bin','.npz','.npy','.mp4','.yuv','.tmp','.pid','.lock')
        git(['commit','--only','-m',MESSAGE,'--',relative]);commit=git(['rev-parse','HEAD'])
        changed=git(['diff-tree','--no-commit-id','--name-only','-r',commit]).splitlines();assert changed and all(Path(p).is_relative_to(Path(relative)) for p in changed)
    dump(ROOT/'publication_status.json',dict(status='COMMITTED',commit=commit,current_branch=branch,remote=remote,remote_branch=remote_branch,push_status='pending'))
    try:
        output=git(['push',remote,f'HEAD:{remote_branch}'])
        tip=git(['ls-remote',remote,'refs/heads/'+remote_branch]).split()[0];assert tip==commit
        git(['fetch',remote,remote_branch])
        verified={}
        for name in ('final_integrity.json','evaluation/per_frame_metrics.csv','results/excess_drift_dataset_summary.csv','results/frame_index_slopes.csv','plots/ulong_qp0_lpips_vs_frame.png'):
            obj=git(['rev-parse',f'FETCH_HEAD:{relative}/{name}']);assert obj==git(['hash-object',str(ROOT/name)]);verified[name]=obj
        dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,current_branch=branch,remote=remote,remote_branch=remote_branch,push_status='success',remote_commit=tip,remote_key_files_verified=verified,push_output=output,finished_unix=time.time()))
        print('PUSH SUCCESS',commit,remote,remote_branch,flush=True)
    except Exception:
        dump(ROOT/'publication_status.json',dict(status='FAIL',commit=commit,current_branch=branch,remote=remote,remote_branch=remote_branch,push_status='failed',error=traceback.format_exc(),local_commit_preserved=True));raise
if __name__=='__main__':
    try:main()
    except Exception:
        if not (ROOT/'publication_status.json').exists():dump(ROOT/'publication_status.json',dict(status='FAIL',push_status='not_started',error=traceback.format_exc()))
        raise
