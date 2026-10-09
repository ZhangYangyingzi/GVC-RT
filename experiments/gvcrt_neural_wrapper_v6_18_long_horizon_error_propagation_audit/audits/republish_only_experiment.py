"""Preserve original commit and publish its experiment-only delta on current remote main."""
import sys,subprocess,json,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];REL=ROOT.relative_to(REPO)
def git(args,cwd=REPO):
    p=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError('git '+repr(args)+'\n'+p.stdout+p.stderr)
    return p.stdout.strip()
def save(v):
    p=ROOT/'publication_status.json';p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    prior=json.loads((ROOT/'publication_status.json').read_text());original=prior['commit'];assert git(['rev-parse','HEAD'])==original
    paths=git(['diff-tree','--no-commit-id','--name-only','-r',original]).splitlines();assert paths and all(Path(p).is_relative_to(REL) for p in paths)
    git(['fetch','origin','main']);base=git(['rev-parse','origin/main']);work=ROOT/'.publication_worktree'
    if work.exists():raise RuntimeError('Publication worktree already exists; preserve it for review')
    git(['worktree','add','--detach',str(work),base]);git(['cherry-pick','--no-commit',original],work)
    destination=work/REL
    # Record ordinary push rejection and an accurate reproducible fallback command.
    for suffix in ('.gitignore','logs/push_attempt_details.txt','audits/republish_only_experiment.py'):
        target=destination/suffix;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/suffix,target)
    runbook=destination/'RUNBOOK.md'
    runbook.write_text(runbook.read_text()+'''\n## Remote main advanced during publication\n\nThe original experiment-only commit is retained in the source worktree. A detached publication worktree inside `.publication_worktree/` applies only that commit on the latest origin/main, then stages only this experiment directory and pushes HEAD:main without force. The publication checkout itself is ignored and never uploaded.\n\n```bash\n/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/audits/republish_only_experiment.py\n```\n\nThe helper refuses to overwrite an existing publication checkout. Publication status, both original/published commits and remote visibility checks are saved in `publication_status.json`.\n''')
    metadata=dict(status='READY',original_local_commit=original,remote_base=base,publication_mode='detached worktree; experiment-only delta; ordinary HEAD:main push',original_commit_preserved=True,ordinary_push_rejection=prior.get('push_details','non-fast-forward'),source_worktree_not_switched=True)
    (destination/'audits/publication_recovery.json').write_text(json.dumps(metadata,indent=2)+'\n')
    git(['add','--',str(REL)],work);staged=git(['diff','--cached','--name-only'],work).splitlines();assert all(Path(p).is_relative_to(REL) for p in staged)
    assert all('.publication_worktree' not in Path(p).parts for p in staged)
    git(['commit','-m','experiment: add v6.18 long-horizon error propagation audit'],work);published=git(['rev-parse','HEAD'],work)
    metadata.update(commit=published,current_branch='main',remote='origin',remote_branch='main',worktree=str(work),status='COMMITTED',push_status='pending');save(metadata)
    git(['push','origin','HEAD:main'],work);tip=git(['ls-remote','origin','refs/heads/main'],work).split()[0];assert tip==published
    git(['fetch','origin','main'],work);verified={}
    for suffix in ('final_integrity.json','evaluation/per_frame_metrics.csv','results/excess_drift_dataset_summary.csv','results/frame_index_slopes.csv','plots/ulong_qp0_lpips_vs_frame.png'):
        obj=git(['rev-parse',f'origin/main:{REL}/{suffix}'],work);assert obj==git(['hash-object',str(ROOT/suffix)]);verified[suffix]=obj
    assert git(['rev-parse','HEAD'])==original
    metadata.update(status='PASS',push_status='success',remote_commit=tip,remote_key_files_verified=verified,finished_unix=time.time());save(metadata)
    print('PUSH SUCCESS',published,'original preserved',original,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        prior=json.loads((ROOT/'publication_status.json').read_text());prior.update(status='FAIL',push_status='failed',error=traceback.format_exc(),local_commit_preserved=True);save(prior);raise
