"""Publish only explicitly listed stopped-experiment records, without GPU work."""
from common_s2 import *

BRANCH = 'codex/tcg-stage2-history-gate-joint-20261106'
BASE_URL = 'https://github.com/ZhangYangyingzi/GVC-RT'


def git(*args, check=True):
    p = subprocess.run(['git', *args], cwd=WORKTREE, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (ROOT/'logs/user_stop_publication.log').open('a') as f:
        f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
    if check and p.returncode:
        raise RuntimeError('GIT_COMMAND_FAILED '+str(args)+'\n'+p.stdout[-3000:])
    return p.stdout.strip(), p.returncode


def main():
    env()
    with lock('publication.lock'):
        try:
            assert load(ROOT/'run_status.json')['status'] == 'STOPPED_BY_USER'
            assert load(ROOT/'final_integrity.json')['status'] == 'NOT_COMPLETED_USER_STOP'
            assert git('branch','--show-current')[0] == BRANCH
            assert 'ZhangYangyingzi/GVC-RT' in git('remote','get-url','origin')[0]
            remote = git('ls-remote','--heads','origin','refs/heads/'+BRANCH)[0]
            assert remote, 'EXISTING_BRANCH_NOT_FOUND'
            git('fetch','origin',BRANCH)
            assert git('merge-base','--is-ancestor',remote.split()[0],'HEAD',check=False)[1] == 0, 'REMOTE_ADVANCED_OR_DIVERGED'
            names = ['REPORT.md','REPORT_USER_STOP_20261106.md','generate_user_stop_report.py','publish_user_stop_report.py',
                'run_status.json','training_status.json','active_processes.json','stopping_state.json','final_integrity.json',
                'training_history.csv','training_coverage.json','checkpoints_manifest.json','development_metrics.csv',
                'development_summary_user_stop.csv','development_by_QP_user_stop.csv','development_step1000_partial.csv',
                'precision_baseline_comparison.json','precision_baseline_deltas.csv']
            names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'audits/user_stop_20261106').glob('*.json'))]
            names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'stopped_artifacts').glob('*.json.gz'))]
            names += [f'audits/training_segment_{s:04d}.json' for s in [250,500,750,1000]]
            for batch in ['development_matched_FP32','development_joint_0000','development_joint_0250','development_joint_0500','development_joint_0750']:
                names += [f'evaluation/{batch}/metrics.csv',f'evaluation/{batch}/integrity.json']
            files = [ROOT/n for n in sorted(set(names))]
            for p in files:
                assert p.is_file() and not p.is_symlink() and p.is_relative_to(ROOT)
                assert p.stat().st_size < 50*2**20
                assert p.suffix in ['.md','.py','.json','.csv','.gz']
                assert not any(x in p.relative_to(ROOT).parts for x in ['checkpoints','cache','large_artifacts','logs'])
            manifest = ROOT/'user_stop_publication_allowlist.json'
            dump(manifest,dict(scope='User-stopped experiment report and bounded records; no full-experiment completion claim',
                files=[dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
            files.append(manifest)
            allowed = [str(p.relative_to(WORKTREE)) for p in files]
            staged = git('diff','--cached','--name-only','-z')[0]
            assert all(p in allowed for p in staged.split('\0') if p), 'UNRELATED_STAGED_FILES'
            for i in range(0,len(allowed),50):git('add','--',*allowed[i:i+50])
            git('diff','--cached','--check')
            staged = [p for p in git('diff','--cached','--name-only','-z')[0].split('\0') if p]
            assert all(p in allowed for p in staged)
            for name in staged:
                p=WORKTREE/name
                assert p.stat().st_size < 50*2**20
                assert git('rev-parse',':'+name)[0] == git('hash-object',str(p))[0]
            dump(ROOT/'audits/user_stop_20261106/staged_files_check.json',dict(status='PASS',file_count=len(staged),
                task_only=True,all_files_below_50_MiB=True,staged_content_matches_files=True,files=staged,time=time.time()))
            # Add the staged-file audit explicitly after validating all primary files.
            audit_path=str((ROOT/'audits/user_stop_20261106/staged_files_check.json').relative_to(WORKTREE))
            git('add','--',audit_path);git('diff','--cached','--check')
            if git('diff','--cached','--quiet',check=False)[1]:
                git('commit','-m','Record user stop at 1000 joint updates and development results')
            head=git('rev-parse','HEAD')[0]
            git('push','origin',f'HEAD:refs/heads/{BRANCH}')
            assert git('ls-remote','--heads','origin','refs/heads/'+BRANCH)[0].split()[0] == head
            core=['REPORT.md','REPORT_USER_STOP_20261106.md','run_status.json','final_integrity.json','development_metrics.csv','training_coverage.json']
            blobs=[]
            for name in core:
                p=ROOT/name;rel=str(p.relative_to(WORKTREE))
                local=git('hash-object',str(p))[0]
                assert git('rev-parse',f'{head}:{rel}')[0] == local
                assert git('rev-parse',f'refs/remotes/origin/{BRANCH}:{rel}')[0] == local
                blobs.append(dict(file=name,blob=local))
            dump(ROOT/'publication_status.json',dict(status='PUBLISHED',phase='user_stopped_results',branch=BRANCH,commit=head,
                commit_url=BASE_URL+'/commit/'+head,remote_verified=True,core_blobs_verified=True,core_blobs=blobs,
                experiment_status='STOPPED_BY_USER',experiment_complete=False,time=time.time()))
            git('add','--',str((ROOT/'publication_status.json').relative_to(WORKTREE)));git('diff','--cached','--check')
            if git('diff','--cached','--quiet',check=False)[1]:git('commit','-m','Record verified publication of user-stopped experiment')
            audit_head=git('rev-parse','HEAD')[0]
            git('push','origin',f'HEAD:refs/heads/{BRANCH}')
            assert git('ls-remote','--heads','origin','refs/heads/'+BRANCH)[0].split()[0] == audit_head
            dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',branch=BRANCH,published_artifact_commit=head,
                branch_HEAD=audit_head,HEAD_commit_url=BASE_URL+'/commit/'+audit_head,remote_verified=True,core_blobs_verified=True,
                experiment_status='STOPPED_BY_USER',experiment_complete=False,time=time.time()))
            print(json.dumps(dict(status='PUBLISHED',experiment_status='STOPPED_BY_USER',report_commit=head,branch_HEAD=audit_head,remote_verified=True)))
        except Exception as exc:
            dump(ROOT/'publication_status.json',dict(status='FAILED',phase='user_stopped_results',branch=BRANCH,
                error=str(exc),experiment_status='STOPPED_BY_USER',experiment_complete=False,remote_verified=False,time=time.time()))
            raise


if __name__ == '__main__':
    main()
