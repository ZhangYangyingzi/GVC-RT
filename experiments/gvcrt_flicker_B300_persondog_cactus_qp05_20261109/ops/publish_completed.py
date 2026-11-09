"""Resume only publication of completed measurements after a documented check failure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_s2 import *
from task import verify

BRANCH = 'codex/tcg-flicker-pilot-abcd-20261109'


def git(*args):
    return subprocess.run(['git', *args], cwd=WORKTREE, check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    env()
    with lock('runner.lock'):
        assert load(ROOT / 'final_integrity.json')['status'] == 'PASS'
        assert load(ROOT / 'audits/post_export_video_validation.json')['status'] == 'PASS'
        assert verify()
        assert git('branch', '--show-current') == BRANCH
        relative_root = str(ROOT.relative_to(WORKTREE)) + '/'
        staged = git('diff', '--cached', '--name-only').splitlines()
        assert all(f.startswith(relative_root) for f in staged), 'UNRELATED_STAGED_FILES'
        remote = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH)
        assert remote
        git('merge-base', '--is-ancestor', remote.split()[0], 'HEAD')
        files = []
        for p in ROOT.rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            if any(x in {'cache', 'tmp', 'logs', 'locks', 'GT_roi_selection',
                         '__pycache__', 'checkpoints'} for x in p.relative_to(ROOT).parts):
                continue
            if p.suffix not in {'.py', '.md', '.csv', '.json', '.png', '.diff'}:
                continue
            assert p.stat().st_size < 50 * 2 ** 20
            files.append(str(p.relative_to(WORKTREE)))
        state = load(ROOT / 'status.json')
        state.update(status='COMPLETE', complete=True, current_stage='publication_retry',
                     PID=None, publication_status='PENDING', snapshot_time=time.time())
        dump(ROOT / 'status.json', state)
        record = dict(status='PREPARED', branch=BRANCH, remote_verified=False,
                      core_blobs_verified=False, retry_after='documented_whitespace_fix',
                      allowlist=sorted(files), time=time.time())
        dump(ROOT / 'publication_status.json', record)
        for i in range(0, len(files), 100):
            git('add', '--sparse', '--force', '--', *files[i:i + 100])
        assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
        for f in files:
            assert int(git('cat-file', '-s', ':' + f)) < 50 * 2 ** 20
        git('diff', '--cached', '--check')
        git('commit', '-m', 'B300 vs Original: measured person-dog and Cactus QP0/5 with native detail videos')
        core = git('rev-parse', 'HEAD')
        git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
        assert git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).split()[0] == core
        mutable = {str((ROOT / n).relative_to(WORKTREE)) for n in
                   ['status.json', 'publication_status.json', 'publication_remote_receipt.json']}
        checks = []
        for f in files:
            content = subprocess.check_output(['git', 'show', core + ':' + f], cwd=WORKTREE)
            assert hashlib.sha256(content).hexdigest() == sha(WORKTREE / f)
            if f not in mutable:
                checks.append(dict(file=f, blob=git('rev-parse', core + ':' + f),
                                   SHA256=sha(WORKTREE / f)))
        record.update(status='PUBLISHED', core_commit=core,
                      commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{core}',
                      remote_verified=True, core_blobs_verified=True,
                      core_files=checks, verified_at=time.time())
        dump(ROOT / 'publication_status.json', record)
        state.update(current_stage='finished', publication_status='PUBLISHED',
                     publication_core_commit=core, snapshot_time=time.time())
        dump(ROOT / 'status.json', state)
        receipt = dict(status='VERIFIED', branch=BRANCH, core_commit=core,
                       remote_verified=True, core_blobs_verified=True, time=time.time())
        dump(ROOT / 'publication_remote_receipt.json', receipt)
        for f in mutable:
            git('add', '--sparse', '--force', '--', f)
        git('diff', '--cached', '--check')
        git('commit', '-m', 'Verify B300 comparison publication and final COMPLETE status')
        head = git('rev-parse', 'HEAD')
        git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
        assert git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).split()[0] == head
        for check in checks:
            assert git('rev-parse', head + ':' + check['file']) == check['blob']
        receipt.update(commit=head,
                       commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',
                       verified_at=time.time())
        dump(ROOT / 'publication_remote_receipt.json', receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        dump(ROOT / 'audits/publication_resume_failure.json', dict(
            status='FAILED', exception=repr(error), stdout=getattr(error, 'stdout', None),
            stderr=getattr(error, 'stderr', None), time=time.time()))
        raise
