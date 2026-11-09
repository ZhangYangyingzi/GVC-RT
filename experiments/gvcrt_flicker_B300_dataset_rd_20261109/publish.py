"""Publish only this derived, lightweight measured R-D package."""
from build import ROOT, load, dump, sha
import hashlib
import subprocess
import time

WORKTREE = ROOT.parents[1]
BRANCH = 'codex/tcg-flicker-pilot-abcd-20261109'


def git(*args):
    return subprocess.run(['git', *args], cwd=WORKTREE, check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    state = dict(status='PREPARED', branch=BRANCH, remote_verified=False,
                 core_blobs_verified=False, time=time.time())
    try:
        assert load(ROOT / 'final_integrity.json')['status'] == 'PASS'
        assert git('branch', '--show-current') == BRANCH
        staged = set(git('diff', '--cached', '--name-only').splitlines())
        remote = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH)
        assert remote
        git('merge-base', '--is-ancestor', remote.split()[0], 'HEAD')
        names = ['build.py', 'publish.py', 'task_identity.json', 'protocol.json',
                 'source_manifest.json', 'status.json', 'final_integrity.json',
                 'REPORT.md', 'metrics_paired.csv', 'rd_points_by_qp.csv',
                 'gains_by_qp.csv', 'paired_deltas_by_video_qp.csv',
                 'stream_identity_checks.csv', 'audits/publication_csv_line_endings.json']
        names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'plots').glob('*.png'))]
        assert len(names) == 24
        for name in names:
            assert (ROOT / name).is_file() and (ROOT / name).stat().st_size < 50 * 2 ** 20
        dump('publication_status.json', state)
        files = [str((ROOT / n).relative_to(WORKTREE)) for n in names + ['publication_status.json']]
        assert staged <= set(files), 'UNRELATED_STAGED_FILES'
        git('add', '--sparse', '--force', '--', *files)
        assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
        git('diff', '--cached', '--check')
        git('commit', '-m', 'B300 measured R-D curves vs Original and A2000 on ULong dev12')
        core = git('rev-parse', 'HEAD')
        git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
        assert git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).split()[0] == core
        stable = [f for f in files if not f.endswith(('/status.json', '/publication_status.json'))]
        checks = []
        for f in stable:
            data = subprocess.check_output(['git', 'show', core + ':' + f], cwd=WORKTREE)
            assert hashlib.sha256(data).hexdigest() == sha(WORKTREE / f)
            checks.append(dict(file=f, SHA256=sha(WORKTREE / f),
                               blob=git('rev-parse', core + ':' + f)))
        state.update(status='PUBLISHED', core_commit=core,
                     commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{core}',
                     remote_verified=True, core_blobs_verified=True, files=checks)
        dump('publication_status.json', state)
        status = load(ROOT / 'status.json')
        status.update(publication_status='PUBLISHED', publication_core_commit=core)
        dump('status.json', status)
        update = [str((ROOT / n).relative_to(WORKTREE)) for n in ['status.json', 'publication_status.json']]
        git('add', '--sparse', '--force', '--', *update)
        git('diff', '--cached', '--check')
        git('commit', '-m', 'Verify B300 ULong dev12 R-D publication receipt')
        head = git('rev-parse', 'HEAD')
        git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
        assert git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).split()[0] == head
        for check in checks:
            assert git('rev-parse', head + ':' + check['file']) == check['blob']
        receipt = dict(status='VERIFIED', branch=BRANCH, commit=head, core_commit=core,
                       commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',
                       remote_verified=True, core_blobs_verified=True, time=time.time())
        dump('publication_remote_receipt.json', receipt)
        print(receipt)
    except Exception as error:
        state.update(status='FAILED', exception=repr(error),
                     stdout=getattr(error, 'stdout', None), stderr=getattr(error, 'stderr', None))
        dump('publication_status.json', state)
        raise


if __name__ == '__main__':
    main()
