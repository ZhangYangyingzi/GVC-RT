"""Recover result publication only; preserve raw interim CSVs and all frozen sources."""
from eval_parallel_common import *
import csv
import datetime
import io
import traceback

REPAIR = ROOT / 'repairs/publication_csv_v1'


def preserve(relative):
    source = ROOT / relative
    destination = REPAIR / 'before' / (relative + '.gz')
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(gzip.compress(source.read_bytes(), mtime=0))


def parsed(data):
    return list(csv.reader(io.StringIO(data.decode(), newline='')))


def main():
    os.environ.setdefault('GIT_SSH_COMMAND', 'ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3')
    with lock('eval_parallel_controller.lock'), lock('runner_v3.lock'), lock('runner.lock'):
        verify_parallel_sources()
        REPAIR.mkdir(parents=True, exist_ok=True)
        for name in ['run_status.json', 'publication_status_v3.json', 'migrations/eval_parallel_v1/publication_results.json', 'REPORT.md', 'training_summary_v3.json', 'logs/live_status_v3.json', 'interim_results/20261104T083850Z/SHA256_MANIFEST.json']:
            preserve(name)
        final = load(ROOT / 'final_integrity_v3.json')
        assert final['status'] == 'PASS' and final['development_points'] == 960 and final['final_points'] == 675
        selected = load(ROOT / 'selected_models_v3.json')
        evaluation = load(ROOT / 'final_evaluation_v3.json')
        assert sha(ROOT / evaluation['metrics_file']) == evaluation['metrics_sha256']
        protected_paths = [ROOT / name for name in ['protocol.json', 'source_manifest.json', 'protocol_v3.json', 'source_manifest_v3.json', 'selected_models_v3.json', 'final_integrity_v3.json', evaluation['metrics_file'], 'metrics_v3.csv', 'paired_deltas_v3.csv']]
        protected = {str(path.relative_to(ROOT)): sha(path) for path in protected_paths}
        audit_path = REPAIR / 'format_audit.json'
        if not audit_path.exists():
            changes = []
            snapshot = ROOT / 'interim_results/20261104T083850Z'
            for name in ['metrics.csv', 'paired_deltas.csv', 'summary_by_QP.csv']:
                path = snapshot / name
                before = path.read_bytes()
                after = before.replace(b'\r\n', b'\n')
                assert parsed(before) == parsed(after)
                assert before.count(b'\r') == before.count(b'\r\n')
                preserve(str(path.relative_to(ROOT)))
                path.write_bytes(after)
                changes.append(dict(file=str(path.relative_to(ROOT)), before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=sha(path), before_bytes=len(before), after_bytes=len(after), data_rows=len(parsed(after))-1, parsed_values_identical=True, change='CRLF to LF only; original bytes retained as deterministic gzip'))
            manifest = load(snapshot / 'SHA256_MANIFEST.json')
            for entry in manifest['files']:
                path = snapshot / entry['file']
                entry.update(size_bytes=path.stat().st_size, sha256=sha(path))
            manifest['serialization_revision'] = 'publication_csv_v1; original snapshot manifest and bytes retained in repairs/publication_csv_v1/before'
            dump(snapshot / 'SHA256_MANIFEST.json', manifest)
            dump(audit_path, dict(status='PASS', root_cause='Three optional interim snapshot CSVs used CRLF, rejected by git diff --cached --check. Canonical evaluation CSVs and frozen sources unchanged.', changes=changes, canonical_and_frozen_hashes=protected, script_sha256=sha(Path(__file__)), training_or_evaluation_rerun=False, tolerance_changes=False))
        assert all(sha(ROOT / name) == h for name, h in protected.items())
        verify_parallel_sources()
        summary = load(ROOT / 'training_summary_v3.json')
        summary['status'] = 'TRAINING_EVALUATION_INTEGRITY_COMPLETE'
        summary['final_integrity_file'] = 'final_integrity_v3.json'
        dump(ROOT / 'training_summary_v3.json', summary)
        report = ROOT / 'REPORT.md'
        body = report.read_text().replace('"status": "NUMERICAL_RESULTS_READY_PENDING_FINAL_INTEGRITY"', '"status": "TRAINING_EVALUATION_INTEGRITY_COMPLETE"')
        if 'Publication serialization repair v1:' not in body:
            body = body.replace('\n\n', '\n\nPublication serialization repair v1: training and evaluation are complete; final_integrity_v3.json is PASS. Three interim CSVs were converted from CRLF to LF with identical parsed cells; original bytes and failed publication records are retained in repairs/publication_csv_v1/. Publication success is recorded separately in publication_status_v3.json.\n\n', 1)
        report.write_text(body)
        publication = load(ROOT / 'publication_status_v3.json')
        for key in ['error', 'recorded_unix']:
            publication.pop(key, None)
        publication.update(status='RESULT_PUBLICATION_RETRY', previous_failure_record='repairs/publication_csv_v1/before/migrations/eval_parallel_v1/publication_results.json.gz')
        dump(ROOT / 'publication_status_v3.json', publication)
        current = load(ROOT / 'run_status.json')
        current.pop('error', None)
        current.update(status='RUNNING', stage='result_publication', updated=time.time(), active_processes=[], repair='repairs/publication_csv_v1/format_audit.json')
        dump(ROOT / 'run_status.json', current)
        dump(ROOT / 'logs/live_status_v3.json', current)
        command = [PYTHON, '-B', '-u', str(ROOT / 'eval_parallel_publish.py'), '--phase', 'results']
        event('PUBLICATION_CSV_RETRY_STARTED', command=command)
        subprocess.run(command, cwd=REPO, check=True)
        publication = load(ROOT / 'publication_status_v3.json')
        parallel = load(EP / 'publication_results.json')
        assert publication['results_remote_verified'] and publication['core_blobs_verified'] and parallel['addendum_core_blobs_verified']
        assert all(sha(ROOT / name) == h for name, h in protected.items())
        import v3_publish
        pub = Path(load(ROOT / 'audits/publication_v3_worktree.local.json')['path'])
        data_head = v3_publish.remote_head()
        assert data_head == publication['results_commit']
        current.update(status='PASS', stage='complete', updated=time.time(), active_processes=[], verified_commit=data_head)
        dump(ROOT / 'run_status.json', current)
        dump(ROOT / 'logs/live_status_v3.json', current)
        completion = dict(status='PASS', repaired_csv_cells_identical=True, training_and_evaluation_not_repeated=True, final_integrity='PASS', parent_GTFlow_reused_points=156, development_points=960, final_points=675, K_common=selected['K_common'], verified_results_commit=data_head, remote_verified=True, core_blobs_verified=True, recovery_script_sha256=sha(Path(__file__)), completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump(REPAIR / 'completion.json', completion)
        # Publish the completion snapshot in a normal successor commit. The snapshot
        # names its already-verified data commit, avoiding a self-referential hash.
        names = ['run_status.json', 'logs/live_status_v3.json', 'migrations/eval_parallel_v1/publication_results.json', 'repairs/publication_csv_v1/completion.json']
        assert not v3_publish.git_run(['diff', '--cached', '--name-only'], pub)
        for name in names:
            target = pub / v3_publish.REL / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        paths = [v3_publish.REL + '/' + name for name in names]
        v3_publish.git_run(['add', '--', *paths], pub)
        staged = v3_publish.git_run(['diff', '--cached', '--name-only'], pub).splitlines()
        assert set(staged) <= set(paths)
        assert all((pub / name).stat().st_size < 50 * 2**20 for name in staged)
        v3_publish.git_run(['diff', '--cached', '--check'], pub)
        if staged:
            v3_publish.git_run(['commit', '-m', 'Record verified TCG result publication recovery and completed execution'], pub)
        head = v3_publish.git_run(['rev-parse', 'HEAD'], pub)
        v3_publish.git_run(['push', 'origin', 'HEAD:refs/heads/' + v3_publish.BRANCH], pub)
        assert v3_publish.remote_head() == head
        for name in ['REPORT.md', 'metrics_v3.csv', 'paired_deltas_v3.csv', 'final_integrity_v3.json', *names]:
            blob = subprocess.check_output(['git', 'show', f'{head}:{v3_publish.REL}/{name}'], cwd=pub)
            assert hashlib.sha256(blob).hexdigest() == sha(ROOT / name)
        publication.update(results_commit=head, results_commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}', results_remote_verified=True, core_blobs_verified=True, status='PUBLISHED')
        dump(ROOT / 'publication_status_v3.json', publication)
        current['verified_commit'] = head
        current['updated'] = time.time()
        dump(ROOT / 'run_status.json', current)
        dump(ROOT / 'logs/live_status_v3.json', current)
        completion.update(verified_commit=head, commit_url=publication['results_commit_url'])
        dump(REPAIR / 'remote_verification.json', completion)
        event('PUBLICATION_CSV_RECOVERY_COMPLETE', commit=head, branch=v3_publish.BRANCH)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = dict(status='FAILED', stage='result_publication', error=traceback.format_exc(), updated=time.time(), resume_command=f'{PYTHON} -B -u {Path(__file__)}')
        dump(REPAIR / f'failure_{time.time_ns()}.json', failure)
        dump(ROOT / 'run_status.json', failure)
        traceback.print_exc()
        sys.exit(1)
