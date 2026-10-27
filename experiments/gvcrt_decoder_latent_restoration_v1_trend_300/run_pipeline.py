"""Resume the fixed experiment using only user-authorized physical GPUs 0 and 4."""
import fcntl
import subprocess
import time

from common_run import HERE, PYTHON, dump, load
from wait_for_gpu import snapshot


def stage(name, tasks):
    pending = list(tasks)
    running = {}
    failures = []
    logs = HERE / 'local_artifacts/logs'
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or running:
            for gpu, (process, handle, task) in list(running.items()):
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                del running[gpu]
                if status:
                    failures.append(f'{task[0]} failed with exit {status}; see {logs}')
                    pending.clear()
            if pending:
                rows, error = snapshot()
                if error:
                    raise RuntimeError(f'GPU query failed: {error}')
                for gpu, used, total, utilization in rows:
                    if gpu not in (0, 4) or gpu in running:
                        continue
                    if total - used < 24000 or not pending:
                        continue
                    task = pending.pop(0)
                    label, script, arguments = task
                    handle = (logs / f'{name}_{label}.log').open('a')
                    command = [PYTHON, str(HERE / script), *map(str, arguments), '--gpu', str(gpu)]
                    process = subprocess.Popen(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT)
                    running[gpu] = (process, handle, task)
            dump(HERE / 'pipeline_status.json', dict(status='RUNNING', stage=name,
                 pending=[x[0] for x in pending], running={str(g): dict(task=x[2][0], pid=x[0].pid) for g, x in running.items()},
                 failures=failures))
            if pending or running:
                time.sleep(15)
        if failures:
            raise RuntimeError('; '.join(failures))
    finally:
        # Never terminate any running process, including our tasks on a stage failure.
        for _, handle, _ in running.values():
            handle.close()


def run():
    lock_path = HERE / 'local_artifacts/pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([PYTHON, str(HERE / 'offline_audit.py')], check=True)
        stage('cache', [(str(i), 'cache_training.py', ['--start', i, '--stop', min(i+75, 300)])
                        for i in range(0, 300, 75)])
        stage('preflight', [('identity_gradient', 'preflight.py', [])])
        for stop in (2, 300):
            tasks = []
            for branch in ('G', 'R', 'RQ'):
                tasks.append((branch, 'train_branch.py', ['--branch', branch, '--stop', stop]))
            stage(f'train_to_{stop}', tasks)
        decode = []; quality = []
        for item in load(HERE / 'stream_manifest.json'):
            for branch in ('A', 'G', 'R', 'RQ'):
                label = f"{item['sequence']}_q{item['external_qp']}_{branch}"
                decode.append((label, 'decode_stream.py', ['--stream', item['stream_path'],
                               '--source', item['sequence'], '--branch', branch]))
                quality.append((label, 'evaluate_quality.py', ['--source', item['sequence'],
                                '--qp', item['external_qp'], '--branch', branch]))
        stage('decode', decode)
        stage('quality', quality)
        stage('timing', [('output_branch', 'benchmark.py', [])])
        subprocess.run([PYTHON, str(HERE / 'finalize.py')], check=True)
        subprocess.run([PYTHON, str(HERE / 'build_overviews.py')], check=True)
        subprocess.run([PYTHON, str(HERE / 'verify_delivery.py')], check=True)
        dump(HERE / 'pipeline_status.json', dict(status='LOCAL_CHECKS_FINISHED',
             remaining=['Update factual REPORT.md', 'Commit and push results',
                        'Verify remote SHA', 'Write COMPLETE.json only after all deliverables pass']))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        dump(HERE / 'pipeline_failure.json', dict(status='BLOCKED_OR_FAILED', error=repr(error)))
        raise
