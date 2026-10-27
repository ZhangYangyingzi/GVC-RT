"""Restart only this experiment's waiting launcher to load updated GPU policy."""
import os
import signal
import subprocess
import time
from pathlib import Path
from common_run import HERE, PYTHON, load


def main():
    status_path = HERE / 'pipeline_status.json'
    if status_path.exists():
        status = load(status_path)
        if status.get('running'):
            raise RuntimeError('Pipeline has active tasks; refusing to stop it.')
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if len(args) < 2 or b'python' not in Path(os.fsdecode(args[0])).name.encode():
                continue
            cwd = (entry / 'cwd').resolve(strict=True)
            script_args = [os.fsdecode(arg) for arg in args[1:] if arg and not arg.startswith(b'-')]
            if not script_args:
                continue
            script = Path(script_args[0])
            if not script.is_absolute():
                script = cwd / script
            if script.resolve() == HERE / 'run_pipeline.py':
                matches.append(int(entry.name))
        except (OSError, ValueError):
            continue
    for pid in matches:
        # Recheck before terminating only our launcher, never its tasks or other jobs.
        if status_path.exists() and load(status_path).get('running'):
            raise RuntimeError('A task started during inspection; refusing to stop launcher.')
        os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if all(not Path(f'/proc/{pid}').exists() for pid in matches):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old launcher did not exit; no second launcher started.')
    logs = HERE / 'local_artifacts/logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'pipeline.log').open('a') as handle:
        process = subprocess.Popen([PYTHON, '-u', str(HERE / 'run_pipeline.py')], cwd=HERE,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    print('Pipeline started:', process.pid, '; GPUs 0 and 4; no utilization threshold', flush=True)


if __name__ == '__main__':
    main()
