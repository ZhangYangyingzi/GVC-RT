"""Nonintrusive queue for physical GPUs 0 and 4; never touches other processes."""
import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone

from common_run import HERE, dump


def snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        return None, result.stderr.strip()
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) == 4:
            rows.append(tuple(int(x.strip()) for x in row))
    return rows, None


def wait():
    begun = time.time(); attempts = 0; failures = 0
    while True:
        rows, error = snapshot(); attempts += 1
        if error:
            failures += 1
        else:
            for index, used, total, utilization in rows:
                if index in (0, 4) and total-used >= 24000:
                    result = dict(physical_gpu=index, utc=datetime.now(timezone.utc).isoformat(),
                                  waited_seconds=time.time()-begun, polls=attempts,
                                  nvidia_smi_failures=failures, final_snapshot=rows)
                    dump(HERE / 'resource_wait.json', result)
                    print(index, flush=True)
                    return
        if attempts % 10 == 0:
            dump(HERE / 'resource_wait.json',
                 dict(status='WAITING', utc=datetime.now(timezone.utc).isoformat(),
                      waited_seconds=time.time()-begun, polls=attempts,
                      nvidia_smi_failures=failures, last_error=error, last_snapshot=rows))
        time.sleep(30)


if __name__ == '__main__':
    wait()
