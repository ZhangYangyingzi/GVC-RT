"""Read-only PID/GPU registry and bounded phase snapshots for this experiment."""
from common_s2 import *


def snapshot():
    processes=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'cwd').resolve()!=ROOT:continue
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'python' not in cmd:continue
            e=(p/'environ').read_bytes().split(b'\0');gpu=next((x.split(b'=',1)[1].decode() for x in e if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            processes.append(dict(PID=int(p.name),CUDA_VISIBLE_DEVICES=gpu,command=cmd.strip()))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    updates={}
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';updates[g]=load(p)['committed_updates'] if p.exists() else 0
    coverage={f'{m}_step{s}':len(list((ROOT/f'evaluation/development/{m}/step_{s:04d}/points').glob('*.json')))
        for m,s in [('B300',0),('Original',0),*[(g,s) for g in GROUPS for s in (50,100)]]}
    record=dict(time=time.time(),PID=os.getpid(),committed_updates=updates,evaluation_coverage=coverage,
        cache_h_windows=len(list((ROOT/'cache/h').glob('*.pt'))),processes=processes,
        disk_free_bytes=shutil.disk_usage(ROOT).free)
    dump(ROOT/'readable_status.json',record)
    return record


if __name__=='__main__':
    env()
    with lock('locks/monitor.lock'):
        while not (ROOT/'queue/STOP').exists():snapshot();time.sleep(15)
        snapshot()
