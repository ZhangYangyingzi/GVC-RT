"""Overlap unchanged, cached CPU diagnostics with the remaining RD queue."""
from io21 import *
import fcntl,argparse
from diagnostic_cache import SCRIPTS,valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');a=ap.parse_args()
    if a.launch:
        status=ROOT/'cpu_diagnostics_status.json'
        if status.exists():
            pid=load(status).get('PID');cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and str(Path(__file__).resolve()).encode() in cmd.read_bytes():print('CPU DIAGNOSTICS ALREADY ACTIVE',pid);return
        log=(ROOT/'logs/early_cpu_diagnostics.log').open('ab')
        argv=[PYTHON,'-B','-u',str(Path(__file__).resolve())];command(argv)
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        print('CPU DIAGNOSTICS STARTED',p.pid,flush=True);return
    lock=(ROOT/'cpu_diagnostics.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending={(script,d) for script in SCRIPTS for d in (*DATASETS,'mcljcv_exclude_overlap29')}
    active={};done=[]
    while pending or active:
        for key,(p,log) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            if rc!=0:raise RuntimeError(f'CPU diagnostic failed: {key}, exit {rc}, log {log}')
            assert valid(*key),key
            done.append(key);del active[key]
        for script,d in sorted(pending):
            if len(active)>=4:break
            source='mcljcv' if d.endswith('29') else d
            if not all(point(source,m,v['video_index'],q).exists() for v in sources(source) for m in METHODS for q in range(10)):continue
            argv=[PYTHON,'-B','-u',str(ROOT/'diagnostic_cache.py'),'--script',script,'--dataset',d]
            log=ROOT/'logs'/f'early_{script[:-3]}_{d}.log';command(argv)
            p=subprocess.Popen(argv,stdout=log.open('ab'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
               env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            active[(script,d)]=(p,str(log));pending.remove((script,d))
        dump(ROOT/'cpu_diagnostics_status.json',dict(status='RUNNING' if pending or active else 'PASS',PID=os.getpid(),
             completed=done,active=[dict(script=s,dataset=d,PID=p.pid,log=log) for (s,d),(p,log) in active.items()],
             pending=sorted(pending),maximum_workers=4,threads_per_worker=2))
        if pending or active:time.sleep(15)

if __name__=='__main__':main()
