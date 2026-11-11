"""Two bounded producers, training-first, isolated Original codec subprocesses."""
from common_s2 import *
import traceback


def cursors():
    out=[]
    for g in GROUPS:
        p=ROOT/g/'training/progress.json'
        out.append(load(p)['next_cursor'] if p.exists() else 0)
    return out


def worker():
    env();gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (2,3)
    from cache_h import ready,ensure
    from cache_codec import cache_paths
    c=None;rows=plan_rows();produced=0
    tag=os.environ.get('CACHE_WORKER_TAG','')
    initial=rows[:4]+load(ROOT/'manifests/calibration_windows.json')+load(ROOT/'manifests/monitor_windows.json')
    while not (ROOT/'queue/STOP').exists():
        pos=min(cursors());wanted=initial+rows[pos:min(400,pos+20)]
        wanted=list({r['sample_key']:r for r in wanted}.values())
        pending=[r for r in wanted if not ready(r)]
        dump(ROOT/f'queue/cache_GPU{gpu}{tag}.json',dict(PID=os.getpid(),physical_GPU=gpu,cursors=cursors(),
            pending_slots=[r['plan_slot'] for r in pending],bounded_prefetch=20,time=time.time()))
        found=False
        for row in pending:
            lease=ROOT/f'locks/produce_{row["sample_key"]}.lock'
            with lease.open('a') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if ready(row):continue
                found=True;spec=ROOT/'tmp/cache_jobs'/f'{row["sample_key"]}.json';dump(spec,row)
                _,p,rp=cache_paths(row,row['external_qp'],True)
                if not p.exists() or not rp.exists():
                    with (ROOT/f'logs/original_cache_GPU{gpu}.log').open('ab') as log:
                        subprocess.run([PYTHON,'-B','-u','cache_codec.py','--row',str(spec),'--qp',str(row['external_qp']),'--training'],cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
                if c is None:
                    from experiment_model import construct
                    c=construct('B300',True,0,with_adapter=False)
                    dump(ROOT/f'audits/B300_full_load_GPU{gpu}.json',dict(status='PASS',provenance=c['provenance'],time=time.time()))
                ensure(row,context=c)
                produced+=1
                # Task-owned disposable original cache; never touch historical sources.
                _,p,rp=cache_paths(row,row['external_qp'],True)
                if p.is_relative_to(ROOT) and row['plan_slot']>=4 and p.exists():p.unlink()
                break
        if not found:time.sleep(2)
        limit=int(os.environ.get('CACHE_MAX_WINDOWS','0'))
        if limit and produced>=limit:
            dump(ROOT/f'audits/cache_worker{tag}_completion.json',dict(status='PASS',produced=produced,physical_GPU=gpu,PID=os.getpid(),time=time.time()));return


if __name__=='__main__':
    try:worker()
    except Exception as e:
        env();gpu=os.environ.get('CUDA_VISIBLE_DEVICES');dump(ROOT/f'audits/cache_pipeline_failure_GPU{gpu}_{os.getpid()}.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
