"""A single verified producer, bounded ordered lookahead and recovery-safe eviction.

No all-training or all-development startup barrier. Consumers enter at optimizer
boundaries; ranks agree readiness BEFORE forward/backward or gradient collectives.
"""
from common_s2 import *
from cache_codec import cache_paths
import argparse,traceback

def ready(row,qp,training,verify=False):
    key,p,rp=cache_paths(row,qp,training)
    if not p.exists() or not rp.exists():return False
    a=load(rp)
    assert a['status']=='PASS' and a['cache_identity']==key and p.stat().st_size==a['cache_bytes'],('INVALID_CACHE_RECEIPT',key)
    if verify:assert sha(p)==a['cache_sha256'],('INVALID_CACHE_HASH',key)
    return True

def cursors():
    out={}
    for g in GROUPS:
        p=ROOT/g/'training/progress.json'
        # progress is published only AFTER the atomic latest checkpoint commit.
        out[g]=load(p)['committed_updates']*4 if p.exists() else 0
        d=ROOT/g/'training/completion.json'
        if d.exists() and load(d)['status']=='PASS':out[g]=1200
    return out

def inventory():
    files=list((ROOT/'cache/derived').glob('*.pt'))
    return sum(p.stat().st_size for p in files),len(files)

def trim(rows,positions):
    safe=min(positions.values())
    for r in rows[:safe]:
        key,p,rp=cache_paths(r,r['external_qp'],True)
        if p.exists() and p.is_relative_to(ROOT):
            with lock('locks/derived_'+key+'.lock',blocking=True):
                # Retain immutable stream and hash receipt for identity auditing.
                if p.exists():p.unlink()
                req=ROOT/'cache/requests'/f'{key}.json'
                if req.exists():req.unlink()

def pending_jobs(rows,positions,cfg):
    # Both groups have exactly the same immutable order, never completion order.
    lo=min(positions.values());hi=min(1200,lo+cfg['prefetch_windows'])
    wanted=list(range(lo,hi))
    for g,cur in positions.items():
        if cur<1200:
            for i in range(cur,min(cur+4,1200)):
                if i not in wanted:wanted.insert(0,i)
    for i in wanted:
        r=rows[i]
        if not ready(r,r['external_qp'],True):yield r,r['external_qp'],True
    # Dev jobs are requested on demand at a validation boundary, with priority
    # over speculative training lookahead in the producer main loop.

def requests():
    safe=min(cursors().values())
    for p in sorted((ROOT/'cache/requests').glob('*.json')):
        a=load(p)
        if a['training'] and a['row'].get('plan_slot',1200)<safe:
            p.unlink();continue
        if not ready(a['row'],a['qp'],a['training']):yield a['row'],a['qp'],a['training']

def request(row,qp,training=False):
    key,_,_=cache_paths(row,qp,training)
    if not ready(row,qp,training):dump(ROOT/'cache/requests'/f'{key}.json',dict(row=row,qp=qp,training=training))
    return key

def wait_rows(rows,qp=None,training=True,group=None):
    start=time.time()
    for r in rows:request(r,r['external_qp'] if qp is None else qp,training)
    while not all(ready(r,r['external_qp'] if qp is None else qp,training) for r in rows):
        failure=ROOT/'audits/cache_producer_failure.json'
        if failure.exists():raise RuntimeError(('CACHE_PRODUCER_FAILED',load(failure)))
        time.sleep(.5)
    elapsed=time.time()-start
    if group:
        p=ROOT/group/'cache_wait.json';old=load(p) if p.exists() else dict(seconds=0)
        dump(p,dict(seconds=old['seconds']+elapsed,next_sample_keys=[r['sample_key'] for r in rows if 'sample_key' in r],time=time.time()))
    return elapsed

def producer():
    env();rows=plan_rows();active={};done=0;seconds=0.;started=time.time();known=set()
    for line in (ROOT/'logs/events.jsonl').read_text().splitlines():
        a=json.loads(line)
        if a.get('event')=='cache_READY' and a['key'] not in known:
            known.add(a['key']);done+=1;seconds+=a['seconds']
    completed_start=done
    with lock('locks/cache_producer.lock'):
        try:
            while True:
                cfg=load(ROOT/'scheduler_config.json');devices=cfg.get('cache_GPUs',[cfg.get('cache_GPU',0)])
                assert len(devices)==len(set(devices))==cfg['gpu_cache_workers']
                stopping=(ROOT/'cache/STOP').exists()
                for key,job in list(active.items()):
                    rc=job['process'].poll()
                    # Record total device memory, including other jobs; a paused
                    # rank is never treated as releasing its allocations.
                    used_mib=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(job['GPU'])],text=True).strip())
                    job['peak_MiB']=max(job['peak_MiB'],used_mib)
                    if rc is None:continue
                    job['log'].close()
                    assert rc==0,('CACHE_CHILD_FAILED',key,rc,job['GPU'])
                    row,q,tr=job['task'];assert ready(row,q,tr,True)
                    request_path=ROOT/'cache/requests'/f'{key}.json'
                    if request_path.exists():request_path.unlink()
                    elapsed=time.time()-job['started'];done+=1;seconds+=elapsed
                    event('cache_READY',key=key,training=tr,seconds=elapsed,physical_GPU=job['GPU'],total_GPU_peak_MiB=job['peak_MiB'],producer_pid=job['process'].pid,controlled_GPU_workers=len(devices))
                    active.pop(key)
                positions=cursors();trim(rows,positions);used,count=inventory()
                # Each key is assigned once by this sole queue controller; the
                # inherited per-key flock additionally protects atomic writers.
                jobs=list(requests()) or list(pending_jobs(rows,positions,cfg))
                available=[i for i in devices if i not in {j['GPU'] for j in active.values()}]
                backpressure=False
                if not stopping:
                    for row,q,tr in jobs:
                        key,_,_=cache_paths(row,q,tr)
                        if key in active or ready(row,q,tr):continue
                        if not available:break
                        estimate=(1 if tr else 8)*2**30
                        reserved=sum(j['reserved'] for j in active.values())
                        if used+reserved+estimate>cfg['derived_byte_cap'] or shutil.disk_usage(ROOT).free-reserved-estimate<cfg['reserve_bytes']:
                            backpressure=True;break
                        gpu=available.pop(0);rowfile=ROOT/'tmp/cache_jobs'/f'{key}.json';dump(rowfile,row)
                        args=[PYTHON,'-B','-u',str(ROOT/'cache_codec.py'),'--row',str(rowfile),'--qp',str(q),'--rank','0']
                        if tr:args+=['--training']
                        e=dict(os.environ,MODEL_KIND='original',CUDA_VISIBLE_DEVICES=str(gpu))
                        log=(ROOT/'logs'/f'cache_GPU{gpu}.log').open('ab')
                        child=subprocess.Popen(args,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                        active[key]=dict(process=child,task=(row,q,tr),GPU=gpu,reserved=estimate,peak_MiB=0,started=time.time(),log=log)
                elapsed=time.time()-started
                progress=dict(status='DRAINING' if stopping else 'BACKPRESSURE' if backpressure else 'RUNNING',time=time.time(),READY_windows=sum(ready(r,r['external_qp'],True) for r in rows),cache_bytes=used,cache_files=count,positions=positions,produced=done,production_seconds=seconds,
                    windows_per_second=(done-completed_start)/elapsed if elapsed else None,cumulative_producer_window_seconds=seconds,produced_since_controller_start=done-completed_start,controller_started=started,
                    next_needed={g:rows[c]['sample_key'] if c<1200 else None for g,c in positions.items()},gpu_workers=len(devices),cache_GPUs=devices,active={k:dict(physical_GPU=j['GPU'],PID=j['process'].pid,started=j['started'],peak_MiB=j['peak_MiB']) for k,j in active.items()})
                dump(ROOT/'cache/progress.json',progress)
                if stopping and not active:break
                time.sleep(1)
        except Exception as error:
            failed=[dict(key=k,GPU=j['GPU'],PID=j['process'].pid,returncode=j['process'].poll(),row=j['task'][0]) for k,j in active.items()]
            dump(ROOT/'audits/cache_producer_failure.json',dict(status='FAILED',exception=repr(error),tasks=failed,traceback=traceback.format_exc(),time=time.time()))
            # Drain existing children so recovery never starts extra untracked
            # GPU production. Each already-completed atomic cache is retained.
            for j in active.values():j['process'].wait();j['log'].close()
            raise

if __name__=='__main__':producer()
