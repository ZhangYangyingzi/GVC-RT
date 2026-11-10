"""One queue owner, bounded GPU helpers, strict consumers and safe disk eviction."""
from common_s2 import *
from execution_policy import h2_training_allowed
from cache_codec import cache_paths
import traceback

def ready(row,qp,training,verify=False):
    key,p,rp=cache_paths(row,qp,training)
    try:
        # Readiness must not expose a tensor whose writer is still publishing its receipt.
        with lock('locks/derived_'+key+'.lock',blocking=verify):
            if not p.exists() or not rp.exists():return False
            a=load(rp);assert a['status']=='PASS' and a['cache_identity']==key and p.stat().st_size==a['cache_bytes']
            if verify:assert sha(p)==a['cache_sha256']
            return True
    except (BlockingIOError,FileNotFoundError):return False
    except (AssertionError,json.JSONDecodeError) as error:
        # Preserve the error record; retire only this task's invalid derived copy.
        with lock('locks/lease_'+key+'.lock',blocking=True),lock('locks/derived_'+key+'.lock',blocking=True):
            report=dict(status='REJECTED_REBUILD_REQUESTED',key=key,path=str(p),receipt=str(rp),reason=repr(error),observed_SHA256=sha(p) if p.exists() else None,previous_receipt_SHA256=sha(rp) if rp.exists() else None,original_sources_modified=False,time=time.time())
            dump(ROOT/'audits/cache_rejected'/f'{key}.json',report)
            if p.is_relative_to(ROOT) and p.exists():p.unlink()
            if rp.is_relative_to(ROOT) and rp.exists():
                dest=ROOT/'audits/codec_cache_history'/key/f'rejected_{time.time_ns()}.json';dest.parent.mkdir(parents=True,exist_ok=True);rp.replace(dest)
        return False

def request(row,qp,training):
    key,_,_=cache_paths(row,qp,training)
    if not ready(row,qp,training):dump(ROOT/'cache/requests'/f'{key}.json',dict(row=row,qp=qp,training=training,requested=time.time()))
    return key

def wait_rows(rows,training=True,qp=None):
    begin=time.time()
    while True:
        for r in rows:request(r,r['external_qp'] if qp is None else qp,training)
        if all(ready(r,r['external_qp'] if qp is None else qp,training) for r in rows):
            if all(ready(r,r['external_qp'] if qp is None else qp,training,True) for r in rows):return time.time()-begin
        failure=ROOT/'audits/cache_failure.json'
        if failure.exists():raise RuntimeError(('CACHE_PRODUCTION_FAILURE',load(failure)))
        time.sleep(.5)

def positions():
    out={}
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';a=load(p)['committed_updates'] if p.exists() else 0
        out[g]=a*4
    gate=ROOT/'audits/H2_gate.json'
    if gate.exists() and not h2_training_allowed():out[GROUPS[2]]=400
    if (ROOT/'training_completion.json').exists():return {g:400 for g in GROUPS}
    return out

def inventory():return sum(p.stat().st_size for p in (ROOT/'cache/derived').glob('*.pt'))

def evict_training(rows,positions):
    safe=min(positions.values()) if (ROOT/'gradient_calibration.json').exists() else 0
    for r in rows[:safe]:
        key,p,rp=cache_paths(r,r['external_qp'],True)
        if p.is_relative_to(ROOT) and p.exists():
            with lock('locks/derived_'+key+'.lock',blocking=True):
                if p.exists():p.unlink()
        req=ROOT/'cache/requests'/f'{key}.json'
        if req.exists():req.unlink()

def evict_native(cfg,need_bytes=0):
    candidates=[]
    for p in (ROOT/'cache/derived').glob('*.pt'):
        rp=ROOT/'audits/codec_cache'/f'{p.stem}.json'
        if rp.exists() and not load(rp)['training']:candidates.append(p)
    for p in sorted(candidates,key=lambda p:p.stat().st_mtime):
        if inventory()+need_bytes<=cfg['derived_byte_cap'] and shutil.disk_usage(ROOT).free-need_bytes>=cfg['min_free_bytes'] and len(candidates)<=cfg['max_native_ready']:break
        try:
            with lock('locks/lease_'+p.stem+'.lock'):
                if p.exists():
                    rp=ROOT/'audits/codec_cache'/f'{p.stem}.json'
                    if rp.exists():
                        archived=ROOT/'audits/codec_cache_history'/p.stem/f'{time.time_ns()}.json';archived.parent.mkdir(parents=True,exist_ok=True);rp.replace(archived)
                    p.unlink();candidates.remove(p)
        except BlockingIOError:continue

def task_list(rows,cfg,pos):
    # Retention uses formal committed cursors; prefetch uses the active consumer phase.
    demand=dict(pos)
    for group in GROUPS:
        engineering=ROOT/group/'engineering/progress.json'
        if not (ROOT/group/'training/latest.pt').exists() and engineering.exists() and not (ROOT/f'audits/engineering_{group}.json').exists():
            demand[group]=load(engineering)['next_cursor']
    cursors=sorted({n for n in demand.values() if n<400});safe=min(pos.values());requested=[]
    for p in sorted((ROOT/'cache/requests').glob('*.json'),key=lambda p:p.stat().st_mtime):
        a=load(p)
        if a['training'] and a['row'].get('plan_slot',400)<safe:p.unlink();continue
        if not ready(a['row'],a['qp'],a['training']):requested.append(a)
    # Bounded, fair lookahead for groups at different committed cursors.
    # H2 backfill does not prevent H0/H1 next-batch preparation.
    limit=max(4,cfg['training_prefetch_windows']//max(1,len(cursors)))
    planned=[]
    for offset in range(limit):
        for cursor in cursors:
            index=cursor+offset
            if index>=400:continue
            row=rows[index]
            if not ready(row,row['external_qp'],True):planned.append(dict(row=row,qp=row['external_qp'],training=True))
    unique=[];seen=set()
    for item in planned:
        key=item['row']['sample_key']
        if key not in seen:unique.append(item);seen.add(key)
    planned=unique
    urgent={rows[i]['sample_key'] for cursor in cursors for i in range(cursor,min(400,cursor+4))}
    native=[a for a in requested if not a['training']];extra=[a for a in requested if a['training'] and a['row']['sample_key'] not in {x['row']['sample_key'] for x in planned}]
    early=[a for a in planned+extra if a['row']['sample_key'] in urgent];later=[a for a in planned+extra if a['row']['sample_key'] not in urgent]
    enough=all(sum(ready(r,r['external_qp'],True) for r in rows[n:min(400,n+16)])>=min(16,400-n) for n in cursors)
    return early+(native+later if enough else later+native)

def produce():
    env();rows=plan_rows();active={};produced=0;begin=time.time();latency=0.
    with lock('locks/producer.lock'):
        try:
            while True:
                cfg=load(ROOT/'scheduler_config.json');stopping=(ROOT/'cache/STOP').exists()
                for key,j in list(active.items()):
                    code=j['p'].poll();raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','-i',str(j['gpu'])],text=True).strip().split(',')
                    j['peak']=max(j['peak'],int(raw[0]));j['utilization']=int(raw[1])
                    if code is None:continue
                    assert code==0,('CACHE_CHILD_FAILED',key,code,j['gpu']);a=j['job']
                    assert ready(a['row'],a['qp'],a['training'],True),('CACHE_COMPLETION_NOT_VERIFIABLE',key,j['gpu'])
                    req=ROOT/'cache/requests'/f'{key}.json'
                    if req.exists():req.unlink()
                    dt=time.time()-j['start'];produced+=1;latency+=dt;event('shared_cache_READY',key=key,training=a['training'],seconds=dt,physical_GPU=j['gpu'],total_GPU_peak_MiB=j['peak'],PID=j['p'].pid);j['log'].close();j.get('gpu_lease') and j['gpu_lease'].close();active.pop(key)
                pos=positions();evict_training(rows,pos);evict_native(cfg)
                assert set(cfg['cache_GPUs']).issubset({0,1,2,3})
                jobs=task_list(rows,cfg,pos);free=[g for g in cfg['cache_GPUs'] if g not in {j['gpu'] for j in active.values()}][:max(0,cfg['max_cache_workers']-len(active))]
                backpressure=False
                if not stopping:
                    for a in jobs:
                        key,_,_=cache_paths(a['row'],a['qp'],a['training'])
                        if key in active:continue
                        if not free:break
                        estimate=(1 if a['training'] else 7)*2**30;evict_native(cfg,estimate+sum(j['reserve'] for j in active.values()))
                        if inventory()+estimate+sum(j['reserve'] for j in active.values())>cfg['derived_byte_cap'] or shutil.disk_usage(ROOT).free-estimate-sum(j['reserve'] for j in active.values())<cfg['min_free_bytes']:backpressure=True;break
                        gpu=free.pop(0);lease=(ROOT/f'locks/GPU{gpu}.lock').open('a')
                        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:lease.close();break
                        file=ROOT/'tmp/cache_jobs'/f'{key}.json';dump(file,a)
                        e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MODEL_KIND='original');log=(ROOT/'logs'/f'cache_GPU{gpu}.log').open('ab')
                        p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'cache_job.py'),str(file)],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                        active[key]=dict(p=p,gpu=gpu,job=a,start=time.time(),peak=0,utilization=0,reserve=estimate,log=log,gpu_lease=lease)
                elapsed=time.time()-begin
                dump(ROOT/'cache/progress.json',dict(status='DRAINING' if stopping else 'BACKPRESSURE' if backpressure else 'RUNNING',READY_training_windows=sum(ready(r,r['external_qp'],True) for r in rows),cache_bytes=inventory(),positions=pos,next_windows={g:rows[n]['sample_key'] if n<400 else None for g,n in pos.items()},produced=produced,windows_per_second=produced/elapsed if elapsed else None,producer_latency_sum=latency,active={k:dict(PID=j['p'].pid,physical_GPU=j['gpu'],peak_MiB=j['peak'],GPU_utilization=j['utilization']) for k,j in active.items()},time=time.time()))
                if stopping and not active:break
                time.sleep(1)
        except Exception as error:
            dump(ROOT/'audits/cache_failure.json',dict(status='FAILED',exception=repr(error),traceback=traceback.format_exc(),keys=list(active),time=time.time()))
            for j in active.values():j['p'].wait();j['log'].close();j.get('gpu_lease') and j['gpu_lease'].close()
            raise
if __name__=='__main__':produce()
