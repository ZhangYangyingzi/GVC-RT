"""One queue owner, bounded GPU helpers, strict consumers and safe disk eviction."""
from common_s2 import *
from cache_codec import cache_paths
from cache_teacher import paths as teacher_paths
import traceback

def ready(row,qp,training,verify=False):
    key,p,rp=cache_paths(row,qp,training)
    try:
        # Readiness must not expose a tensor whose writer is still publishing its receipt.
        with lock('locks/derived_'+key+'.lock',blocking=verify):
            if not p.exists() or not rp.exists():return False
            a=load(rp);assert a['status']=='PASS' and a['cache_identity']==key and p.stat().st_size==a['cache_bytes']
            if verify:assert sha(p)==a['cache_sha256']
            if training:
                _,t,tp=teacher_paths(row,qp)
                if not t.exists() or not tp.exists():return False
                ta=load(tp);assert ta['status']=='PASS' and ta['codec_identity']==key and ta['Teacher_SHA256']==TEACHER_SHA
                if verify:assert sha(t)==ta['SHA256']
            return True
    except (BlockingIOError,FileNotFoundError):return False

def request(row,qp,training):
    key,_,_=cache_paths(row,qp,training)
    if not ready(row,qp,training):dump(ROOT/'cache/requests'/f'{key}.json',dict(row=row,qp=qp,training=training,requested=time.time()))
    return key

def wait_rows(rows,training=True,qp=None):
    begin=time.time()
    for r in rows:request(r,r['external_qp'] if qp is None else qp,training)
    while not all(ready(r,r['external_qp'] if qp is None else qp,training) for r in rows):
        fail=ROOT/'audits/cache_failure.json'
        if fail.exists():raise RuntimeError(('CACHE_PRODUCTION_FAILURE',load(fail)))
        time.sleep(.5)
    return time.time()-begin

def positions():
    out={}
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';a=load(p)['committed_updates'] if p.exists() else 0
        out[g]=a*4
    if (ROOT/'training_completion.json').exists():return {g:1200 for g in GROUPS}
    return out

def inventory():return sum(p.stat().st_size for p in (ROOT/'cache/derived').glob('*.pt'))

def evict_training(rows,positions):
    safe=min(positions.values())
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
    safe=min(pos.values());tasks=[]
    for p in sorted((ROOT/'cache/requests').glob('*.json'),key=lambda p:p.stat().st_mtime):
        a=load(p)
        if a['training'] and a['row'].get('plan_slot',1200)<safe:p.unlink();continue
        if not ready(a['row'],a['qp'],a['training']):tasks.append(a)
    # Preflight only asks for first8. Once formal training starts, keep a finite
    # lookahead at the slowest committed cursor without an all-cache barrier.
    if (ROOT/'gradient_calibration.json').exists() and (ROOT/'initialization_audit.json').exists():
        for i in range(safe,min(1200,safe+cfg['training_prefetch_windows'])):
            r=rows[i]
            if not ready(r,r['external_qp'],True):tasks.append(dict(row=r,qp=r['external_qp'],training=True))
    return tasks

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
                    dt=time.time()-j['start'];produced+=1;latency+=dt;event('shared_cache_READY',key=key,training=a['training'],seconds=dt,physical_GPU=j['gpu'],total_GPU_peak_MiB=j['peak'],PID=j['p'].pid);j['log'].close();active.pop(key)
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
                        gpu=free.pop(0);file=ROOT/'tmp/cache_jobs'/f'{key}.json';dump(file,a)
                        e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MODEL_KIND='original');log=(ROOT/'logs'/f'cache_GPU{gpu}.log').open('ab')
                        p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'cache_job.py'),str(file)],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                        active[key]=dict(p=p,gpu=gpu,job=a,start=time.time(),peak=0,utilization=0,reserve=estimate,log=log)
                elapsed=time.time()-begin
                dump(ROOT/'cache/progress.json',dict(status='DRAINING' if stopping else 'BACKPRESSURE' if backpressure else 'RUNNING',READY_training_windows=sum(ready(r,r['external_qp'],True) for r in rows),cache_bytes=inventory(),positions=pos,next_windows={g:rows[n]['sample_key'] if n<1200 else None for g,n in pos.items()},produced=produced,windows_per_second=produced/elapsed if elapsed else None,producer_latency_sum=latency,active={k:dict(PID=j['p'].pid,physical_GPU=j['gpu'],peak_MiB=j['peak'],GPU_utilization=j['utilization']) for k,j in active.items()},time=time.time()))
                if stopping and not active:break
                time.sleep(1)
        except Exception as error:
            dump(ROOT/'audits/cache_failure.json',dict(status='FAILED',exception=repr(error),traceback=traceback.format_exc(),keys=list(active),time=time.time()))
            for j in active.values():j['p'].wait();j['log'].close()
            raise
if __name__=='__main__':produce()
