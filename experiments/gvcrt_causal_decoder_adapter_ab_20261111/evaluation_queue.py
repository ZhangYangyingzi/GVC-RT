"""Bounded development queue with training cache priority and task-owned workers."""
from common_s2 import *
import traceback


def states():
    out=[('B300',0),('Original',0)]
    for g in GROUPS:
        for s in (50,100):
            if (ROOT/g/f'training/checkpoints/step_{s:04d}.pt').exists():out.append((g,s))
    return out


def cursor():
    out=[]
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';out.append(load(p)['next_cursor'] if p.exists() else 0)
    return min(out)


def worker():
    env();from cache_h import ready
    from cache_codec import cache_paths
    from evaluation import point_path
    active={};videos=load(ROOT/'manifests/development_videos.json');rows=plan_rows();cooldowns={}
    with lock('locks/evaluation_queue.lock'):
        while not (ROOT/'queue/STOP').exists():
            for key,j in list(active.items()):
                code=j['p'].poll()
                if code is not None:
                    j['log'].close()
                    if code:
                        failures=list((ROOT/'audits/evaluation_failures').glob(f'{j["spec"].stem}_*.json'))
                        retryable=failures and load(failures[-1]).get('retryable_OOM')
                        if not retryable:raise RuntimeError(('EVALUATION_WORKER_FAILED',str(j['spec']),code))
                        cooldowns[j['gpu']]=time.time()+60
                    del active[key]
            available_states=states();pending=[];coverage={}
            for m,s in available_states:coverage[f'{m}_step{s}']=0
            for v in videos:
                for q in (0,5,9):
                    tasks=[]
                    for m,s in available_states:
                        if point_path(v,q,m,s).exists():coverage[f'{m}_step{s}']+=1
                        else:tasks.append(dict(model=m,step=s))
                    if tasks:
                        key=v['sequence']+f'_qp{q}'
                        if key not in active:
                            _,p,rp=cache_paths(v.get('source_row',v),q,False)
                            pending.append(dict(key=key,video=v,QP=q,tasks=tasks,cache_exists=p.exists() and rp.exists()))
            pending.sort(key=lambda b:not b['cache_exists'])
            pos=cursor();training_ready=pos>=400 or all(ready(r) for r in rows[pos:min(400,pos+4)])
            gpus=[2,3]
            for g in GROUPS:
                if (ROOT/g/'training/completion.json').exists():gpus.append(GROUPS.index(g))
            training_done=all((ROOT/g/'training/completion.json').exists() for g in GROUPS)
            config=load(ROOT/'queue/evaluation_config.json') if (ROOT/'queue/evaluation_config.json').exists() else {}
            limits={g:1 for g in gpus}
            if training_done:
                limits.update({int(g):n for g,n in config.get('after_training_workers_per_GPU',{'0':1,'1':1,'2':1,'3':1}).items()})
                gpus=gpus+[g for g in gpus for _ in range(max(0,limits[g]-1))]
            if training_ready:
                gpuinfo=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                free={int(line.split(',')[0]):int(line.split(',')[1]) for line in gpuinfo.strip().splitlines()}
                for gpu in gpus:
                    if sum(j['gpu']==gpu for j in active.values())>=limits[gpu] or time.time()<cooldowns.get(gpu,0):continue
                    if not pending:break
                    eligible=[b for b in pending if (b['cache_exists'] or not any(j['gpu']==gpu and j.get('creating_native_cache') for j in active.values()))
                        and free[gpu]>=(9000 if b['cache_exists'] else 18000)]
                    if not eligible:continue
                    bundle=eligible[0];pending.remove(bundle);free[gpu]-=9000 if bundle['cache_exists'] else 18000
                    spec=ROOT/'tmp/evaluation_jobs'/f'{bundle["key"]}_{time.time_ns()}.json';dump(spec,bundle)
                    log=(ROOT/f'logs/evaluation_GPU{gpu}.log').open('ab')
                    child=subprocess.Popen([PYTHON,'-B','-u','evaluation_worker.py',str(spec)],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[bundle['key']]=dict(p=child,gpu=gpu,log=log,spec=spec,start=time.time(),creating_native_cache=not bundle['cache_exists'])
            dump(ROOT/'queue/evaluation_progress.json',dict(status='RUNNING',coverage=coverage,
                active={k:dict(PID=j['p'].pid,physical_GPU=j['gpu'],spec=str(j['spec']),seconds=time.time()-j['start']) for k,j in active.items()},
                bounded_workers=sum(limits.values()),workers_per_GPU=limits,training_cache_priority=True,training_ready_next_batch=training_ready,training_cursor=pos,time=time.time()))
            if len(available_states)==6 and all(n==96 for n in coverage.values()) and not active:
                dump(ROOT/'evaluation_completion.json',dict(status='PASS',coverage=coverage,time=time.time()));break
            time.sleep(5)


if __name__=='__main__':
    try:worker()
    except Exception as e:env();dump(ROOT/'audits/evaluation_queue_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
