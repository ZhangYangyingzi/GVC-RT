"""Persistent single-instance supervisor; atomic commits are the only sampler cursor.

Usage: runner.py start | status | resume. start/resume detach a supervisor.
All children are task-private and GPU visibility is always one authorized card.
"""
from common_s2 import *
import argparse,traceback,signal


def launch(args,gpu,tag):
    assert gpu in range(4)
    # A supervisor restart adopts matching surviving workers rather than
    # launching a second process. PID, exact argv and cwd all have to agree.
    markers={'calibrate.py':'gradient_calibration.json','preflight.py':'audits/preflight.json','evaluate.py':'evaluation/completion.json','videos.py':'video_manifest.json'}
    if args[0] in markers or args[0]=='train.py':
        for saved in (ROOT/'receipts/processes').glob('*.json'):
            record=load(saved);pid=record.get('PID');proc=Path(f'/proc/{pid}/cmdline')
            if record.get('args')!=args or not proc.exists():continue
            actual=[s.decode() for s in proc.read_bytes().split(b'\0') if s]
            if actual!=[PYTHON,'-u',*args] or Path(f'/proc/{pid}/cwd').resolve()!=ROOT:continue
            class AdoptedWorker:
                def __init__(self):self.pid=pid;self.returncode=None
                def poll(self):
                    stat=Path(f'/proc/{self.pid}/stat')
                    if stat.exists() and stat.read_text().split()[2]!='Z':return None
                    if args[0]=='train.py':
                        g=args[2];failure=ROOT/g/'training/failure.json'
                        self.returncode=1 if failure.exists() and load(failure)['time']>=record['start_time'] else (0 if progress(g)['committed_updates']==500 else 6)
                    else:
                        target=markers[args[0]]
                        if args[0]=='evaluate.py' and '--through' in args and args[args.index('--through')+1]=='250':target='evaluation/through_250.json'
                        marker=ROOT/target;self.returncode=0 if marker.exists() and load(marker).get('status')=='PASS' else 1
                    return self.returncode
            event('worker_adopted_after_supervisor_resume',PID=pid,args=args,physical_GPU=gpu)
            return AdoptedWorker()
    out=(ROOT/'logs'/f'{tag}.log').open('a')
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if args[0]=='train.py' and (ROOT/args[2]/'training/failure.json').exists():
        if load(ROOT/args[2]/'training/failure.json').get('OOM'):e['GVC_GRAD_CHECKPOINT']='1'
    p=subprocess.Popen([PYTHON,'-u',*args],cwd=ROOT,env=e,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'receipts/processes'/f'{p.pid}.json',dict(PID=p.pid,args=args,physical_GPU=gpu,process_GPU=0,start_time=time.time(),log=str(out.name)))
    return p


def progress(g):
    p=ROOT/g/'training/progress.json'
    return load(p) if p.exists() else dict(committed_updates=0,next_cursor=0)


def snapshot(stage,children=None,state='RUNNING',error=None):
    result=dict(status=state,stage=stage,experiment_directory=str(ROOT),PID=os.getpid(),heartbeat=time.time(),experiment_complete=False,committed_steps={g:progress(g)['committed_updates'] for g in GROUPS},progress={g:progress(g) for g in GROUPS},children=[] if children is None else [dict(PID=p.pid,running=p.poll() is None) for p in children],error=error)
    result['evaluation']={g:{str(s):len(list((ROOT/'evaluation'/g/f'step_{s:04d}/points').glob('*.json'))) for s in [0,250,500]} for g in GROUPS}
    result['ETA_seconds']={g:(500-progress(g)['committed_updates'])*progress(g).get('seconds',0) or None for g in GROUPS}
    if state=='COMPLETE':result['experiment_complete']=True
    dump(ROOT/'status.json',result)
    return result


def wait_stage(p,stage,others=()):
    while p.poll() is None:
        snapshot(stage,[p,*others]);time.sleep(5)
    if p.returncode:raise RuntimeError(f'{stage} exited {p.returncode}; see logs and audits')


def cache_request(items,tag,gpu=3):
    path=ROOT/'cache/requests'/f'{tag}.json';dump(path,items)
    for receipt in (ROOT/'receipts/processes').glob('*.json'):
        record=load(receipt);args=record.get('args',[])
        if len(args)!=2 or args[0]!='cache_job.py' or record.get('physical_GPU')!=gpu:continue
        prior_request=Path(args[1])
        if not prior_request.is_relative_to(ROOT/'cache/requests') or not prior_request.exists():continue
        pid=record['PID'];proc=Path(f'/proc/{pid}/cmdline')
        try:
            actual=[v.decode() for v in proc.read_bytes().split(b'\0') if v]
            alive=actual==[PYTHON,'-u',*args] and Path(f'/proc/{pid}/cwd').resolve()==ROOT
        except FileNotFoundError:alive=False
        if not alive:continue
        prior_items=load(prior_request)
        if not all(a['training'] for a in prior_items) or not all(a['training'] for a in items):
            if args!=['cache_job.py',str(path)]:continue
        class Adopted:
            def __init__(self):self.pid=pid;self.returncode=None
            def poll(self):
                stat=Path(f'/proc/{self.pid}/stat')
                try:
                    if stat.exists() and stat.read_text().split()[2]!='Z':return None
                except FileNotFoundError:pass
                from cache_codec import cache_paths
                # Successfully consumed tensors may already be evicted. Their
                # atomic immutable receipts remain the worker completion proof.
                receipts=[cache_paths(a['row'],a['qp'],a['training'])[2] for a in prior_items]
                self.returncode=0 if all(r.exists() and load(r).get('status')=='PASS' for r in receipts) else 1
                return self.returncode
        event('cache_worker_adopted',PID=pid,existing_request=str(prior_request),new_cursor_request=str(path))
        return Adopted()
    return launch(['cache_job.py',str(path)],gpu,tag)


def cached(row):
    from cache_codec import cache_paths
    _,p,r=cache_paths(row,row['external_qp'],True)
    return p.exists() and r.exists()


def supervise():
    env()
    with lock('locks/runner.lock'):
        dump(ROOT/'runner.pid.json',dict(PID=os.getpid(),start=time.time(),command=sys.argv,identity=load(ROOT/'identity.json')))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used','--format=csv,noheader'],text=True)
        dump(ROOT/'audits/GPU_mapping.json',dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',physical_records=gpu,group_mapping={g:dict(physical_index=i,process_index=0,visible_devices=str(i)) for i,g in enumerate(GROUPS)},only_compute_GPUs=[0,1,2,3]))
        from prepare import main
        main();rows=plan_rows();cal=load(ROOT/'plans/calibration.json.gz')
        if not (ROOT/'audits/preflight.json').exists():
            p=cache_request([dict(row=r,qp=r['external_qp'],training=True) for r in rows[:16]],'startup_16')
            wait_stage(p,'startup_cache_16')
            p=launch(['preflight.py'],0,'preflight');wait_stage(p,'mechanical_preflight')
        assert load(ROOT/'audits/preflight.json')['status']=='PASS'
        if not (ROOT/'gradient_calibration.json').exists():
            p=cache_request([dict(row=r,qp=r['external_qp'],training=True) for r in cal],'calibration_cache')
            wait_stage(p,'calibration_cache')
            p=launch(['calibrate.py'],0,'GAN_calibration');wait_stage(p,'GAN_warmup_and_gradient_calibration')
        assert load(ROOT/'gradient_calibration.json')['status']=='PASS'
        verify_sources()
        producer=None;trainers={};attempts={g:0 for g in GROUPS};last_publish=0;failed=set();cache_attempts=0;mid=None;mid_attempted=(ROOT/'evaluation/through_250.json').exists()
        while any(progress(g)['committed_updates']<500 for g in GROUPS if g not in failed):
            # Never keep a CUDA model resident waiting for unavailable data.
            for g,p in list(trainers.items()):
                if p.poll() is not None:
                    del trainers[g]
                    if p.returncode not in [0,6,7]:
                        attempts[g]+=1
                        fail=load(ROOT/g/'training/failure.json') if (ROOT/g/'training/failure.json').exists() else {}
                        if not fail.get('OOM') or attempts[g]>4:
                            failed.add(g);dump(ROOT/g/'stopped.json',dict(status='STOPPED_ENGINEERING_FAULT',failure=fail,time=time.time()))
            if not mid_attempted and not failed and all(progress(g)['committed_updates']>=250 for g in GROUPS):
                mid=launch(['evaluate.py','--through','250'],0,'development_through250');mid_attempted=True
            if mid is not None and mid.poll() is not None:
                if mid.returncode:dump(ROOT/'audits/mid_evaluation_failure.json',dict(status='FAILED_EVALUATION_ONLY',returncode=mid.returncode,time=time.time()))
                mid=None
            active=[g for g in GROUPS if g not in failed and progress(g)['committed_updates']<500]
            if not active:break
            slow=min(progress(g)['next_cursor'] for g in active)
            stop=min(2000,slow+48)
            if producer is None or producer.poll() is not None:
                if producer is not None and producer.returncode:
                    failure=load(ROOT/'audits/cache_failures'/f'{producer.pid}.json')
                    if 'out of memory' in failure['error'].lower() and cache_attempts<4:
                        cache_attempts+=1
                        dump(ROOT/'J3/yield_for_cache.json',dict(reason='producer OOM; release GPU at next committed update boundary',attempt=cache_attempts,time=time.time()))
                        if 'J3' in trainers:
                            snapshot('waiting_for_J3_to_release_GPU_for_producer',list(trainers.values()));time.sleep(5);continue
                        producer=None
                    else:raise RuntimeError('cache producer fault; inspect cache_failures')
                missing=[r for r in rows[slow:stop] if not cached(r)]
                if missing:producer=cache_request([dict(row=r,qp=r['external_qp'],training=True) for r in missing],f'prefetch_{slow:04d}',3)
                elif (ROOT/'J3/yield_for_cache.json').exists():(ROOT/'J3/yield_for_cache.json').unlink()
            for i,g in enumerate(GROUPS):
                if g in failed:continue
                cursor=progress(g)['next_cursor']
                if g not in trainers and cursor<2000 and all(cached(r) for r in rows[cursor:cursor+4]):
                    if i==0 and not failed and progress(g)['committed_updates']>=250 and (not mid_attempted or mid is not None):continue
                    if i==3:
                        if (ROOT/'J3/yield_for_cache.json').exists():continue
                        if producer is not None and producer.poll() is None:
                            available=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                            if available<10000:continue
                    trainers[g]=launch(['train.py','--group',g,'--until','250' if progress(g)['committed_updates']<250 else '500'],i,f'train_{g}')
            # A slow active consumer pins every unconsumed training cache.
            keep={r['sample_key'] for r in rows[max(0,slow-4):stop]}
            from cache_codec import cache_paths
            for r in rows[:max(0,slow-4)]:
                _,path,receipt=cache_paths(r,r['external_qp'],True)
                if path.is_relative_to(ROOT) and path.exists():
                    with lock('locks/lease_'+path.stem+'.lock',blocking=True):path.unlink()
            snapshot('training_and_shared_prefetch',[*trainers.values(),*([producer] if producer else []),*([mid] if mid else [])])
            if time.time()-last_publish>900:
                subprocess.run([PYTHON,'publish.py','--interim'],cwd=ROOT,check=False);last_publish=time.time()
            time.sleep(5)
        if producer is not None:wait_stage(producer,'last_prefetch')
        if mid is not None:wait_stage(mid,'development_through250_finishing')
        if failed:
            subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)
            snapshot('partial_training_fault',state='FAILED_PARTIAL',error='stopped groups: '+','.join(sorted(failed)))
            subprocess.run([PYTHON,'publish.py','--interim'],cwd=ROOT,check=False)
            return
        # Independent evaluation uses released training GPUs. No final-test path exists.
        p=launch(['evaluate.py'],0,'development_evaluation');wait_stage(p,'development_96_points_0_250_500')
        p=launch(['videos.py'],0,'lossless_videos');wait_stage(p,'fixed_development_FFV1')
        subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)
        snapshot('delivery',state='RUNNING')
        subprocess.run([PYTHON,'publish.py'],cwd=ROOT,check=True)
        snapshot('complete',state='COMPLETE')
        subprocess.run([PYTHON,'publish.py'],cwd=ROOT,check=True)


def start():
    env()
    try:
        with lock('locks/runner.lock'):pass
    except BlockingIOError:
        print(json.dumps(load(ROOT/'status.json'),indent=2));return
    out=(ROOT/'logs/runner.log').open('a')
    p=subprocess.Popen([PYTHON,'-u',str(ROOT/'runner.py'),'supervise'],cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(status='STARTED',PID=p.pid,experiment_directory=str(ROOT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','resume','status','supervise']);a=parser.parse_args()
    if a.action=='status':print(json.dumps(load(ROOT/'status.json'),indent=2))
    elif a.action in ['start','resume']:start()
    else:
        try:supervise()
        except Exception as error:
            env();dump(ROOT/'runner_stop_request.json',dict(reason='supervisor fault',error=repr(error),time=time.time()));dump(ROOT/'audits/runner_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()));snapshot('fault',state='FAILED',error=repr(error));raise
