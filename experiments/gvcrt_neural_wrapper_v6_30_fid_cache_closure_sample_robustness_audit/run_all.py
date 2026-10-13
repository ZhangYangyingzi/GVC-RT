"""One detached owner: preflight -> closure -> new clips -> diagnostics -> publish."""
from audit_io import *
import argparse, fcntl, itertools, subprocess, threading, traceback
STATE=dict(status='RUNNING',phase='starting',completed=0,total=0,running=0,recent_progress='starting',failure_count=0,PID=os.getpid(),session=os.environ.get('V630_SESSION','gvcrt_v630'),started=time.time())
STOP=threading.Event();STATE_LOCK=threading.Lock()
def heartbeat():
    last_hour=0
    while not STOP.is_set():
        with STATE_LOCK:
            s=dict(STATE);s['heartbeat']=time.strftime('%Y-%m-%d %H:%M:%S %z')
        dump(ROOT/'status.json',s);textfile(ROOT/'heartbeat',s['heartbeat']+'\n')
        dump(ROOT/'pipeline_status.json',s)
        if time.time()-last_hour>=3600:
            textfile(ROOT/'hourly_summary.txt','\n'.join(f'{k}: {s[k]}' for k in ('status','phase','completed','total','running','failure_count','recent_progress','heartbeat'))+'\n');last_hour=time.time()
        STOP.wait(60)
def setstate(**values):
    with STATE_LOCK:STATE.update(values)
    dump(ROOT/'status.json',dict(STATE,heartbeat=time.strftime('%Y-%m-%d %H:%M:%S %z')))
def proc_token(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except OSError:return None
def available_RAM():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
def GPUs():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
    return {int(a):float(b) for a,b in (x.split(',') for x in out.splitlines()) if int(a) in (4,5,6,7)}
def tasks(kind,vs,qps=QPS):
    return [dict(kind=kind,dataset=d,video=v['video_index'],method=m,qp=q) for d in DATASETS for v in vs[d] for m,q in itertools.product(METHODS,qps)]
def gts(vs):return [dict(kind='gt',dataset=d,video=v['video_index'],method='GT',qp=-1) for d in DATASETS for v in vs[d]]
def cpu_task(d,label,ids,kind):return dict(kind='cpu',dataset=d,video=label,method='pool',qp=-1,spec=dict(label=label,videos=ids,kind=kind))
class Queue:
    def __init__(self):
        self.peak_gpu=16000.;self.peak_ram=8192.;self.peak_cpu=1024.;self.concurrent=16;self.failures=[]
        if (ROOT/'audits/resource_measurements.json').exists():
            r=load(ROOT/'audits/resource_measurements.json');self.peak_gpu=r['GPU_peak_MiB'];self.peak_ram=r['GPU_worker_RAM_MiB'];self.peak_cpu=r['CPU_worker_RAM_MiB'];self.concurrent=r['max_concurrent_after_OOM']
    def measure(self,t):
        r=load(output(t))
        if t['kind'] in ('cpu','selftest'):self.peak_cpu=max(self.peak_cpu,r.get('peak_RSS_MiB',0))
        else:self.peak_gpu=max(self.peak_gpu,r.get('peak_memory_MiB',0));self.peak_ram=max(self.peak_ram,r.get('peak_RSS_MiB',0))
        dump(ROOT/'audits/resource_measurements.json',dict(GPU_peak_MiB=self.peak_gpu,GPU_worker_RAM_MiB=self.peak_ram,CPU_worker_RAM_MiB=self.peak_cpu,max_concurrent_after_OOM=self.concurrent,GPU_reservation_MiB=1.25*self.peak_gpu,per_card_spare_MiB=2048,RAM_margin_MiB=8192))
    def run(self,name,items,cpu=False,startup=False):
        setstate(phase=name,total=len(items),completed=0,running=0)
        pending=[t for t in items if not valid(t)];completed=len(items)-len(pending);active={};attempts={}
        jobsfile=ROOT/'audits'/f'queue_{name}.json'
        if jobsfile.exists():
            for key,r in load(jobsfile).items():
                if r['task'] in pending and proc_token(r['PID'])==r['start_token']:
                    active[key]=dict(r,process=None,fh=None);pending.remove(r['task'])
        def persist():dump(jobsfile,{k:{n:v for n,v in r.items() if n not in ('process','fh')} for k,r in active.items()})
        while pending or active:
            for key,r in list(active.items()):
                exitfile=Path(r['exit_path']);p=r['process']
                alive=(p.poll() is None) if p is not None else proc_token(r['PID'])==r['start_token']
                if alive:continue
                if r['fh']:r['fh'].close()
                code=load(exitfile)['exit_code'] if exitfile.exists() else p.returncode if p is not None else -999
                t=r['task'];del active[key]
                if code==0 and valid(t):
                    self.measure(t);completed+=1;setstate(recent_progress='completed '+task_id(t))
                else:
                    tail=Path(r['log']).read_text(errors='replace').splitlines()[-80:]
                    error=dict(task=t,exit_code=code,log=r['log'],last_80_lines=tail,attempt=r['attempt'])
                    append_error(error);dump(ROOT/'audits/errors'/f'{key}_{time.time_ns()}.json',error)
                    oom=any('out of memory' in x.lower() for x in tail)
                    if oom and r['attempt']==0:
                        attempts[key]=1;pending.append(t);self.concurrent=max(1,min(self.concurrent,len(active)+1)//2);self.peak_gpu=max(self.peak_gpu,r['reserved_gpu_MiB']/1.25);self.peak_ram=max(self.peak_ram,r['reserved_ram_MiB']/1.25)
                        setstate(recent_progress='OOM retry queued, reduced concurrency: '+key)
                    else:self.failures.append(error)
                    setstate(failure_count=len(self.failures))
                persist()
            free={} if cpu else GPUs();ram=available_RAM()
            # Reservations account for workers not yet at their loading/compute peak.
            for r in active.values():
                ram-=r['reserved_ram_MiB']
                if not cpu:free[r['GPU']]=free.get(r['GPU'],0)-r['reserved_gpu_MiB']
            for t in list(pending):
                cap=1 if startup else (min(4,self.concurrent) if cpu else self.concurrent)
                if len(active)>=cap:break
                rg=0 if cpu else self.peak_gpu*1.25;rr=(self.peak_cpu if cpu else self.peak_ram)*1.25
                if ram<rr+8192:break
                gpu=None
                if not cpu:
                    good=[g for g in (4,5,6,7) if free.get(g,0)>=rg+2048]
                    if not good:break
                    gpu=max(good,key=lambda g:free[g]);free[gpu]-=rg
                key=task_id(t);attempt=attempts.get(key,0);log=ROOT/'logs'/f'{name}_{key}_{time.time_ns()}.log';exitfile=ROOT/'audits/exits'/f'{key}_{time.time_ns()}.json';exitfile.parent.mkdir(exist_ok=True)
                cmd=[PYTHON,'-B','-u',str(ROOT/'task_entry.py'),'--task',json.dumps(t),'--exit',str(exitfile)]
                if gpu is not None:cmd+=['--gpu',str(gpu)]
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
                fh=log.open('w');p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=dict(task=t,PID=p.pid,start_token=proc_token(p.pid),GPU=gpu,reserved_gpu_MiB=rg,reserved_ram_MiB=rr,log=str(log),exit_path=str(exitfile),attempt=attempt,process=p,fh=fh);pending.remove(t);ram-=rr
                with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=cmd,PID=p.pid,GPU=gpu,time=time.time()))+'\n')
                persist()
            setstate(completed=completed,running=len(active))
            if pending or active:time.sleep(5)
        failed=[t for t in items if not valid(t)]
        dump(ROOT/'audits'/f'{name}_integrity.json',dict(expected=len(items),completed=completed,failed=failed,all_signatures_valid=not failed,computational_status='PASS' if not failed else 'FAIL'))
        return not failed
def publish():
    setstate(phase='publication',status='COMPUTATION_FINISHED')
    phase='final_integrity_verified' if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS' else 'execution_failure_findings'
    env=dict(os.environ,V630_PUBLICATION_PHASE=phase)
    with (ROOT/'logs/publication.log').open('a') as f:
        p=subprocess.run([PYTHON,'-B','-u',str(ROOT/'publish_github.py')],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode!=0:
        append_error(dict(stage='publication',exit_code=p.returncode,last_80_lines=(ROOT/'logs/publication.log').read_text(errors='replace').splitlines()[-80:]))
    return p.returncode==0
def main():
    for n in ('logs','audits','tasks','cache'): (ROOT/n).mkdir(exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('DONE.json','FAILED.json','BLOCKED.json'):
        p=ROOT/name
        if p.exists():
            folder=ROOT/'audits/archived_terminal_states';folder.mkdir(exist_ok=True);p.rename(folder/f'{time.time_ns()}_{name}')
    dump(ROOT/'launch.json',dict(PID=os.getpid(),start_token=proc_token(os.getpid()),session=STATE['session'],started=time.time(),cwd=str(REPO),script=str(ROOT/'run_all.py'),resume=f'{PYTHON} -B -u {ROOT}/run_all.py'))
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        setstate(phase='preflight',recent_progress='hashing frozen historical dependencies and selected reference files')
        from preflight import main as preflight
        preflight()
        from selftest30 import main as selftest
        selftest();setstate(failure_count=0)
        q=Queue();vs={d:sources(d) for d in DATASETS};starter={d:[vs[d][0]] for d in DATASETS}
        q.run('startup_GT',gts(starter),startup=True)
        q.run('startup_16',tasks('closure',starter,(0,9)),startup=True)
        measured=[load(output(t)) for t in tasks('closure',starter,(0,9)) if valid(t)]
        if measured:
            q.peak_gpu=max(r['peak_memory_MiB'] for r in measured);q.peak_ram=max(r['peak_RSS_MiB'] for r in measured)
            dump(ROOT/'audits/startup_peak_measurements.json',dict(points=len(measured),GPU_peak_MiB=q.peak_gpu,RAM_peak_MiB=q.peak_ram,includes_loading=True,GPU_safety_multiplier=1.25))
            q.measure(tasks('closure',starter,(0,9))[0])
        q.run('all_GT',gts(vs))
        closure_ok=q.run('closure_420',tasks('closure',vs))
        pooltasks=[cpu_task(d,d+'_old64',[v['video_index'] for v in vs[d]],'closure_pool') for d in DATASETS]
        pooltasks.append(cpu_task('mcljcv','mcl29_old64',[v['video_index'] for v in vs['mcljcv'] if v['name']!='videoSRC17_1920x1080_24'],'closure_pool'))
        if closure_ok:q.run('closure_pooled',pooltasks,cpu=True)
        from finalize30 import closure_report,finalize
        gate=closure_report()
        if not gate:
            dump(ROOT/'BLOCKED.json',dict(reason='stage1 closure or pooled comparison failed/incomplete; stages2/3 blocked',stage1_completed=sum(valid(t) for t in tasks('closure',vs)),expected=420,updates=0))
            finalize(blocked=True);publish();setstate(status='BLOCKED',phase='finished');return
        setstate(phase='prepare_new',recent_progress='extract fixed non-overlapping source segments, before quality evaluation')
        from prepare_new import main as prepare
        if not (ROOT/'audits/new_clip_preparation.json').exists():prepare()
        else:
            for d in DATASETS:
                for v in sources(d,True):
                    if v['status']=='AVAILABLE':
                        for i,h in enumerate(v['frame_rgb_sha256']):
                            name=f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png'
                            assert sha(Path(v['input_dir'])/name)==v['frame_png_sha256' if d=='hevc_b' else 'frame_file_sha256'][i]
        newvs={d:[v for v in sources(d,True) if v['status']=='AVAILABLE'] for d in DATASETS}
        newtasks=tasks('new',newvs)
        if newtasks:
            # Encoding loads two codec pairs. Re-measure with a conservative initial reservation.
            q.peak_gpu=max(q.peak_gpu,16000.);q.run('new_loading_probe',newtasks[:1],startup=True)
            if valid(newtasks[0]):q.peak_gpu=max(load(output(newtasks[0]))['peak_memory_MiB'],max((r['peak_memory_MiB'] for r in measured),default=0))
        q.run('new_clips',newtasks)
        statstasks=[]
        for d,label,exclude in (('hevc_b','hevc_b',False),('mcljcv','mcljcv',False),('mcljcv','mcl29',True)):
            ids=[v['video_index'] for v in newvs[d] if not exclude or v['name']!='videoSRC17_1920x1080_24']
            if not ids:continue
            if not all(valid(t) for t in tasks('new',{x:([v for v in newvs[x] if v['video_index'] in ids] if x==d else []) for x in DATASETS})):
                append_error(dict(stage='temporal_pools',dataset=d,reason='incomplete new points; do not silently drop failed videos'));continue
            for kind in ('matched_old64','new64','combined_two_clips'):statstasks.append(cpu_task(d,label+'_'+kind,ids,kind))
        for i,s in enumerate(load(ROOT/'subset_manifest.json')['subsets']):statstasks.append(cpu_task(s['dataset'],f'{s["dataset"]}_subset_{s["size"]}_{i:02d}',s['videos'],'subset'))
        q.run('sensitivity_statistics',statstasks,cpu=True)
        setstate(phase='aggregate_report_integrity',recent_progress='write numerical tables, six-section report and independent completion checks')
        integrity=finalize()
        if integrity['status']=='PASS':dump(ROOT/'DONE.json',dict(status='COMPUTATION_COMPLETE',updates=0,closure_points=420,new_points=integrity['new_completed'],report=str(ROOT/'REPORT.md'),publication='separately recorded',finished=time.time()))
        else:dump(ROOT/'FAILED.json',dict(status='FAIL',reason=integrity['missing'],completed=integrity,finished=time.time()))
        ok=publish();setstate(status='DONE' if integrity['status']=='PASS' else 'FAILED',phase='finished',running=0,recent_progress='publication '+('PASS' if ok else 'FAIL (recorded separately)'))
    except Exception:
        error=traceback.format_exc();append_error(dict(stage=STATE['phase'],traceback=error));dump(ROOT/'FAILED.json',dict(status='FAIL',stage=STATE['phase'],error=error,updates=0));setstate(status='FAILED',recent_progress=error.splitlines()[-1])
        if (ROOT/'final_integrity.json').exists():
            previous=load(ROOT/'final_integrity.json');previous.update(status='FAIL',terminal_error=error,calculation_complete=False);dump(ROOT/'final_integrity.json',previous)
        textfile(ROOT/'REPORT.md','# V6.30 execution failure\n\nNo updates.\n\n```text\n'+error+'\n```\n')
        if (ROOT/'config.json').exists():publish()
        setstate(status='FAILED',phase='finished',running=0)
        raise
    finally:
        STOP.set();dump(ROOT/'status.json',dict(STATE,heartbeat=time.strftime('%Y-%m-%d %H:%M:%S %z')));textfile(ROOT/'heartbeat',time.strftime('%Y-%m-%d %H:%M:%S %z')+'\n')
if __name__=='__main__':main()
