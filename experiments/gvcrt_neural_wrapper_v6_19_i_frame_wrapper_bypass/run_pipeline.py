"""Memory-aware resumable pilot gate, 480 points, numerical outputs and publication."""
import argparse,fcntl,traceback
from io19 import *
def identity(pid):
    try:
        a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split();return None if a[0]=='Z' else a[19]
    except (FileNotFoundError,ProcessLookupError):return None
def main():
    os.chdir(REPO);os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');(ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);command([PYTHON,'-B','-u',str(Path(__file__))])
    def cpu(script):
        argv=[PYTHON,'-B','-u',str(ROOT/script)];command(argv)
        with (ROOT/'logs'/(Path(script).stem+'.log')).open('a') as f:subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (ROOT/'audits/baseline_reuse.json').exists():cpu('prepare.py')
    frozen();from adapter import validate
    from prepare import preflight_video
    jobs={}
    for d in ('ulong','uvg_holdout'):
        v=next(v for v in sources(d) if preflight_video(v))
        for m in METHODS:jobs[f'pilot_{d}_{m}']=dict(dataset=d,method=m,video=v['video_index'],qps=[0,9],pilot=True)
    for d in DATASETS:
        for m in METHODS:
            for v in sources(d):jobs[f'final_{d}_{m}_{v["video_index"]}']=dict(dataset=d,method=m,video=v['video_index'],qps=list(range(10)),pilot=False)
    prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};events=prior.get('events',[]);attempts=prior.get('attempts',{});active={};peak=prior.get('peak_memory_MiB',0);concurrency=prior.get('max_concurrency',4);verified={}
    def done(j):
        v=next(v for v in sources(j['dataset']) if v['video_index']==j['video'])
        for q in j['qps']:
            p=point(j['dataset'],j['method'],j['video'],q)
            if not p.exists():return False
            stamp=(p.stat().st_size,p.stat().st_mtime_ns)
            if verified.get(str(p))!=stamp:validate(load(p),v,q,j['method']);verified[str(p)]=stamp
        return True
    for k,w in prior.get('active',{}).items():
        if identity(w['PID'])==w['start_identity']:active[k]=dict(**w,proc=None,stream=None)
    gpu_start=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
    def status(phase):
        completed=sum(point(d,m,v['video_index'],q).exists() for d in DATASETS for m in METHODS for v in sources(d) for q in range(10));s=dict(status='RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),expected_points=480,completed_points=completed,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},events=events,attempts=attempts,peak_memory_MiB=peak,max_concurrency=concurrency);dump(ROOT/'pipeline_status.json',s);dump(ROOT/'gpu_schedule.json',dict(initial_GPU_state=gpu_start,allowed_GPUs=[4,5,6,7],events=events,peak_memory_MiB=peak,max_concurrency=concurrency,other_processes_untouched=True))
    gate=ROOT/'audits/implementation_check.json'
    while True:
        for k,w in list(active.items()):
            rc=w['proc'].poll() if w['proc'] is not None else None if identity(w['PID'])==w['start_identity'] else 0 if done(jobs[k]) else -1
            if rc is None:continue
            if w['stream']:w['stream'].close()
            del active[k];events.append(dict(event='finish',task=k,gpu=w['gpu'],returncode=rc,unix=time.time()));print('FINISH',k,rc,flush=True)
            if rc or not done(jobs[k]):
                with open(w['log']) as f:f.seek(w.get('log_offset',0));err=f.read()
                if 'out of memory' in err.lower() and attempts.get(k,0)<8:concurrency=max(1,concurrency-1);events.append(dict(event='OOM_RETRY',task=k,gpu=w['gpu'],unix=time.time()));continue
                raise RuntimeError(k+' failed\n'+err[-8000:])
        if not gate.exists() and all(done(j) for j in jobs.values() if j['pilot']):
            checks=[]
            for d in ('ulong','uvg_holdout'):
                v=next(v for v in sources(d) if preflight_video(v))
                for q in (0,9):
                    rs={m:load(point(d,m,v['video_index'],q)) for m in METHODS};a=rs['original']['first_frame_audit'];b=rs['B_i_bypass']['first_frame_audit']
                    for key in ('payload_sha256','payload_bytes','decoded_tensor_sha256','encoder_initial_reference_sha256','decoder_initial_reference_sha256','independent_initial_reference_sha256'):assert a[key]==b[key],('I bypass/original mismatch',d,q,key)
                    assert rs['B_standard']['historical_reproduction_verified'] and rs['original']['historical_reproduction_verified'];assert rs['B_standard']['runtime_audit']==rs['B_i_bypass']['runtime_audit'];checks.append(dict(dataset=d,sequence=v['name'],QP=q,I_payload_and_initial_reference_exact=True,standard_historical_reproduction=True,weights_exact=True,independent_decode=True,models_frozen=True))
            dump(gate,dict(status='PASS',points=12,checks=checks));print('IMPLEMENTATION PASS',flush=True)
        for d in DATASETS:
            for m in METHODS:
                for v in sources(d):
                    for q in range(10):
                        p=point(d,m,v['video_index'],q)
                        if p.exists():peak=max(peak,load(p).get('peak_memory_MiB',0))
        if not active and all(done(j) for j in jobs.values()):break
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(i):int(f) for i,f in (x.split(',') for x in lines) if int(i) in (4,5,6,7)};available=[g for g in free if g not in {w['gpu'] for w in active.values()}]
        for k,j in jobs.items():
            if k in active or done(j) or (not j['pilot'] and not gate.exists()):continue
            if any((w['dataset'],w['method'],w['video'])==(j['dataset'],j['method'],j['video']) for name in active for w in (jobs[name],)):continue
            required=max(10000,int(peak*1.3+2048));gpus=[g for g in available if free[g]>=required]
            if not gpus or len(active)>=concurrency:continue
            gpu=max(gpus,key=lambda g:free[g]);available.remove(gpu);argv=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--dataset',j['dataset'],'--method',j['method'],'--video',str(j['video']),'--qps',','.join(map(str,j['qps'])),'--gpu',str(gpu)];command(argv);log=ROOT/'logs'/f'{k}.log';offset=log.stat().st_size if log.exists() else 0;stream=log.open('a');proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);active[k]=dict(PID=proc.pid,start_identity=identity(proc.pid),gpu=gpu,log=str(log),log_offset=offset,proc=proc,stream=stream);attempts[k]=attempts.get(k,0)+1;events.append(dict(event='start',task=k,gpu=gpu,PID=proc.pid,unix=time.time()));print('START',k,'GPU',gpu,flush=True)
        status('evaluation' if gate.exists() else 'implementation_verification');time.sleep(10)
    status('aggregate');cpu('report.py');assert load(ROOT/'final_integrity.json')['status']=='PASS';status('publication');cpu('publish_github.py');pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS';s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',active={},publication=pub);dump(ROOT/'pipeline_status.json',s);print('PIPELINE COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'pipeline_status.json',dict(status='FAIL',error=traceback.format_exc()));dump(ROOT/'blocking_error.json',dict(status='FAIL',error=traceback.format_exc()));raise
