"""Dependency-driven overlap; adopt live workers instead of interrupting their evaluations."""
import fcntl,traceback
from v67b_io import *
def process_identity(pid):
    try:
        proc=Path('/proc')/str(pid);fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return dict(start=fields[19],cmd=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None
def tasks():
    out={}
    for d in DATASETS:
        for v in sources(d):
            for m in FRESH:
                k=f'eval_{d}_{m}_{v["video_index"]}'
                out[k]=dict(kind='eval',script='evaluate.py',args=['--dataset',d,'--method',m,'--video',str(v['video_index'])],gpu=True,memory=16000,dataset=d,method=m,video=v['video_index'])
    if load(ROOT/'audits/vimeo_heldout_manifest.json')['status']=='AVAILABLE':
        for m in B_METHODS:out[f'heldout_{m}']=dict(kind='heldout',script='heldout.py',args=['--method',m],gpu=True,memory=6000,method=m)
    for d in DATASETS:
        for m in METHODS:out[f'fid_{d}_{m}']=dict(kind='fid',script='fid_worker.py',args=['--dataset',d,'--method',m],gpu=False,dataset=d,method=m)
    return out
def complete(job):
    if job['kind']=='eval':return all(point(job['dataset'],job['method'],job['video'],q).exists() for q in range(10))
    if job['kind']=='fid':return (ROOT/'parts'/f'fid_done_{job["dataset"]}_{job["method"]}.json').exists()
    return (ROOT/'parts'/f'heldout_done_{job["method"]}.json').exists()
def ready(job):
    if job['kind']!='fid':return True
    return all(point(job['dataset'],job['method'],v['video_index'],q).exists() for v in sources(job['dataset']) for q in range(10))
def run(lock=None):
    if lock is None:
        lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
    joblist=tasks();active={};events=[];started=time.time();max_gpu_jobs=2;max_cpu_jobs=4
    for key,info in prior.get('active',{}).items():
        if key not in joblist:continue
        identity=process_identity(info['PID'])
        if identity and str(ROOT/joblist[key]['script']) in identity['cmd']:
            active[key]=dict(pid=info['PID'],gpu=info.get('gpu'),start=identity['start'],proc=None,log=None,adopted=True)
            events.append(dict(event='adopt',task=key,PID=info['PID'],gpu=info.get('gpu'),unix=time.time()))
    def status(phase='parallel_evaluation_fid_heldout',**extra):
        count={m:sum(len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for d in DATASETS) for m in METHODS}
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'FAIL' if phase=='failed' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),method_points=count,method_target=310,raw_RD_points=sum(count.values()),FID_points=len(list((ROOT/'parts/fid').glob('*/*/qp*.json'))),FID_target=200,heldout_points=len(list((ROOT/'parts/heldout').glob('*/*.json'))),max_jobs_per_gpu=max_gpu_jobs,max_CPU_FID_jobs=max_cpu_jobs,active={k:dict(PID=j['pid'],gpu=j['gpu'],adopted=j['adopted'],kind=joblist[k]['kind'],log=str(ROOT/'logs'/f'{k}.log')) for k,j in active.items()},pending=sum(not complete(j) and k not in active for k,j in joblist.items()),**extra))
        dump(ROOT/'audits/parallel_execution_events.json',dict(controller_PID=os.getpid(),started_unix=started,events=events))
    def start(key,gpu):
        j=joblist[key];path=ROOT/'logs'/f'{key}.log';log=path.open('a');args=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args']]
        if gpu is not None:args+=['--gpu',str(gpu)]
        proc=subprocess.Popen(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        identity=process_identity(proc.pid)
        active[key]=dict(pid=proc.pid,gpu=gpu,start=identity['start'] if identity else None,proc=proc,log=log,adopted=False)
        events.append(dict(event='start',task=key,PID=proc.pid,gpu=gpu,unix=time.time()));print('STARTED',key,'GPU',gpu,'PID',proc.pid,flush=True)
    try:
        frozen();assert load(ROOT/'audits/real_RANS_smoke.json')['status']=='PASS';assert load(ROOT/'audits/fid_smoke.json')['status']=='PASS'
        status()
        while True:
            for key,j in list(active.items()):
                if j['proc'] is not None:
                    rc=j['proc'].poll()
                    if rc is None:continue
                    assert rc==0,f'{key} exited {rc}; see its log'
                else:
                    identity=process_identity(j['pid'])
                    if identity and identity['start']==j['start']:continue
                    rc='adopted_process_exited'
                assert complete(joblist[key]),f'{key} exited with missing outputs'
                if j['log']:j['log'].close()
                del active[key];events.append(dict(event='finish',task=key,PID=j['pid'],returncode=rc,unix=time.time()));print('FINISHED',key,flush=True)
            pending=[k for k,j in joblist.items() if k not in active and not complete(j)]
            if not pending and not active:break
            cpu_used=sum(j['gpu'] is None for j in active.values())
            for key in pending:
                job=joblist[key]
                if not job['gpu'] and ready(job) and cpu_used<max_cpu_jobs:start(key,None);cpu_used+=1
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(a):int(b) for a,b in (line.split(',') for line in out.splitlines()) if int(a) in (4,5,6,7)}
            used={g:sum(j['gpu']==g for j in active.values()) for g in free}
            # Small independent held-out jobs fill additional GPU slots first.
            gpu_pending=sorted((k for k in pending if joblist[k]['gpu']),key=lambda k:joblist[k]['kind']!='heldout')
            for key in gpu_pending:
                eligible=[g for g in free if used[g]<max_gpu_jobs and free[g]>=joblist[key]['memory']]
                if not eligible:continue
                card=max(eligible,key=lambda g:free[g]);start(key,card);used[card]+=1
                # Reserve launch headroom immediately, before CUDA allocations appear in nvidia-smi.
                free[card]-=joblist[key]['memory']
            status();time.sleep(10)
        status('report')
        with (ROOT/'logs/report.log').open('a') as log:
            proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'report.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:status('report');time.sleep(5)
            assert proc.returncode==0,'Report failed; see logs/report.log'
        assert load(ROOT/'final_integrity.json')['status']=='PASS';status('complete');print('PARALLEL PIPELINE PASS',flush=True)
    except Exception:
        error=traceback.format_exc();status('failed',traceback=error);dump(ROOT/'pipeline_failure.json',dict(status='FAIL',traceback=error,unix=time.time()));print(error,flush=True)
        # Do not signal any worker. A subsequent launch can adopt survivors.
        raise
