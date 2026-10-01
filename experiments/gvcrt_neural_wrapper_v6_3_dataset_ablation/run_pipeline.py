"""Detached, locked preflight -> smoke -> train -> checkpoint -> eval -> report."""
import argparse,fcntl,traceback
from v63_io import *
def gpu_memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (s.split(',') for s in out.splitlines()) if int(a) in (4,5,6,7)}
def held(p):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    for d in ('logs','parts'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    if a.status:
        status=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else dict(status='NOT_STARTED')
        status['branches']={b:load(ROOT/'branches'/b/'training_status.json') if (ROOT/'branches'/b/'training_status.json').exists() else dict(status='QUEUED') for b in BRANCHES}
        print(json.dumps(status,indent=2));return
    if a.launch:
        if held(ROOT/'parts/pipeline.lock'):print('ALREADY RUNNING');return
        with (ROOT/'logs/pipeline.log').open('a') as log:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
            child=subprocess.Popen([PYTHON,'-B','-u',str(Path(__file__).resolve())],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='LAUNCHED',pid=child.pid,log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failures={};events=[]
    def update(phase):
        branch_status={}
        for b in BRANCHES:
            path=ROOT/'branches'/b/'training_status.json'
            row=load(path) if path.exists() else dict(status='QUEUED',update=0,target=1000)
            for key,job in active.items():
                if key in ('train_'+b,'smoke_'+b):row.update(status='SMOKE' if key.startswith('smoke') else 'RUNNING',gpu=job['gpu'],pid=job['process'].pid)
            branch_status[b]=row
        total=sum(len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for d in DATASETS for m in METHODS)
        status=dict(status='RUNNING_WITH_FAILURES' if failures else 'RUNNING',phase=phase,pid=os.getpid(),updated_unix=time.time(),branches=branch_status,
            active_jobs={k:dict(pid=j['process'].pid,gpu=j['gpu'],log=j['log']) for k,j in active.items()},failures=failures,total_completed_points=total,expected_points=1550)
        dump(ROOT/'pipeline_status.json',status);dump(ROOT/'training_status.json',dict(status='PASS' if all(s['status']=='PASS' for s in branch_status.values()) else 'RUNNING',branches=branch_status))
    def start(key,script,args=(),gpu=None):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        path=ROOT/'logs'/f'{key}.log'
        with path.open('a') as log:proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        active[key]=dict(process=proc,gpu=gpu,log=str(path));events.append(dict(event='start',key=key,gpu=gpu,pid=proc.pid,unix=time.time()));print('START',key,'GPU',gpu,'PID',proc.pid,flush=True)
    def reap():
        for key,j in list(active.items()):
            rc=j['process'].poll()
            if rc is None:continue
            events.append(dict(event='exit',key=key,returncode=rc,unix=time.time()));print('EXIT',key,rc,flush=True)
            if rc:failures[key]=dict(returncode=rc,log=j['log'])
            del active[key]
        dump(ROOT/'audits/execution_events.json',dict(events=events,failures=failures))
    def cpu(key,script,args=()):
        start(key,script,args)
        while active:
            reap();update(key)
            if active:time.sleep(5)
        assert not failures,failures
    cpu('preflight','preflight.py')
    cfg=load(ROOT/'config.json')
    def queue(phase,tasks,threshold):
        pending=list(tasks)
        while pending or active:
            reap()
            if not failures:
                free=gpu_memory()
                for gpu in (4,6,7,5):
                    if not pending:break
                    if any(j['gpu']==gpu for j in active.values()) or free.get(gpu,0)<threshold or held(ROOT/'parts'/f'gpu{gpu}.lock'):continue
                    key,script,args=pending.pop(0);start(key,script,[*args,'--gpu',str(gpu)],gpu)
            update(phase)
            if failures and not active:break
            if pending or active:time.sleep(10)
        assert not failures,failures
    tasks=[('smoke_'+b,'train.py',['--branch',b,'--smoke']) for b in BRANCHES if not (ROOT/'branches'/b/'smoke_audit.json').exists()]
    queue('smoke',tasks,cfg['minimum_free_training_mib']);cpu('smoke_gate','gates.py',['smoke'])
    if a.smoke_only:print('SMOKE ONLY COMPLETE',flush=True);return
    tasks=[('train_'+b,'train.py',['--branch',b]) for b in BRANCHES if not ((ROOT/'branches'/b/'training_status.json').exists() and load(ROOT/'branches'/b/'training_status.json')['status']=='PASS')]
    queue('training',tasks,cfg['minimum_free_training_mib']);cpu('checkpoint_integrity','gates.py',['checkpoints'])
    tasks=[]
    for d in DATASETS:
        for v in sources(d):
            for m in BRANCHES:
                if all(point(d,m,v['video_index'],q).exists() for q in range(10)):continue
                tasks.append((f'eval_{d}_{m}_{v["video_index"]}','evaluate.py',['--dataset',d,'--method',m,'--video',str(v['video_index'])]))
    queue('evaluation',tasks,cfg['minimum_free_evaluation_mib']);cpu('report','report.py')
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    update('complete');status=load(ROOT/'pipeline_status.json');status['status']='PASS';dump(ROOT/'pipeline_status.json',status)
    print('V6.3 PIPELINE COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        failure=dict(status='FAIL',pid=os.getpid(),traceback=traceback.format_exc(),unix=time.time());dump(ROOT/'pipeline_failure.json',failure)
        dump(ROOT/'final_integrity.json',dict(status='FAIL',reason='See pipeline_failure.json'))
        if (ROOT/'pipeline_status.json').exists():
            status=load(ROOT/'pipeline_status.json');status.update(status='FAIL',failure=failure);dump(ROOT/'pipeline_status.json',status)
        raise
