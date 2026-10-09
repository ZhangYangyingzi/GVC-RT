"""Persistent original-style process-per-GPU orchestration; safe restart by point."""
import fcntl
from v61_io import *
def launch(script,args,log,gpu=None,python=PYTHON):
    assert gpu is None or gpu in (4,5,6,7)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'logs'/log).open('ab') as out:
        p=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print('START',script,args,'PID',p.pid,'GPU',gpu,flush=True);return p
def cpu(script,args,log,python=PYTHON):
    p=launch(script,args,log,python=python)
    if p.wait()!=0:raise RuntimeError(f'{script} failed; see logs/{log}')
def locked(path):
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True
def data_ready():
    while not (ROOT/'data_split_integrity.json').exists():
        path=ROOT/'parts/data.lock'
        with path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);running=False
            except BlockingIOError:running=True
        if not running:cpu('prepare_data.py',[],'prepare_data_resume.log')
        else:time.sleep(20)
    assert load(ROOT/'data_split_integrity.json')['status']=='PASS'
def complete(split,method):
    vs=load(ROOT/('validation_sources.json' if split=='validation' else 'final_sources.json'))['videos'];vs=[v for v in vs if v['dataset']==split]
    qps=load(ROOT/'config.json')['validation_qps'] if split=='validation' else list(range(10))
    return all(point(split,method,v['video_index'],q).exists() for v in vs for q in qps)
def batch(jobs,gpus):
    pending=list(jobs);active={};attempts={};completed=[]
    while pending or active:
        for gpu,(job,p) in list(active.items()):
            if p.poll() is None:continue
            del active[gpu]
            if p.returncode:
                attempts[job]=attempts.get(job,0)+1
                dump(ROOT/'logs'/f'failure_{time.time_ns()}.json',dict(job=job,gpu=gpu,exit_code=p.returncode,attempt=attempts[job]))
                assert attempts[job]<=2,('Persistent job failure',job)
                pending.append(job)
            else:completed.append(job)
        for gpu in gpus:
            if gpu in active or not pending:continue
            job=pending.pop(0);split,method,shard,shards,parity=job
            args=['--split',split,'--method',method,'--gpu',str(gpu),'--shard',str(shard),'--shards',str(shards)]
            if parity:args.append('--parity-only')
            active[gpu]=(job,launch('evaluate.py',args,f'eval_{split}_{method}_{shard}.log',gpu))
        if pending or active:time.sleep(10)
def train_validate():
    cfg=load(ROOT/'config.json');training=None;attempts=0;workers={};tries={};done=set()
    methods=['original']+[f'step_{s}' for s in cfg['checkpoint_additional_updates']]
    gpu_map=dict(zip(methods,[5,6,7,5,6,7,5]))
    while True:
        trained=(ROOT/'parts/train_done.json').exists()
        if not trained and (training is None or training.poll() is not None) and not locked(ROOT/'parts/train.lock'):
            if training is not None:
                attempts+=1;dump(ROOT/'logs'/f'train_failure_{time.time_ns()}.json',dict(exit_code=training.returncode,retry=attempts));assert attempts<=2,'Training repeated failure'
            training=launch('train.py',['--gpu','4'],'train.log',4)
        if trained:assert load(ROOT/'parts/train_done.json')['status']=='PASS'
        for gpu,(method,p) in list(workers.items()):
            if p.poll() is None:continue
            del workers[gpu]
            if p.returncode:
                tries[method]=tries.get(method,0)+1;dump(ROOT/'logs'/f'validation_failure_{time.time_ns()}.json',dict(method=method,exit_code=p.returncode,retry=tries[method]));assert tries[method]<=2,(method,'repeated failure')
            else:assert complete('validation',method);done.add(method)
        busy={gpu_map[m]:m for m in methods if locked(ROOT/'parts'/f'eval_validation_{m}_0.lock')}
        for method in methods:
            if method in done or method in {m for m,_ in workers.values()}:continue
            if method in busy.values():continue
            if complete('validation',method):done.add(method);continue
            if method!='original' and not (ROOT/'parts'/f"checkpoint_{int(method.split('_')[-1]):05d}.json").exists():continue
            idle=gpu_map[method]
            if idle in workers or idle in busy:continue
            workers[idle]=(method,launch('evaluate.py',['--split','validation','--method',method,'--gpu',str(idle)],f'validation_{method}.log',idle))
        trainstatus=load(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {'status':'STARTING'}
        status(status='RUNNING',phase='training_and_validation',training=trainstatus,validation_completed_methods=sorted(done),validation_active={**{str(g):m for g,m in busy.items()},**{str(g):m for g,(m,_) in workers.items()}})
        if trained and len(done)==len(methods) and not workers and not busy:break
        time.sleep(20)
def main():
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':return
    try:
        assert load(ROOT/'initialization_smoke_audit.json')['status']=='PASS'
        assert load(ROOT/'runtime_dependency_audit.json')['status']=='PASS'
        assert load(ROOT/'final_cohort_split_audit.json')['status']=='PASS'
        dump(ROOT/'pipeline_process.json',dict(pid=os.getpid(),start_unix=time.time(),gpu_training=4,gpu_validation=[5,6,7]))
        data_ready()
        if not (ROOT/'final_sources.json').exists():cpu('prepare_eval.py',[],'prepare_eval.log')
        train_validate()
        cpu('report.py',['--split','validation'],'report_validation.log');cpu('plot.py',['--split','validation'],'plot_validation.log',PLOT_PYTHON)
        assert load(ROOT/'checkpoint_selection.json')['used_final_test'] is False
        status(status='RUNNING',phase='baseline_parity')
        batch([(d,'original',0,1,True) for d in ('ulong','uvg')],[4,5])
        cpu('baseline_reuse.py',[],'baseline_reuse.log')
        status(status='RUNNING',phase='final_evaluation')
        jobs=[(d,m,s,2,False) for d in ('ulong','uvg','virat720') for m in ('original','selected') if not complete(d,m) for s in range(2)]
        batch(jobs,[4,5,6,7])
        for d in ('ulong','uvg','virat720'):
            cpu('report.py',['--split',d],f'report_{d}.log');cpu('plot.py',['--split',d],f'plot_{d}.log',PLOT_PYTHON)
        cpu('finalize.py',[],'finalize.log')
        print('COMPLETE',ROOT,load(ROOT/'checkpoint_selection.json')['checkpoint'],flush=True)
    except BaseException as exc:
        status(status='FAIL',error=repr(exc));dump(ROOT/'logs'/f'pipeline_failure_{time.time_ns()}.json',dict(error=repr(exc)));raise
if __name__=='__main__':main()
