"""Resumable V5-A.2 queue; launches evaluation, reporting, and audits only."""
import fcntl
import os
import subprocess
import time
from audit_utils import *
GPUS=(4,6)
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def launch(script,args,log,gpu=None,python=PYTHON):
    assert gpu is None or gpu in GPUS
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (ROOT/'logs'/log).open('ab') as out:
        child=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT)
    print('START',script,args,'pid',child.pid,'gpu',gpu,flush=True)
    return child

def cpu(script,args,log,python=PYTHON):
    dump('pipeline_status.json',dict(status='RUNNING',phase=script,pid=os.getpid(),evaluation_only=True,gpu_ids=list(GPUS)))
    child=launch(script,args,log,python=python)
    assert child.wait()==0,f'{script} failed; inspect logs/{log}'

def complete(dataset,method):
    return all(point_path(dataset,method,i,q).exists() for i in range(len(videos(dataset))) for q in range(10))

def counts():
    return {d:{m:len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for m in METHODS} for d in ('ulong','uvg')}

def gpu_free(gpu):
    # Do not stop or contend with unrelated existing CUDA workloads.
    text=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,util=map(int,text.strip().split(','))
    return memory<512 and util<10

def main():
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};stats=None
    try:
        assert load('config.json')['evaluation_only'] and not load('config.json')['training_allowed']
        assert load('canonical_loader_audit.json')['status']=='PASS'
        assert load('canonical_source_audit.json')['status']=='PASS'
        if not (ROOT/'source_statistics_audit.json').exists():stats=launch('source_statistics.py',[],'source_statistics.log')
        queue=[(d,m) for m in METHODS for d in ('ulong','uvg') if not complete(d,m)]
        while queue or active:
            if stats is not None and stats.poll() not in (None,0):raise RuntimeError('source statistics failed')
            for gpu,(dataset,method,child) in list(active.items()):
                if child.poll() is not None:
                    assert child.returncode==0 and complete(dataset,method),f'{dataset}/{method} failed: exit {child.returncode}'
                    del active[gpu]
            for gpu in GPUS:
                if gpu not in active and queue and gpu_free(gpu):
                    dataset,method=queue.pop(0)
                    child=launch('evaluate.py',['--gpu',str(gpu),'--dataset',dataset,'--method',method],f'{dataset}_{method}.log',gpu)
                    active[gpu]=(dataset,method,child)
            status=dict(status='RUNNING',phase='fresh_RANS_evaluation',evaluation_only=True,pid=os.getpid(),gpu_ids=list(GPUS),
                active={str(g):dict(dataset=d,method=m,pid=p.pid) for g,(d,m,p) in active.items()},queue=queue,
                points=counts(),expected_points={'ulong':240,'uvg':210},updated_unix=time.time())
            dump('pipeline_status.json',status)
            if active or queue:time.sleep(20)
        if stats is not None:assert stats.wait()==0
        for dataset in ('ulong','uvg'):
            cpu('report.py',['--dataset',dataset],f'report_{dataset}.log')
            cpu('plot_results.py',['--dataset',dataset],f'plots_{dataset}.log',PLOT_PYTHON)
        cpu('finalize.py',[],'finalize.log')
        cpu('print_status.py',[],'final_stdout.log')
        assert load('final_integrity.json')['status']=='PASS'
        dump('pipeline_status.json',dict(status='PASS',phase='complete',evaluation_only=True,no_training=True,points=counts()))
        print('PIPELINE PASS',flush=True)
    except BaseException as exc:
        for _,_,child in active.values():
            if child.poll() is None:child.terminate()
        if stats is not None and stats.poll() is None:stats.terminate()
        dump('pipeline_status.json',dict(status='FAIL',phase='stopped',evaluation_only=True,error=repr(exc),points=counts()))
        raise
if __name__=='__main__':main()

