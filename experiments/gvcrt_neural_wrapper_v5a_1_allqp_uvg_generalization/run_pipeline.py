"""Resumable evaluation-only queue. Never launches a training command."""
import fcntl
import os
import subprocess
import time
from audit_utils import *

GPUS=(4,6)  # GPU5/7 belong to other live processes at launch; GPUs0-3 never used.
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def launch(script,args,log,gpu=None,python=PYTHON):
    assert gpu is None or gpu in GPUS
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (ROOT/'logs'/log).open('ab') as out:
        child=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print('START',script,args,'pid',child.pid,'gpu',gpu,flush=True);return child

def cpu(script,args,log,python=PYTHON):
    dump('pipeline_status.json',dict(status='RUNNING',phase=script,pid=os.getpid(),arguments=args,
        evaluation_only=True,gpu_ids=list(GPUS),log=str(ROOT/'logs'/log)))
    child=launch(script,args,log,python=python)
    if child.wait()!=0:raise RuntimeError(f'{script} failed; inspect logs/{log}')

def complete(dataset,method):
    return all(point_path(dataset,method,i,q).is_file() for i in range(len(videos(dataset))) for q in range(10))

def main():
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={}
    try:
        dump('pipeline_status.json',dict(status='RUNNING',phase='resume_checks',evaluation_only=True,pid=os.getpid(),gpu_ids=list(GPUS)))
        assert load('config.json')['evaluation_only']
        assert load('qp0_3_backward_compatibility_audit.json')['status']=='PASS'
        for dataset in ('ulong','uvg'):
            queue=[m for m in METHODS if not complete(dataset,m)];active={};last=None
            while queue or active:
                for gpu,(method,p) in list(active.items()):
                    if p.poll() is not None:
                        if p.returncode!=0 or not complete(dataset,method):raise RuntimeError(f'{dataset}/{method} failed: exit={p.returncode}')
                        del active[gpu]
                for gpu in GPUS:
                    if gpu not in active and queue:
                        method=queue.pop(0)
                        active[gpu]=(method,launch('evaluate.py',['--gpu',str(gpu),'--dataset',dataset,'--method',method],f'{dataset}_{method}.log',gpu))
                status=dict(status='RUNNING',phase=dataset,evaluation_only=True,pid=os.getpid(),gpu_ids=list(GPUS),
                    active={str(g):dict(method=m,pid=p.pid) for g,(m,p) in active.items()},
                    points={d:{m:len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for m in METHODS} for d in ('ulong','uvg')},
                    expected_points={'ulong':320,'uvg':len(videos('uvg'))*40})
                if status!=last:
                    dump('pipeline_status.json',status);print(status,flush=True);last=status
                if active:time.sleep(20)
            cpu('report.py',['--dataset',dataset],f'report_{dataset}.log')
            cpu('plot_results.py',['--dataset',dataset],f'plots_{dataset}.log',PLOT_PYTHON)
        cpu('finalize.py',[],'finalize.log')
        cpu('print_status.py',[],'final_stdout.log')
        assert load('final_integrity.json')['status']=='PASS'
        dump('pipeline_status.json',dict(status='PASS',phase='complete',evaluation_only=True,no_training_performed=True))
        print('PIPELINE PASS',flush=True)
    except Exception as exc:
        # Stop only this pipeline's own still-running children after a hard audit failure.
        for _,p in active.values():
            if p.poll() is None:p.terminate()
        dump('pipeline_status.json',dict(status='FAIL',phase='stopped',evaluation_only=True,error=repr(exc)));raise

if __name__=='__main__':main()
