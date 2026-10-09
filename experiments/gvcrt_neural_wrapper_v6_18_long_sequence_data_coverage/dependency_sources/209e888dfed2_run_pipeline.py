import fcntl
import os
import subprocess
import time
from diag_utils import *
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
def launch(script,args,log,gpu=None,python=PYTHON):
    assert gpu is None or gpu in (4,6)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (ROOT/'logs'/log).open('ab') as f:
        p=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    print('START',script,args,p.pid,flush=True);return p
def status(phase,**extra):dump('pipeline_status.json',dict(status='RUNNING',phase=phase,pid=os.getpid(),evaluation_only=True,**extra))
def cpu(script,args,log,python=PYTHON):
    status(script);p=launch(script,args,log,python=python)
    assert p.wait()==0,f'{script} failed; see logs/{log}'
def main():
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={}
    try:
        assert load('conversion_smoke.json')['status']=='PASS'
        if not (ROOT/'uvg_color_conversion_audit.json').exists():cpu('conversion.py',[],'conversion.log')
        cpu('plots_tables.py',[],'plots_tables.log',PLOT_PYTHON)
        tasks=[(i,False) for i in range(len(samples()))]+[(0,True)]
        tasks=[t for t in tasks if not (ROOT/f'parts/comparison_{tag(samples()[t[0]])}_{"709" if t[1] else "current"}.json').exists()]
        while tasks or running:
            for gpu,(task,p) in list(running.items()):
                if p.poll() is not None:
                    assert p.returncode==0,f'worker {task} failed: {p.returncode}'
                    del running[gpu]
            for gpu in (4,6):
                if gpu not in running and tasks:
                    task=tasks.pop(0);i,smoke=task
                    args=['--gpu',str(gpu),'--sample',str(i)]+(['--smoke'] if smoke else [])
                    running[gpu]=(task,launch('worker.py',args,f'worker_{i}_{"709" if smoke else "current"}.log',gpu))
            status('diagnostic_workers',active={str(g):dict(sample=t[0],smoke=t[1],pid=p.pid) for g,(t,p) in running.items()},
                remaining_jobs=len(tasks),completed_jobs=len(list((ROOT/'parts').glob('comparison_*.json'))))
            if running:time.sleep(20)
        cpu('finalize.py',[],'finalize.log')
        cpu('print_status.py',[],'final_stdout.log')
        assert load('final_integrity.json')['status']=='PASS'
        dump('pipeline_status.json',dict(status='PASS',phase='complete',evaluation_only=True))
        print('DIAGNOSTIC PIPELINE PASS',flush=True)
    except Exception as exc:
        for _,p in running.values():
            if p.poll() is None:p.terminate()
        dump('pipeline_status.json',dict(status='FAIL',error=repr(exc),evaluation_only=True));raise
if __name__=='__main__':main()
