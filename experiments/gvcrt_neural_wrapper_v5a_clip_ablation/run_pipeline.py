import fcntl
import os
import subprocess
import time
from v5_utils import ROOT,REPO,PYTHON,BRANCHES,METHODS,checkpoint,load,dump

PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def active(path):
    with open(path,'a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        return False

def launch(script,args,log,gpu=None,python=PYTHON):
    assert gpu is None or gpu in (4,5,6,7)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with open(ROOT/'logs'/log,'ab') as out:
        p=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print('START',script,args,'pid',p.pid,'gpu',gpu,flush=True);return p

def cpu(script,args,log,python=PYTHON):
    p=launch(script,args,log,python=python)
    if p.wait()!=0:raise RuntimeError(f'{script} failed; inspect logs/{log}')

def complete(split,method):
    n=6 if split=='validation' else 8
    return all((ROOT/'parts'/split/method/f'video_{i}_qp{q}.json').exists() for i in range(n) for q in range(4))

def training_status():
    return {b:{'complete':(ROOT/'parts'/f'train_{b}_done.json').exists(),
               'lock_active':active(ROOT/'parts'/f'train_{b}.lock')} for b in BRANCHES}

def phase(split):
    mapping={'original':6,'v41_20000':7,'clip4_control':6 if split=='validation' else 4,'clip8':7 if split=='validation' else 5}
    children={};attempted=set();last=None
    while True:
        done={m for m in METHODS if complete(split,m)}
        busy={m:mapping[m] for m in METHODS if active(ROOT/'parts'/f'eval_{split}_{m}.lock')}
        for method,p in children.items():
            code=p.poll()
            if code is None:busy[method]=mapping[method]
            elif code!=0:raise RuntimeError(f'{split}/{method} exited {code}; see logs/{split}_{method}.log')
            elif method not in done:raise RuntimeError(f'{split}/{method} exited with missing points')
        status=dict(status='RUNNING',phase=split,points={m:len(list((ROOT/'parts'/split/m).glob('video_*_qp*.json'))) for m in METHODS},training=training_status())
        if status!=last:
            dump('pipeline_status.json',dict(status,updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print(status,flush=True);last=status
        if len(done)==len(METHODS) and not busy:return
        for method in METHODS:
            if method in done or method in busy or mapping[method] in busy.values():continue
            if method in attempted:raise RuntimeError(f'{method} stopped before completion; resume using this pipeline')
            if method in BRANCHES and not (ROOT/'parts'/f'train_{method}_done.json').exists():continue
            children[method]=launch('evaluate.py',['--gpu',str(mapping[method]),'--split',split,'--method',method],f'{split}_{method}.log',mapping[method])
            attempted.add(method);busy[method]=mapping[method]
        if split=='validation':
            for branch,state in training_status().items():
                if not state['complete'] and not state['lock_active']:raise RuntimeError(f'training {branch} stopped; inspect its log before resuming')
        time.sleep(20)

def main():
    lock=open(ROOT/'parts/pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        dump('pipeline_status.json',dict(status='RUNNING',phase='initialization',pid=os.getpid()))
        for b,gpu in [('clip4_control',4),('clip8',5)]:
            if not (ROOT/'parts'/f'train_{b}_done.json').exists() and not active(ROOT/'parts'/f'train_{b}.lock'):
                p=launch('train.py',['--gpu',str(gpu),'--branch',b],f'train_{b}.log',gpu)
                deadline=time.monotonic()+120
                while not active(ROOT/'parts'/f'train_{b}.lock'):
                    if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError(f'{b} startup failed')
                    time.sleep(2)
        while not all((ROOT/'parts'/f'temporal_{b}.json').exists() and (ROOT/'training_logs'/f'positions_{b}.csv').exists() for b in BRANCHES):
            for b,state in training_status().items():
                if not state['complete'] and not state['lock_active']:raise RuntimeError(f'{b} initialization failed')
            time.sleep(10)
        cpu('audit_training.py',['--initial'],'initialization_audit.log')
        phase('validation')
        cpu('audit_training.py',[],'training_audit.log')
        cpu('report.py',['--split','validation'],'report_validation.log')
        cpu('plot_results.py',['--split','validation'],'plot_validation.log',PLOT_PYTHON)
        phase('final')
        cpu('report.py',['--split','final'],'report_final.log')
        cpu('plot_results.py',['--split','final'],'plot_final.log',PLOT_PYTHON)
        cpu('finalize.py',[],'finalize.log')
        cpu('print_status.py',[],'final_stdout.log')
        assert load('final_integrity.json')['status']=='PASS'
        dump('pipeline_status.json',dict(status='PASS',phase='complete',branches=list(BRANCHES),checkpoint_selection=False))
        print('PIPELINE PASS',flush=True)
    except Exception as exc:
        dump('pipeline_status.json',dict(status='FAIL',error=repr(exc)));raise

if __name__=='__main__':main()
