"""Resume existing jobs, validate, select, then evaluate final test."""
import fcntl
import json
import os
import subprocess
import time
from experiment_utils import ROOT,REPO,PYTHON,dump,sha

PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def active(lock_path):
    with open(lock_path,'a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        return False

def complete(split,method):
    n=6 if split=='validation' else 8
    return all((ROOT/'parts'/split/method/f'video_{i}_qp{q}.json').exists() for i in range(n) for q in range(4))

def launch(script,args,log,gpu=None,python=PYTHON):
    env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['OMP_NUM_THREADS']='4'; env['MKL_NUM_THREADS']='4'; env['OPENBLAS_NUM_THREADS']='4'
    env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    assert gpu is None or gpu in (4,5,6,7)
    with open(ROOT/'logs'/log,'ab') as out:
        p=subprocess.Popen([python,'-B',str(ROOT/script),*args],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print('START',script,args,'pid',p.pid,'gpu',gpu,flush=True)
    return p

def cpu(script,args,log,python=PYTHON):
    p=launch(script,args,log,python=python)
    if p.wait()!=0: raise RuntimeError(f'{script} failed; see logs/{log}')

def phase(split):
    methods=['original','v3','v4_10000','v4_15000','v4_20000'] if split=='validation' else ['original','v3','v4_10000','selected']
    gpu_map={'original':6,'v3':5,'v4_10000':7,'v4_15000':6,'v4_20000':5,'selected':4}
    children={}; attempted=set(); last=None
    while True:
        if split=='final':
            selection=json.loads((ROOT/'selected_checkpoint.json').read_text())
            if selection['step']==10000 and complete('final','v4_10000') and not complete('final','selected'):
                assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
                for path in sorted((ROOT/'parts/final/v4_10000').glob('video_*_qp*.json')):
                    row=json.loads(path.read_text())
                    assert row['checkpoint_sha256']==selection['checkpoint_sha256']
                    assert row['feature_sha256']==sha(row['feature_path']) and row['bitstream_sha256']==sha(row['bitstream_path'])
                    row.update(method='selected',step=10000,checkpoint=selection['checkpoint'],
                               provenance='selected identical step10000; reuse verified final spatial, RANS and new FloLPIPS/FID features')
                    dump(ROOT/'parts/final/selected'/path.name,row)
        finished={m for m in methods if complete(split,m)}
        busy={}
        for method in methods:
            if active(ROOT/'parts'/f'eval_{split}_{method}.lock'):
                busy[method]=gpu_map[method]
        for method,p in list(children.items()):
            result=p.poll()
            if result is None: busy[method]=gpu_map[method]
            elif result!=0: raise RuntimeError(f'{split} {method} exited {result}; inspect logs/{split}_{method}.log')
            elif not complete(split,method): raise RuntimeError(f'{split} {method} exited without all points')
        state={m:len(list((ROOT/'parts'/split/m).glob('video_*_qp*.json'))) for m in methods}
        if state!=last:
            dump('pipeline_status.json',dict(status='RUNNING',phase=split,points=state,updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print(split,state,flush=True); last=state
        if len(finished)==len(methods) and not busy: break
        for method in methods:
            if method in finished or method in busy: continue
            if split=='final' and method=='selected' and selection['step']==10000: continue
            if method in attempted: raise RuntimeError(f'{split} {method} stopped before completing; safe to rerun pipeline to resume')
            if gpu_map[method] in busy.values(): continue
            if method in ('v4_15000','v4_20000') and not (ROOT/'checkpoints/beta_high'/f'step_{method.split("_")[-1]}.pt').exists(): continue
            children[method]=launch('evaluate.py',['--gpu',str(gpu_map[method]),'--split',split,'--method',method],f'{split}_{method}.log',gpu_map[method])
            attempted.add(method); busy[method]=gpu_map[method]
        if split=='validation':
            if not (ROOT/'parts/train_done.json').exists() and not active(ROOT/'parts/train.lock'):
                raise RuntimeError('training stopped without train_done.json; resume train_extension.py on GPU4')
        time.sleep(20)

def main():
    lock=open(ROOT/'parts/pipeline.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        dump('pipeline_status.json',dict(status='RUNNING',phase='starting',pid=os.getpid()))
        if not (ROOT/'parts/train_done.json').exists() and not active(ROOT/'parts/train.lock'):
            training=launch('train_extension.py',['--gpu','4'],'train_extension.log',4)
            deadline=time.monotonic()+120
            while not active(ROOT/'parts/train.lock'):
                if training.poll() is not None: raise RuntimeError('training failed to start')
                if time.monotonic()>deadline: raise RuntimeError('training startup lock timeout')
                time.sleep(2)
        phase('validation')
        while not (ROOT/'parts/train_done.json').exists():
            if not active(ROOT/'parts/train.lock'): raise RuntimeError('training exited without completion')
            time.sleep(20)
        cpu('report.py',['--split','validation'],'report_validation.log')
        cpu('plot_results.py',['--split','validation'],'plot_validation.log',PLOT_PYTHON)
        selection=json.loads((ROOT/'selected_checkpoint.json').read_text())
        if selection['status']!='PASS':
            dump('pipeline_status.json',dict(status='NO_ELIGIBLE_CANDIDATE',phase='selection',reason='All requested candidates failed LPIPS/DISTS validation gates; final test not started'))
            dump('final_integrity.json',dict(status='FAIL',checkpoint_selected_without_final_test=False,
                 final_8_ulong_complete=False,reason='NO_ELIGIBLE_CANDIDATE; validation artifacts retained, final test intentionally not started'))
            print('NO_ELIGIBLE_CANDIDATE; final test not started',flush=True)
            return
        phase('final')
        cpu('report.py',['--split','final'],'report_final.log')
        cpu('plot_results.py',['--split','final'],'plot_final.log',PLOT_PYTHON)
        cpu('finalize_integrity.py',[],'finalize_integrity.log')
        cpu('print_status.py',[],'final_stdout.log')
        dump('pipeline_status.json',dict(status='PASS',phase='complete',selected_step=selection['step']))
        print('PIPELINE PASS',flush=True)
    except Exception as exc:
        dump('pipeline_status.json',dict(status='FAIL',error=repr(exc)))
        raise

if __name__=='__main__': main()
