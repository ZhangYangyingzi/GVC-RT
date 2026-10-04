"""Detached, locked and resumable new experiment; never signals any GPU process."""
import argparse,fcntl,traceback
from v68_io import *
def held():
    with (ROOT/'pipeline.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args();os.chdir(REPO);(ROOT/'logs').mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if a.launch:
        if held():print(json.dumps(load(ROOT/'pipeline_status.json')));return
        with (ROOT/'logs/pipeline.log').open('a') as f:proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pipeline.pid').write_text(str(proc.pid)+'\n');print(json.dumps(dict(experiment_directory=str(ROOT),PID=proc.pid,status='LAUNCHED',log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def run(script,args=()):
        with (ROOT/'logs'/f'{Path(script).stem}_pipeline.log').open('a') as f:
            proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/script),*args],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=Path(script).stem,PID=os.getpid(),worker_PID=proc.pid,updated_unix=time.time()))
            while proc.poll() is None:time.sleep(5)
            return proc.returncode
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':print('ALREADY PASS');return
    if not (ROOT/'preflight_audit.json').exists():assert run('prepare.py')==0
    frozen()
    gate=ROOT/'audits/lr_control_smoke.json'
    if not gate.exists() or load(gate)['status']!='PASS':
        from parallel_scheduler import gpu_state
        free,_=gpu_state();cards=sorted((g for g in free if free[g]>=26000),key=lambda g:-free[g])
        assert cards,'Insufficient free VRAM on GPUs 4-7 for unchanged smoke'
        for g in cards:
            rc=run('lr_smoke.py',['--gpu',str(g)])
            if rc==0:break
            err=load(gate)
            assert err.get('OOM',False),'LR control smoke failed; formal training blocked'
        assert load(gate)['status']=='PASS'
    from parallel_scheduler import run as parallel
    parallel()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/failures'/f'pipeline_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
