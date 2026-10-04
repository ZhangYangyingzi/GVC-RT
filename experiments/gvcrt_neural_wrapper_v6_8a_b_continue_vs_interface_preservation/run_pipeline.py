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
    gate=ROOT/'audits/interface_alignment_calibration.json'
    if not gate.exists():
        from parallel_scheduler import gpu_state
        tried=set()
        while len(tried)<4:
            free,_=gpu_state();cards=[g for g in free if g not in tried and free[g]>=26000]
            if not cards:
                if tried:
                    dump(ROOT/'final_integrity.json',dict(status='C1_GPU_OOM_NO_GO',reason='No remaining allowed GPU currently has sufficient memory for unchanged protocol'));return
                dump(ROOT/'pipeline_status.json',dict(status='WAITING_FOR_MEMORY',phase='interface_audit',PID=os.getpid(),free_memory=free));time.sleep(10);continue
            g=max(cards,key=lambda g:free[g]);tried.add(g);rc=run('audit_interface.py',['--gpu',str(g)])
            if rc==0:break
            failed=load(ROOT/'audits/interface_preservation_feasibility.json')
            if failed['status']!='C1_GPU_OOM_NO_GO':
                dump(ROOT/'final_integrity.json',dict(status='NO_GO',reason='alignment interface invalid',audit=failed));print('ALIGNMENT_INTERFACE_NOT_VALID',flush=True);return
        assert gate.exists(),'C1_GPU_OOM_NO_GO'
    assert load(gate)['status']=='PASS'
    from parallel_scheduler import run as parallel
    parallel()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/failures'/f'pipeline_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
