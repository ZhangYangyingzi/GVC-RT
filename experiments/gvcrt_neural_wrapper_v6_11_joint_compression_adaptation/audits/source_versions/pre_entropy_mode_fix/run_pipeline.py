"""Launch and resume the isolated ablation without modifying baseline artifacts."""
import argparse,fcntl,traceback
from v68_io import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    os.chdir(REPO);(ROOT/'logs/failures').mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if args.launch:
        with (ROOT/'pipeline.lock').open('a') as probe:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps(load(ROOT/'pipeline_status.json')));return
        argv=[PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')]
        with (ROOT/'logs/pipeline.log').open('a') as log:
            proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pipeline.pid').write_text(str(proc.pid)+'\n')
        print(json.dumps(dict(status='LAUNCHED',PID=proc.pid,directory=str(ROOT),log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from parallel_scheduler import gpu_state,commands,run
    def child(script,arguments=()):
        argv=[PYTHON,'-B','-u',str(ROOT/script),*arguments];log_path=ROOT/'logs'/f'{Path(script).stem}_pipeline.log'
        commands(argv,log_path)
        with log_path.open('a') as log:
            proc=subprocess.Popen(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=Path(script).stem,PID=os.getpid(),worker_PID=proc.pid,updated_unix=time.time()))
            while proc.poll() is None:time.sleep(5)
            return proc.returncode
    final=ROOT/'final_integrity.json'
    if final.exists() and load(final)['status']=='PASS':print('ALREADY PASS');return
    if not (ROOT/'preflight_audit.json').exists():assert child('prepare.py')==0,'Prepare failed'
    frozen();gate=ROOT/'audits/objective_smoke.json'
    if not gate.exists() or load(gate)['status']!='PASS':
        free,_=gpu_state();eligible=sorted([g for g in free if free[g]>=22000],key=lambda g:-free[g])
        assert eligible,'Insufficient GPU 4–7 memory for disposable objective smoke'
        for gpu in eligible:
            if child('smoke.py',['--gpu',str(gpu)])==0:break
            assert load(gate).get('OOM'), 'Objective smoke failed'
        assert load(gate)['status']=='PASS'
    if not (ROOT/'audits/baseline_reuse_audit.json').exists():
        assert child('reuse.py')==0,'Baseline reuse validation failed'
    run()

if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc()
        dump(ROOT/'logs/failures'/f'controller_{time.time_ns()}.json',dict(traceback=error))
        dump(ROOT/'pipeline_status.json',dict(status='FAIL',phase='controller',PID=os.getpid(),traceback=error,updated_unix=time.time()));raise
