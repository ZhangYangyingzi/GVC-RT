"""Detached recovery supervisor; never terminates any worker or other process."""
from io20 import *
from run_pipeline import identity
if __name__=='__main__':
    command([PYTHON,'-B','-u',str(Path(__file__))]);dump(ROOT/'supervisor_status.json',dict(status='RUNNING',PID=os.getpid(),started_unix=time.time(),command=[PYTHON,'-B','-u',str(Path(__file__))]))
    while True:
        state=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
        if state.get('phase')=='complete':dump(ROOT/'supervisor_status.json',dict(status='PASS',PID=os.getpid(),finished_unix=time.time()));break
        if state.get('status')=='FAIL':dump(ROOT/'supervisor_status.json',dict(status='FAIL',PID=os.getpid(),error=state.get('traceback',state.get('error')),finished_unix=time.time()));break
        pid=state.get('PID')
        if not pid or identity(pid) is None:
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (ROOT/'logs/recovered_pipeline.log').open('a') as stream:subprocess.run([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env)
        time.sleep(10)
