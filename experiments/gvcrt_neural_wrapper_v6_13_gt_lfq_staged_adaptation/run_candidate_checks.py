"""Run only disposable teacher checks; no main training permission gate bypass."""
import subprocess
from io_utils import *
def main():
    command([sys.executable,'-B','-u',str(Path(__file__).resolve())])
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(a):int(b) for a,b in (line.split(',') for line in text.splitlines()) if int(a) in (4,5,6,7)}
    candidates=[('imagenet256',True),('pretrain262144',False),('pretrain262144',True)]
    active=[]
    for candidate,raw in candidates:
        name=candidate+('_raw' if raw else '_ema')
        done=ROOT/'teacher_checks'/name/'teacher_compatibility_raw.json'
        if done.exists() and load(done)['status']=='MEASUREMENTS_COMPLETE_NOT_GATE_PASS':continue
        gpu=max(free,key=free.get);assert free[gpu]>8000
        args=[PYTHON,'-B','-u',str(ROOT/'check_teacher.py'),'--gpu',str(gpu),'--candidate',candidate]+(['--raw'] if raw else [])
        log=(ROOT/'logs'/f'{name}.log').open('a');command(args)
        proc=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);active.append((name,gpu,proc,log));free[gpu]-=8000
        print('START',name,'GPU',gpu,'PID',proc.pid,flush=True)
    while active:
        for name,gpu,proc,log in active[:]:
            rc=proc.poll()
            if rc is None:continue
            log.close();active.remove((name,gpu,proc,log));print('FINISH',name,'RC',rc,flush=True)
            if rc:dump(ROOT/'logs/failures'/f'{name}_{time.time_ns()}.json',dict(returncode=rc,log=str(ROOT/'logs'/f'{name}.log')))
        time.sleep(5)
if __name__=='__main__':main()
