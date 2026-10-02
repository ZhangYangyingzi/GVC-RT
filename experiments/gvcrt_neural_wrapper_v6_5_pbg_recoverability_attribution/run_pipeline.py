"""Detached, resumable inference queue. Never schedule on occupied GPUs."""
import argparse,fcntl,traceback
from v65_io import *

def idle_gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    available=[]
    for line in out.splitlines():
        index,uuid,free=[x.strip() for x in line.split(',')]
        if int(index) in (4,5,6,7) and uuid not in busy and int(free)>=22000:available.append(int(index))
    return available

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    if a.launch:
        pidfile=ROOT/'pipeline.pid'
        if pidfile.exists():
            pid=int(pidfile.read_text())
            try:
                os.kill(pid,0)
                if str(ROOT/'run_pipeline.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes():print(json.dumps(dict(directory=str(ROOT),PID=pid,status=load(ROOT/'pipeline_status.json'),log=str(ROOT/'logs/pipeline.log'))));return
            except (ProcessLookupError,FileNotFoundError):pass
        with (ROOT/'logs/pipeline.log').open('a') as log:
            proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
        pidfile.write_text(str(proc.pid)+'\n');print(json.dumps(dict(directory=str(ROOT),PID=proc.pid,status='LAUNCHED',log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={}
    def status(stage,**extra):dump(ROOT/'pipeline_status.json',dict(status=stage,PID=os.getpid(),updated_unix=time.time(),active=[dict(task=k,gpu=x[1],PID=x[0].pid) for k,x in active.items()],**extra))
    def run(script,*args):subprocess.run([PYTHON,'-B','-u',str(ROOT/script),*args],cwd=REPO,check=True)
    try:
        status('PREFLIGHT');run('prepare.py')
        if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':status('PASS');return
        if not (ROOT/'audits/runtime_smoke.json').exists():
            status('WAITING_FOR_IDLE_GPU_SMOKE')
            while not idle_gpus():time.sleep(20)
            gpu=idle_gpus()[0];status('RUNTIME_SMOKE',gpu=gpu)
            run('factorial_worker.py','--dataset','uvg','--video','0','--gpu',str(gpu),'--smoke')
        assert load(ROOT/'audits/runtime_smoke.json')['status']=='PASS'
        tasks=[(kind,v) for kind in ('proxy','factorial') for v in videos()]
        while tasks or active:
            for key,(proc,gpu,log) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    log.close();del active[key]
                    if code:raise RuntimeError(f'{key} worker exited {code}; see logs/{key}.log')
                    print('FINISHED',key,flush=True)
            tasks=[(kind,v) for kind,v in tasks if not (ROOT/'parts'/f'{kind}_{sid(v)}_done.json').exists()]
            reserved={x[1] for x in active.values()}
            for gpu in idle_gpus():
                if gpu in reserved or not tasks:continue
                kind,v=tasks.pop(0);key=f'{kind}_{sid(v)}';log=(ROOT/'logs'/f'{key}.log').open('a')
                proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/f'{kind}_worker.py'),'--dataset',v['dataset'],'--video',str(v['video_index']),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
                active[key]=(proc,gpu,log);print('STARTED',key,'GPU',gpu,'PID',proc.pid,flush=True)
            status('RUNNING' if active else 'WAITING_FOR_IDLE_GPU',pending=len(tasks),factorial_completed=len(list((ROOT/'parts/factorial').glob('*/*/qp*.json'))),proxy_completed=len(list((ROOT/'parts/proxy').glob('*.json'))))
            if tasks or active:time.sleep(20)
        status('AGGREGATING');run('aggregate.py');status('PASS')
    except Exception:
        status('FAIL',traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        # Let already launched workers finish; never terminate another process.
        for proc,gpu,log in active.values():proc.wait();log.close()
        raise

if __name__=='__main__':main()
