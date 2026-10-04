"""Memory-gated shared-GPU queue: calibration -> smoke -> D/E training -> evaluation -> audits."""
import argparse,fcntl,traceback
from v67_io import *
def free_memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    free={}
    for line in out.splitlines():
        index,uuid,memory=[x.strip() for x in line.split(',')]
        if int(index) in (4,5,6,7):free[int(index)]=int(memory)
    return free
def held():
    with (ROOT/'pipeline.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args();os.chdir(REPO)
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if a.launch:
        if held():print(json.dumps(load(ROOT/'pipeline_status.json')));return
        with (ROOT/'logs/pipeline.log').open('a') as log:
            proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pipeline.pid').write_text(str(proc.pid)+'\n');print(json.dumps(dict(directory=str(ROOT),PID=proc.pid,status='LAUNCHED')));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active={}
    def status(phase,**extra):
        branches={}
        for b in BRANCHES:
            path=ROOT/'branches'/b/'training_status.json';branches[b]=load(path) if path.exists() else dict(status='QUEUED',update=0)
            if 'train_'+b in active:
                j=active['train_'+b];branches[b].update(status='RUNNING',pid=j[0].pid,gpu=j[1],log=str(ROOT/'logs'/f'train_{b}.log'))
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'FAIL' if phase=='failed' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),branches=branches,active={k:dict(PID=j[0].pid,gpu=j[1],log=str(ROOT/'logs'/f'{k}.log')) for k,j in active.items()},**extra))
    def cpu(script):subprocess.run([PYTHON,'-B','-u',str(ROOT/script)],cwd=REPO,check=True)
    def queue(phase,tasks,threshold):
        pending=list(tasks)
        while pending or active:
            for k,(proc,gpu,log) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    log.close();del active[k];assert code==0,f'{k} exited {code}; see log';print('FINISHED',k,flush=True)
            memory=free_memory();used={j[1] for j in active.values()}
            for gpu in (4,6,7,5):
                if not pending:break
                if gpu in used or memory.get(gpu,0)<threshold:continue
                k,script,args=pending.pop(0);log=(ROOT/'logs'/f'{k}.log').open('a')
                proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/script),*args,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,cwd=REPO);active[k]=(proc,gpu,log);print('STARTED',k,'GPU',gpu,'PID',proc.pid,flush=True)
            status('WAITING_FOR_GPU_MEMORY' if pending and not active else phase,pending=len(pending),queued_phase=phase,waiting_for_memory=bool(pending and not active))
            if pending or active:time.sleep(10)
    try:
        status('preflight');cpu('prepare.py');cfg=load(ROOT/'config.json')
        if load(ROOT/'final_integrity.json')['status']=='PASS':status('complete');return
        if not (ROOT/'audits/guard_gradient_calibration.json').exists():queue('calibration',[('calibration_BDE','train.py',['--branch',D,'--calibrate'])],cfg['minimum_free_training_mib'])
        assert load(ROOT/'audits/guard_gradient_calibration.json')['status']=='PASS'
        queue('smoke',[(f'smoke_{b}','train.py',['--branch',b,'--smoke']) for b in BRANCHES if not (ROOT/'branches'/b/'smoke_audit.json').exists()],cfg['minimum_free_training_mib'])
        queue('training',[(f'train_{b}','train.py',['--branch',b]) for b in BRANCHES if not ((ROOT/'branches'/b/'training_status.json').exists() and load(ROOT/'branches'/b/'training_status.json')['status']=='PASS')],cfg['minimum_free_training_mib'])
        status('training_gate');cpu('gates.py')
        tasks=[]
        for d in DATASETS:
            for v in sources(d):
                for b in BRANCHES:
                    if all(point(d,b,v['video_index'],q).exists() for q in range(10)):continue
                    tasks.append((f'eval_{d}_{v["video_index"]}_{b}','evaluate.py',['--dataset',d,'--video',str(v['video_index']),'--method',b]))
        queue('evaluation',tasks,cfg['minimum_free_evaluation_mib'])
        tasks=[]
        for script in ('proxy','attribution'):
            for d in ('uvg','ulong'):
                for v in sources(d):
                    if (ROOT/'parts'/f'{script}_done_{d}_{v["video_index"]}.json').exists():continue
                    tasks.append((f'{script}_{d}_{v["video_index"]}',script+'.py',['--dataset',d,'--video',str(v['video_index'])]))
        queue('proxy_and_attribution',tasks,cfg['minimum_free_evaluation_mib']);status('report');cpu('report.py');status('complete')
    except Exception:
        status('failed',traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        dump(ROOT/'pipeline_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),unix=time.time()))
        for proc,gpu,log in active.values():proc.wait();log.close()
        raise
if __name__=='__main__':main()
