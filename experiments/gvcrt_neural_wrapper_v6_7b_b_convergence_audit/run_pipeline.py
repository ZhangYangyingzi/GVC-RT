"""Detached, resumable real-RANS convergence audit. No training entry points."""
import argparse,fcntl,traceback
from v67b_io import *
def gpu_memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (line.split(',') for line in out.splitlines()) if int(a) in (4,5,6,7)}
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
        (ROOT/'pipeline.pid').write_text(str(proc.pid)+'\n');print(json.dumps(dict(directory=str(ROOT),PID=proc.pid,status='LAUNCHED',log=str(ROOT/'logs/pipeline.log'))));return
    if (ROOT/'audits/real_RANS_smoke.json').exists():
        from parallel_scheduler import run
        return run()
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active={}
    def update(phase,**extra):
        count={m:sum(len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for d in DATASETS) for m in METHODS}
        fidcount=len(list((ROOT/'parts/fid').glob('*/*/qp*.json')))
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'FAIL' if phase=='failed' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),method_points=count,method_target=310,raw_RD_points=sum(count.values()),FID_points=fidcount,FID_target=200,active={k:dict(PID=j[0].pid,gpu=j[1],log=str(ROOT/'logs'/f'{k}.log')) for k,j in active.items()},**extra))
    def cpu(script,args=(),python=PYTHON):
        name=Path(script).stem;path=ROOT/'logs'/f'{name}.log'
        with path.open('a') as f:proc=subprocess.Popen([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        active[name]=(proc,None,f)
        while proc.poll() is None:update(name);time.sleep(5)
        del active[name];assert proc.returncode==0,f'{script} failed; see {path}'
    def queue(phase,tasks,gpu=True):
        pending=list(tasks)
        while pending or active:
            for k,(proc,card,log) in list(active.items()):
                rc=proc.poll()
                if rc is not None:
                    log.close();del active[k];assert rc==0,f'{k} exited {rc}; see logs/{k}.log';print('FINISHED',k,flush=True)
            if gpu:
                memory=gpu_memory();used={j[1] for j in active.values()};slots=[g for g in (4,6,7,5) if g not in used and memory.get(g,0)>=16000]
            else:slots=[None]*max(0,4-len(active))
            for card in slots:
                if not pending:break
                k,script,args=pending.pop(0);log=(ROOT/'logs'/f'{k}.log').open('a');extra=['--gpu',str(card)] if gpu else []
                proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/script),*args,*extra],cwd=REPO,stdout=log,stderr=subprocess.STDOUT);active[k]=(proc,card,log);print('STARTED',k,'GPU',card,'PID',proc.pid,flush=True)
            update(phase,pending=len(pending),waiting_for_memory=bool(pending and not active))
            if pending or active:time.sleep(10)
    try:
        update('preflight');cpu('prepare.py')
        if load(ROOT/'final_integrity.json')['status']=='PASS':update('complete');return
        cpu('selftest.py');cpu('training_audit.py');cpu('plot_training.py',python=PLOT_PYTHON);cpu('reuse.py');cpu('cache_gt.py')
        cpu('fid_worker.py',['--dataset','uvg','--method','B1000','--smoke'])
        assert load(ROOT/'audits/fid_smoke.json')['status']=='PASS'
        smoke_tasks=[(f'smoke_{m}','evaluate.py',['--dataset','uvg','--method',m,'--video','0','--smoke']) for m in FRESH if not (ROOT/'parts'/f'eval_done_uvg_{m}_0_smoke.json').exists()]
        if load(ROOT/'audits/vimeo_heldout_manifest.json')['status']=='AVAILABLE' and not (ROOT/'audits/heldout_smoke.json').exists():smoke_tasks.append(('heldout_smoke','heldout.py',['--method','B250','--smoke']))
        queue('real_RANS_smoke',smoke_tasks)
        smoke=[]
        for m in FRESH:
            for q in (0,9):
                v=next(v for v in sources('uvg') if v['video_index']==0);r=load(point('uvg',m,0,q));validate_point(r,v,q,m);smoke.append(dict(method=m,QP=q,independent_decode=True,real_RANS=True,compression_hash=r['compression_hash_before']))
        dump(ROOT/'audits/real_RANS_smoke.json',dict(status='PASS',points=smoke))
        tasks=[]
        for d in DATASETS:
            for v in sources(d):
                for m in FRESH:
                    if all(point(d,m,v['video_index'],q).exists() for q in range(10)):continue
                    tasks.append((f'eval_{d}_{m}_{v["video_index"]}','evaluate.py',['--dataset',d,'--method',m,'--video',str(v['video_index'])]))
        from parallel_scheduler import run
        return run(lock=lock)
    except Exception:
        error=traceback.format_exc();update('failed',traceback=error);dump(ROOT/'pipeline_failure.json',dict(status='FAIL',traceback=error,unix=time.time()));print(error,flush=True)
        for proc,card,log in active.values():proc.wait();log.close()
        raise
if __name__=='__main__':main()
