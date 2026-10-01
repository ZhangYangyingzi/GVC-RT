"""Persistent process-per-GPU evaluation queue; never preempts other jobs."""
import argparse
import fcntl
import traceback
from v62b_io import *

def available_gpus():
    # User-authorized evaluation on every physical GPU 4-7, including GPU5
    # while unrelated work may already occupy memory. Never terminate it.
    return [4,5,6,7]

def lock_held(path):
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True

def video_complete(dataset,method,index):return all(point(dataset,method,index,q).exists() for q in range(10))
def dataset_complete(dataset):return all(video_complete(dataset,m,v['video_index']) for m in METHODS for v in sources(dataset))

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    statuspath=ROOT/'pipeline_status.json'
    if a.status:
        print(json.dumps(load(statuspath) if statuspath.exists() else {'status':'NOT_STARTED'},indent=2));return
    if a.launch:
        assert load(ROOT/'preflight_audit.json')['status']=='PASS'
        if lock_held(ROOT/'parts/pipeline.lock'):print('ALREADY RUNNING');return
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (ROOT/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([PYTHON,'-B','-u',str(Path(__file__).resolve())],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(status='LAUNCHED',pid=child.pid,log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=check_frozen();assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    dump(ROOT/'execution_code_audit.json',dict(started_unix=time.time(),sources={p.name:sha(p) for p in ROOT.glob('*.py')}))
    active={};failed={};finalized=False
    def launch(key,command,gpu=None):
        logpath=ROOT/'logs'/(key+'.log')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
        with logpath.open('a') as log:
            proc=subprocess.Popen([PYTHON,'-B','-u',*command],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        active[key]=dict(process=proc,gpu=gpu,log=str(logpath),started_unix=time.time());print('START',key,'GPU',gpu,'PID',proc.pid,flush=True)
    def pending():
        for d in DATASETS:
            for method in METHODS:
                for v in sources(d):
                    key=f'eval_{d}_{method}_{v["video_index"]}'
                    if key not in failed and key not in active and not video_complete(d,method,v['video_index']) and not lock_held(ROOT/'parts'/(key+'.lock')):
                        yield key,d,method,v['video_index']
    while True:
        for key,job in list(active.items()):
            rc=job['process'].poll()
            if rc is None:continue
            print('EXIT',key,rc,flush=True)
            if rc:
                failed[key]=dict(returncode=rc,log=job['log'],ended_unix=time.time());dump(ROOT/'failures.json',dict(status='FAIL',jobs=failed))
            if key=='finalize' and rc==0:finalized=True
            del active[key]
        if finalized:
            assert load(ROOT/'final_integrity.json')['status']=='PASS'
            dump(statuspath,dict(status='PASS',pid=os.getpid(),completed_points=cfg['expected_points'],expected_points=cfg['expected_points'],updated_unix=time.time()))
            print('V62B COMPLETE',flush=True);return
        for gpu in available_gpus():
            if any(j['gpu']==gpu for j in active.values()):continue
            task=next(pending(),None)
            if task:
                key,d,m,i=task;launch(key,[str(ROOT/'evaluate.py'),'--dataset',d,'--method',m,'--video',str(i),'--gpu',str(gpu)],gpu)
        if not any(j['gpu'] is None for j in active.values()):
            ready=next((d for d in DATASETS if 'report_'+d not in failed and dataset_complete(d) and not (ROOT/'results'/d/'report_integrity.json').exists()),None)
            if ready:launch('report_'+ready,[str(ROOT/'report.py'),'--dataset',ready])
            elif not failed and all((ROOT/'results'/d/'report_integrity.json').exists() for d in DATASETS):
                launch('finalize',[str(ROOT/'report.py'),'--finalize'])
        counts={d:{m:len(list((ROOT/'parts'/d/m).glob('video_*.json'))) for m in METHODS} for d in DATASETS}
        total=sum(sum(r.values()) for r in counts.values())
        dump(statuspath,dict(status='RUNNING_WITH_FAILURES' if failed else 'RUNNING',pid=os.getpid(),updated_unix=time.time(),
            completed_points=total,expected_points=cfg['expected_points'],points_by_dataset_method=counts,
            dataset_reports={d:(ROOT/'results'/d/'report_integrity.json').exists() for d in DATASETS},allowed_gpus=[4,5,6,7],
            active_jobs={k:dict(pid=j['process'].pid,gpu=j['gpu'],log=j['log'],started_unix=j['started_unix']) for k,j in active.items()},
            gpu_policy='User authorized physical GPUs 4,5,6,7; unrelated processes are left running'))
        if failed and not active and not next(pending(),None):
            dump(ROOT/'final_integrity.json',dict(status='FAIL',failed_jobs=failed))
            s=load(statuspath);s['status']='FAIL';dump(statuspath,s);return
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'pipeline_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time_unix=time.time()))
        dump(ROOT/'final_integrity.json',dict(status='FAIL',reason='See pipeline_failure.json'));raise
