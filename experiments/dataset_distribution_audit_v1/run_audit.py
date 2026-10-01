"""Resumable deterministic phase runner; no training and no external downloads."""
import argparse
import fcntl
import subprocess
import traceback
from audit_io import *
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    if a.launch:
        assert load(ROOT/'cohort_audit.json')['status']=='PASS'
        with (ROOT/'parts/pipeline.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:print('ALREADY RUNNING');return
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (ROOT/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([PYTHON,'-B','-u',str(Path(__file__).resolve())],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(status='LAUNCHED',pid=child.pid,log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'final_integrity.json',dict(status='PENDING',reason='Statistics, codec profiles, coverage, plots and integrity phases pending'))
    if not (ROOT/'execution_sources.json').exists():dump(ROOT/'execution_sources.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
    for path,h in load(ROOT/'frozen_dependencies.json').items():assert sha(path)==h,path
    from select_codec import motion_plan
    motion_plan();jobs={};failed={};events=[]
    def start(name,script,arguments=(),gpu=None,python=PYTHON):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        logpath=ROOT/'logs'/f'{name}.log'
        with logpath.open('a') as log:proc=subprocess.Popen([python,'-B','-u',str(ROOT/script),*arguments],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs[name]=dict(process=proc,gpu=gpu,log=str(logpath));events.append(dict(event='start',job=name,pid=proc.pid,gpu=gpu,unix=time.time()));print('START',name,proc.pid,flush=True)
    def wait_all(phase):
        while jobs:
            for name,job in list(jobs.items()):
                rc=job['process'].poll()
                if rc is None:continue
                events.append(dict(event='exit',job=name,returncode=rc,unix=time.time()));print('EXIT',name,rc,flush=True)
                if rc:failed[name]=dict(returncode=rc,log=job['log'])
                del jobs[name]
            dump(ROOT/'pipeline_status.json',dict(status='RUNNING' if not failed else 'RUNNING_WITH_FAILURES',phase=phase,pid=os.getpid(),updated_unix=time.time(),
                source_completed=len(list((ROOT/'parts/source').glob('*.json'))),source_total=4111,
                motion_completed=len(list((ROOT/'parts/motion').glob('*.json'))),motion_total=4111,
                codec_completed=len(list((ROOT/'parts/codec').glob('*/qp*.json'))),codec_total=2373,
                active={k:dict(pid=v['process'].pid,gpu=v['gpu'],log=v['log']) for k,v in jobs.items()},failures=failed))
            if jobs:time.sleep(20)
        dump(ROOT/'parallel_execution_audit.json',dict(status='PASS' if not failed else 'FAIL',events=events,failures=failed,gpus=[4,5,6,7],one_heavy_worker_per_gpu=True))
        assert not failed,failed
    source_done=(ROOT/'source_progress.json').exists() and load(ROOT/'source_progress.json')['status']=='PASS'
    if not source_done:start('source_stats','source_stats.py',('--workers','12'))
    for gpu in (4,5,6,7):
        done=ROOT/'parts'/f'motion_gpu{gpu}_done.json'
        if not done.exists():start(f'motion_gpu{gpu}','gpu_worker.py',('--phase','motion','--gpu',str(gpu)),gpu)
    wait_all('source_motion_embeddings')
    if not (ROOT/'gpu_sharding_audit.json').exists():start('select_codec','select_codec.py');wait_all('codec_stratification')
    for gpu in (4,5,6,7):
        if not (ROOT/'parts'/f'codec_gpu{gpu}_done.json').exists():start(f'codec_gpu{gpu}','gpu_worker.py',('--phase','codec','--gpu',str(gpu)),gpu)
    wait_all('real_RANS_codec')
    start('aggregate','aggregate.py');wait_all('statistics_coverage_aggregation')
    start('plots','plot_statistics.py',python=PLOT_PYTHON);wait_all('plots')
    start('finalize','finalize.py');wait_all('final_integrity')
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    dump(ROOT/'pipeline_status.json',dict(status='PASS',pid=os.getpid(),updated_unix=time.time()));print('DISTRIBUTION AUDIT COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'pipeline_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),unix=time.time()))
        dump(ROOT/'final_integrity.json',dict(status='FAIL',reason='See pipeline_failure.json and logs'));raise
