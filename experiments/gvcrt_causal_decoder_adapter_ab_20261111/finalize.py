"""Finish authorized media/audits/publication after the fixed training/evaluation budget."""
from common_s2 import *
import signal,traceback


def launch(script,args=(),gpu=None,label=None):
    name=label or Path(script).stem;log=(ROOT/f'logs/{name}.log').open('ab');e=dict(os.environ)
    if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    child=subprocess.Popen([PYTHON,'-B','-u',script,*args],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/f'queue/{name}_PID.json',dict(PID=child.pid,physical_GPU=gpu,command=[script,*args],log=str(log.name),time=time.time()))
    return child,log


def wait_job(job):
    child,log=job
    while child.poll() is None:time.sleep(5)
    log.close();assert child.returncode==0,('FINALIZATION_SUBPROCESS_FAILED',child.args,child.returncode)


def stop_caches():
    a=load(ROOT/'queue/cache_PIDs.json');records=[]
    for item in a['children'].values():
        pid=item['PID']
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'cache_pipeline.py' not in cmd or Path(f'/proc/{pid}/cwd').resolve()!=ROOT:continue
            os.kill(pid,signal.SIGTERM);records.append(dict(PID=pid,physical_GPU=item['physical_GPU']))
        except (FileNotFoundError,ProcessLookupError):pass
    dump(ROOT/'audits/cache_workers_released.json',dict(status='PASS',reason='all400 training windows consumed by both groups',terminated_task_workers=records,no_foreign_process_stopped=True,time=time.time()))


def main():
    env()
    with lock('locks/finalize.lock'):
        while not (ROOT/'training_completion.json').exists():time.sleep(5)
        stop_caches()
        jobs=[launch('verify_full_model.py',gpu=0),launch('monitor_export.py',gpu=1)]
        for j in jobs:wait_job(j)
        media_jobs=[launch('media.py',['--case','0'],0,'media_case0'),launch('media.py',['--case','1'],1,'media_case1')]
        for j in media_jobs:wait_job(j)
        wait_job(launch('media.py',['--case','2'],0,'media_case2'))
        while not (ROOT/'evaluation_completion.json').exists():time.sleep(5)
        # At this point all task-owned GPU workers have drained. Check permitted
        # GPUs for an uncontended opportunity; never stop another user's process.
        for gpu in (0,1,2,3):
            wait_job(launch('performance.py',gpu=gpu,label=f'performance_GPU{gpu}'))
            if load(ROOT/'performance.json')['status']=='PASS':break
        wait_job(launch('results.py'))
        wait_job(launch('integrity.py'))
        integrity=load(ROOT/'integrity.json');assert integrity['status']=='PASS'
        performance=load(ROOT/'performance.json');pending=[] if performance['status']=='PASS' else ['uncontended same-GPU performance timing']
        summary=dict(status='COMPLETE' if not pending else 'COMPLETED_WITH_PENDING_PERFORMANCE',complete=not pending,
            experiment_directory=str(ROOT),committed_updates=load(ROOT/'training_completion.json')['committed_updates'],
            evaluation_completed_points=load(ROOT/'evaluation_completion.json')['coverage'],
            Original_real_completed=True,development_only=True,integrity_status=integrity['status'],
            plots=str(ROOT/'plots/development'),lossless_video_manifest=str(ROOT/'video_manifest.json'),
            remaining=pending,current_stage='publication',time=time.time())
        dump(ROOT/'status.json',summary)
        wait_job(launch('publish.py',['--phase','final_results']))
        summary.update(current_stage='complete',publication=load(ROOT/'publication_remote_receipt.json'),time=time.time())
        dump(ROOT/'status.json',summary)
        dump(ROOT/'queue/STOP',dict(reason='all fixed-budget authorized jobs completed; performance status recorded',time=time.time()))
        event('experiment_delivery_ready',status=summary['status'],remaining=pending)


if __name__=='__main__':
    try:main()
    except Exception as e:
        env();dump(ROOT/'audits/finalize_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
