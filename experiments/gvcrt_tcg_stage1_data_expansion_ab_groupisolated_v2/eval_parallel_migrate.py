"""Persistent safe-boundary benchmark, publication, then existing-training continuation."""
from eval_parallel_common import *
import traceback

def heartbeat(stage,**extra):
    r=dict(status='RUNNING',stage=stage,updated=time.time(),protocol_revision=V3,evaluation_addendum=ADDENDUM,controller_PID=os.getpid(),formal_training_paused=True,arms={a:dict(committed_additional_updates=committed(a),absolute_optimizer_step=5000+committed(a)) for a in ARMS},**extra)
    dump(ROOT/'run_status.json',r);dump(ROOT/'logs/live_status_v3.json',r)
    (ROOT/'REPORT.md').write_text('# TCG evaluation scheduling migration — execution snapshot\n\nNot final integrity acceptance.\n\n```json\n'+json.dumps(r,indent=2)+'\n```\n\nTraining protocol, data plan and early stopping remain unchanged. Paused checkpoints and code provenance: migrations/eval_parallel_v1/.\n')

def run(script,logname,*args):
    with (ROOT/'logs'/logname).open('a') as handle:
        command=[PYTHON,'-B','-u',str(ROOT/script),*args];process=subprocess.Popen(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            progress=load(EP/'probe_progress.json') if (EP/'probe_progress.json').exists() else None
            heartbeat('evaluation_parallel_benchmark' if script=='eval_parallel_probe.py' else 'evaluation_parallel_code_publication',active_PID=process.pid,active_command=command,active_log='logs/'+logname,benchmark_progress=progress);time.sleep(15)
        return process.returncode

def main():
    verify_parallel_sources();assert load(EP/'selftest.json')['status']=='PASS'
    with lock('eval_parallel_migration.lock'):
        for name in ('runner.lock','runner_v3.lock','evaluation_v3_workers.lock'):
            with (ROOT/name).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (EP/'benchmark.json').exists() or load(EP/'benchmark.json')['status']!='PASS':
            assert run('eval_parallel_probe.py','eval_parallel_benchmark.log')==0,'EVALUATOR_BENCHMARK_FAILED'
        benchmark=load(EP/'benchmark.json');assert benchmark['correctness_status']=='PASS'
        for a,r in load(EP/'safe_pause.json')['checkpoints'].items():assert sha(latest_path(a))==r['sha256'] and committed(a)==r['additional_step']
        published=EP/'publication_code.json'
        if not published.exists() or load(published)['status']!='PUBLISHED':
            code=run('eval_parallel_publish.py','eval_parallel_publication_code.log','--phase','code')
            if code:event('EVALUATION_PARALLEL_PUBLICATION_FAILED_LOCAL_CONTINUATION',log='logs/eval_parallel_publication_code.log')
        with (ROOT/'logs/launcher_eval_parallel.log').open('a') as log:
            command=[PYTHON,'-B','-u',str(ROOT/'launch_eval_parallel.py'),'--resume'];process=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(EP/'actual_resume.json',dict(status='STARTED',PID=process.pid,command=command,started_unix=time.time(),log='logs/launcher_eval_parallel.log',selected_workers_per_GPU=benchmark['selected_workers_per_gpu'],resume_checkpoints=load(EP/'safe_pause.json')['checkpoints'],no_optimizer_update_replayed=True))
        time.sleep(2);assert process.poll() is None,('CONTROLLER_EXITED_DURING_START',process.returncode)
        dump(EP/'migration_status.json',dict(status='RUNTIME_RESUMED',controller_PID=process.pid,selected_workers_per_GPU=benchmark['selected_workers_per_gpu'],research_experiment_complete=False))

if __name__=='__main__':
    try:main()
    except Exception:
        error=dict(status='FAILED',stage='evaluation_parallel_migration',error=traceback.format_exc(),updated=time.time(),resume_command=f'{PYTHON} -B -u {ROOT}/eval_parallel_migrate.py');dump(EP/'migration_status.json',error);dump(ROOT/'run_status.json',error);traceback.print_exc();sys.exit(1)
