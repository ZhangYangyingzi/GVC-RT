"""Reuse the immutable v3 controller; dispatch only evaluation/report/integrity/publication."""
from eval_parallel_common import *
import traceback

def main():
    verify_parallel_sources();import v3_controller as controller
    decision=load(EP/'benchmark.json');assert decision['correctness_status']=='PASS'
    # Orphan evaluation children inherit this lease. Wait instead of training
    # concurrently or starting duplicate evaluation workers after a restart.
    while True:
        with (ROOT/'evaluation_v3_workers.lock').open('a') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(15);continue
        break
    original_command=controller.command;original_heartbeat=controller.heartbeat
    mapping={'v3_evaluate.py':'eval_parallel.py','v3_finalize.py':'eval_parallel_finalize.py','v3_integrity.py':'eval_parallel_integrity.py','v3_publish.py':'eval_parallel_publish.py'}
    def command(script,logname,*args):
        verify_parallel_sources();new=mapping.get(script,script);result=original_command(new,logname,*args)
        if script=='v3_evaluate.py':result[2].update(physical_GPUs=[4,5,6,7],evaluator_processes=decision['selected_evaluator_processes'],workers_per_GPU=decision['selected_workers_per_gpu'],evaluation_addendum=ADDENDUM)
        return result
    def heartbeat(stage,**extra):
        original_heartbeat(stage,**extra);state=load(ROOT/'run_status.json');state.update(evaluation_runtime_addendum=ADDENDUM,evaluation_workers_per_GPU=decision['selected_workers_per_gpu']);dump(ROOT/'run_status.json',state)
        live=load(ROOT/'logs/live_status_v3.json');live.update(evaluation_runtime_addendum=ADDENDUM,evaluation_workers_per_GPU=decision['selected_workers_per_gpu']);dump(ROOT/'logs/live_status_v3.json',live)
        if stage not in ('final_summary','final_integrity','result_publication','complete'):
            with (ROOT/'REPORT.md').open('a') as f:f.write(f'\nActive evaluation scheduling: {ADDENDUM}, {decision["selected_workers_per_gpu"]} worker(s) per GPU. Training source and optimizer protocol unchanged. Audit: migrations/eval_parallel_v1/.\n')
    controller.command=command;controller.heartbeat=heartbeat;controller.verify_sources=verify_parallel_sources
    with lock('eval_parallel_controller.lock'):return controller.main()

if __name__=='__main__':
    try:sys.exit(main())
    except BlockingIOError:print('Same-task controller lease held; no duplicate started.',flush=True)
    except Exception:
        previous=load(ROOT/'run_status.json');previous.update(status='FAILED',error=traceback.format_exc(),updated=time.time());dump(ROOT/'run_status.json',previous);dump(EP/f'controller_failure_{time.time_ns()}.json',previous);traceback.print_exc();sys.exit(1)
