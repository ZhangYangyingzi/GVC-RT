"""Drain only this experiment's active points before replacing its dispatcher."""
from preflight import *
from data import load
from evaluation_stage import expected_signatures,receipt_valid

def process_state(pid):
    p=Path(f'/proc/{pid}/status')
    if not p.exists():return None
    return next(s.split()[1] for s in p.read_text().splitlines() if s.startswith('State:'))

def main():
    import argparse,fcntl,signal
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
    pid=parser.parse_args().pid
    lock=(R/'evaluation_handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert load(R/'status.json').get('PID')==pid and load(R/'status.json').get('phase')=='evaluation'
    assert str(R/'run_pipeline.py') in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    signatures=expected_signatures();paused=False
    previous=R/'audits/evaluation_dispatcher_handoff.json'
    if previous.exists():save(f'audits/evaluation_dispatcher_handoff_history_{time.time_ns()}.json',load(previous))
    audit=dict(status='DRAINING',old_driver=pid,training_restarted=False,point_work_interrupted=False,numerical_changes=False,started=time.time())
    try:
        os.kill(pid,signal.SIGSTOP);paused=True
        stop_deadline=time.time()+10
        while process_state(pid)!='T':
            assert time.time()<stop_deadline,'Dispatcher did not reach stopped state'
            time.sleep(.1)
        state=load(R/'evaluation_scheduler.json');tasks={w['PID']:w['task'] for w in state.get('running',[])}
        children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
        for child in children:
            command=Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if process_state(child)!='Z':
                if command[0].endswith(b'nvidia-smi'):continue
                assert str(R/'evaluate.py').encode() in command,(child,command)
                task=json.loads(command[command.index(b'--task')+1]);tasks[child]=task
            elif child not in tasks:
                # A short read-only nvidia-smi child may already be reaped.
                assert process_state(child) in (None,'Z')
        audit.update(draining_children=tasks,previous_scheduler=state)
        save('audits/evaluation_dispatcher_handoff.json',audit)
        deadline=time.time()+1800
        while any(process_state(child) not in (None,'Z') for child in children):
            assert time.time()<deadline,'Owned points did not drain within 1800 seconds'
            time.sleep(15)
        for child,task in tasks.items():assert receipt_valid(task,signatures),('Drained point failed strict checks',child,task)
        audit.update(drained_points_all_passed=True,completed_point_count=len(list((R/'results/points').glob('*.json'))))
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT);paused=False
        deadline=time.time()+60
        while Path(f'/proc/{pid}').exists():
            assert time.time()<deadline,'Old evaluation dispatcher did not reap'
            time.sleep(1)
        subprocess.run([sys.executable,'-B',str(R/'launch.py')],cwd=GLC,check=True)
        deadline=time.time()+60
        while load(R/'status.json').get('PID') in (None,pid):
            assert time.time()<deadline,'New evaluation dispatcher missing'
            time.sleep(1)
        audit.update(status='PASS',new_driver=load(R/'status.json')['PID'],completed=time.time(),scheduler_sha256=sha(R/'evaluation_stage.py'))
        save('audits/evaluation_dispatcher_handoff.json',audit);print(json.dumps(audit),flush=True)
    except Exception:
        if paused and process_state(pid) is not None:os.kill(pid,signal.SIGCONT)
        audit.update(status='BLOCKED',error=traceback.format_exc());save('audits/evaluation_dispatcher_handoff.json',audit);raise

if __name__=='__main__':main()
