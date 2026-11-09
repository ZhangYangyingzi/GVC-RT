"""Watch an existing runner; recover terminal interruption without duplicate jobs.

Never signal or terminate any process. Never retry an explicitly FAILED or
finished experiment. A replacement uses the existing runner/checkpoint rules.
"""
from pathlib import Path
import sys
ROOT_PATH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_PATH))
from common_s2 import *

def alive(pid,script=None):
    try:
        p=Path('/proc')/str(int(pid))
        state=p.joinpath('stat').read_text().split(') ')[1].split()[0]
        args=[x.decode(errors='replace') for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]
        return state not in ['Z','X'] and p.joinpath('cwd').resolve()==ROOT and (script is None or any(Path(a).name==script for a in args))
    except (OSError,ValueError,TypeError):return False

def held(name):
    try:
        with lock(name):return False
    except BlockingIOError:return True

def main():
    env()
    with lock('ops/runtime_guard.lock'):
        previous=load(ROOT/'ops/runtime_guard_status.json') if (ROOT/'ops/runtime_guard_status.json').exists() else {}
        restarts=previous.get('automatic_restarts',0)
        while True:
            status_file=ROOT/'status.json'
            if not status_file.exists():raise RuntimeError('TASK_STATUS_MISSING_NO_AUTOMATIC_START')
            current=load(status_file);state=current.get('status');pid=current.get('PID')
            record=dict(status='WATCHING',PID=os.getpid(),watched_runner_PID=pid,watched_stage=current.get('current_stage'),automatic_restarts=restarts,time=time.time(),
                no_processes_signaled=True,guard_source_SHA256=sha(Path(__file__)))
            if state in ['COMPLETE','PARTIAL','FAILED','BLOCKED']:
                record.update(status='STOPPED',reason='Explicit task terminal state; no blind restart',task_status=state)
                dump(ROOT/'ops/runtime_guard_status.json',record);return
            if alive(pid,'runner.py') or held('runner.lock'):
                dump(ROOT/'ops/runtime_guard_status.json',record);time.sleep(15);continue
            live_children=[x.get('pid') for x in current.get('children',{}).values() if alive(x.get('pid'))]
            held_locks=[str(p.relative_to(ROOT)) for p in (ROOT/'locks').glob('*.lock') if held(str(p.relative_to(ROOT)))]
            if live_children or held_locks:
                record.update(status='WAITING_EXISTING_WORKERS',existing_worker_PIDs=live_children,held_locks=held_locks)
                dump(ROOT/'ops/runtime_guard_status.json',record);time.sleep(15);continue
            if state!='RUNNING' or restarts>=1:
                record.update(status='STOPPED',reason='Recovery conditions not satisfied; no restart')
                dump(ROOT/'ops/runtime_guard_status.json',record);return
            from freeze import verify
            verify()
            timestamp=time.strftime('%Y%m%d_%H%M%S');log=ROOT/'logs'/f'guard_resume_{timestamp}.log'
            environment=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
            environment.pop('MODEL_KIND',None)
            with log.open('ab') as output:
                child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'runner.py')],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            restarts+=1;record.update(status='INTERRUPTED_RUNNER_RECOVERED',automatic_restarts=restarts,replacement_PID=child.pid,log=str(log),
                reason='Prior status RUNNING but runner absent and all known worker/lifecycle locks released; resume atomic checkpoint without replay')
            dump(ROOT/'ops/runtime_guard_status.json',record);dump(ROOT/f'audits/runtime_guard_recovery_{timestamp}.json',record)
            time.sleep(15)

if __name__=='__main__':main()
