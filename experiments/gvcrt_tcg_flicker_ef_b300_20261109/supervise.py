"""Task-private CPU watchdog; recover interruptions, never retry recorded failures."""
from common_s2 import *
import signal

def held(name):
    try:
        with lock(name):return False
    except BlockingIOError:return True

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
    except OSError:return False

def launch(name):
    e=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1,2,3')
    with (ROOT/'logs'/f'{name}_supervised.log').open('ab') as f:
        p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/f'{name}.py')],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    event('supervisor_recover_process',process_name=name,PID=p.pid)
    return p.pid

def main():
    env();children={}
    with lock('locks/supervisor.lock'):
        while True:
            status=load(ROOT/'status.json') if (ROOT/'status.json').exists() else dict(status='RUNNING',complete=False)
            receipt=load(ROOT/'publication_remote_receipt.json') if (ROOT/'publication_remote_receipt.json').exists() else {}
            if status.get('complete') and receipt.get('status')=='VERIFIED' and receipt.get('phase')=='results_final':
                assert status['status']=='COMPLETE'
                (ROOT/'cache/STOP').touch()
                dump(ROOT/'supervisor_status.json',dict(status='COMPLETE',complete=True,publication=receipt,heartbeat=time.time(),PID=os.getpid()))
                return
            failure=ROOT/'audits/cache_producer_failure.json'
            if status['status']=='FAILED' or failure.exists():
                dump(ROOT/'supervisor_status.json',dict(status='DIAGNOSTIC_STOP',complete=False,reason='recorded engineering/training/cache failure; automatic retry disabled',runner=status,cache_failure=load(failure) if failure.exists() else None,heartbeat=time.time(),PID=os.getpid()))
                return
            cache_active=held('locks/cache_producer.lock');controller_active=held('runner.lock')
            if not cache_active:
                cache=load(ROOT/'cache/progress.json') if (ROOT/'cache/progress.json').exists() else {}
                # An interrupted controller may leave a cache child finishing an
                # atomic tensor. Drain it before assigning any new GPU worker.
                pending=[a for a in cache.get('active',{}).values() if alive(a['PID'])]
                if not pending:
                    flag=ROOT/'cache/STOP'
                    if flag.exists():flag.unlink()
                    if 'pipeline' not in children or not alive(children['pipeline']):children['pipeline']=launch('pipeline')
            if not controller_active and ('runner' not in children or not alive(children['runner'])):children['runner']=launch('runner')
            # The regular controller adopts live formal chunks through their
            # two-rank process locks. No committed optimizer step is replayed.
            if not held('locks/status_monitor.lock') and ('monitor' not in children or not alive(children['monitor'])):children['monitor']=launch('monitor')
            dump(ROOT/'supervisor_status.json',dict(status='RUNNING',complete=False,PID=os.getpid(),heartbeat=time.time(),controller_active=controller_active,cache_active=cache_active,children=children,recorded_failures_are_not_retried=True,only_this_experiment=True))
            time.sleep(5)
if __name__=='__main__':main()
