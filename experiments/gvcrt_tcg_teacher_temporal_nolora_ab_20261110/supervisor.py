"""Detached task owner; no duplicate runs and no restarting completed training."""
from common_s2 import *
import argparse,traceback

def serve():
    env()
    with lock('locks/supervisor.lock'):
        dump(ROOT/'supervisor.json',dict(status='RUNNING',PID=os.getpid(),time=time.time()))
        if (ROOT/'cache/STOP').exists() and load(ROOT/'status.json').get('status')!='COMPLETE':(ROOT/'cache/STOP').unlink()
        if load(ROOT/'status.json').get('status')=='COMPLETE':
            dump(ROOT/'supervisor.json',dict(status='COMPLETE',PID=os.getpid(),already_complete=True,time=time.time()));return
        # A live runner from a previous supervisor remains its sole controller.
        try:
            with lock('locks/runner.lock'):pass
        except BlockingIOError:raise RuntimeError('RUNNER_ALREADY_ACTIVE_NO_DUPLICATE')
        log=(ROOT/'logs/runner.log').open('ab');p=subprocess.Popen([PYTHON,'-B','-u','runner.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            dump(ROOT/'supervisor.json',dict(status='RUNNING',PID=os.getpid(),runner_PID=p.pid,time=time.time()));time.sleep(10)
        log.close()
        if p.returncode:
            status=load(ROOT/'status.json');status.update(status='FAILED',complete=False,runner_exit_code=p.returncode,resume_command=f'{PYTHON} -B {ROOT}/supervisor.py --start',time=time.time());dump(ROOT/'status.json',status)
        dump(ROOT/'supervisor.json',dict(status='COMPLETE' if p.returncode==0 else 'FAILED',PID=os.getpid(),runner_PID=p.pid,exit_code=p.returncode,time=time.time()))

def main():
    env();p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');a=p.parse_args()
    if a.start:
        try:
            with lock('locks/supervisor.lock'):pass
        except BlockingIOError:print(json.dumps(load(ROOT/'supervisor.json')));return
        f=(ROOT/'logs/supervisor.log').open('ab');p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'supervisor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3'));f.close();print(json.dumps(dict(started_PID=p.pid)))
    else:serve()
if __name__=='__main__':main()
