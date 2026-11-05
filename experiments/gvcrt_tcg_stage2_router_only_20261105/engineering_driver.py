from common_s2 import *
import traceback
def wave(jobs):
    running=[]
    for mode,rank in jobs:
        dest=ROOT/('audits/gradient_and_state.json' if mode=='gradient' else f'audits/forward_{mode}.json')
        if dest.exists() and load(dest).get('status')=='PASS':continue
        handle=(ROOT/f'logs/probe_{mode}.log').open('a')
        child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'engineering.py'),'--mode',mode,'--rank',str(rank)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        running.append((child,handle,mode));event('ENGINEERING_PROBE_STARTED',mode=mode,PID=child.pid,physical_GPU=rank+4)
    codes=[]
    for child,handle,mode in running:
        code=child.wait();handle.close();codes.append(code);event('ENGINEERING_PROBE_EXIT',mode=mode,exit_code=code)
    assert not any(codes),('ENGINEERING_PROBE_FAILED',codes)
def main():
    env()
    with lock('engineering.lock'):
        status('engineering_validation',PID=os.getpid(),formal_training_started=False,committed_router_updates=0)
        wave([('reference',0)])
        wave([('reference_repeat',0),('bypass',1),('zero',2),('one',3)])
        wave([('off',0),('enabled',1),('gradient',2)])
        subprocess.run([PYTHON,'-B','-u',str(ROOT/'engineering.py'),'--mode','compare'],cwd=ROOT,check=True)
        event('FORWARD_AND_GRADIENT_ENGINEERING_PASS')
if __name__=='__main__':
    try:main()
    except Exception:
        status('engineering_validation','FAILED',error=traceback.format_exc(),formal_training_started=False,committed_router_updates=0);raise
