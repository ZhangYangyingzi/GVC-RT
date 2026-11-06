"""Resume only missing engineering probes; never launch an additional training line."""
from common_s2 import *
import traceback
def wave(jobs):
    ps=[]
    for mode,rank in jobs:
        target=ROOT/('audits/branch_precision_and_state.json' if mode=='layers' else 'audits/gradient_equivalence.json' if mode=='gradient' else f'audits/{mode}.json' if mode.startswith('rans_') else f'audits/forward_{mode}.json')
        if target.exists() and load(target).get('status')=='PASS':continue
        f=(ROOT/f'logs/engineering_{mode}.log').open('a')
        p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'engineering.py'),'--mode',mode,'--rank',str(rank)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        ps.append((p,f,mode));event('ENGINEERING_STARTED',mode=mode,PID=p.pid,physical_GPU=4+rank)
    codes=[]
    for p,f,mode in ps:
        code=p.wait();f.close();codes.append(code);event('ENGINEERING_EXIT',mode=mode,exit_code=code)
    assert not any(codes),('ENGINEERING_FAILURE',codes)
def main():
    env()
    with lock('engineering.lock'):
        wave([('layers',0),('gradient',1)])
        wave([('reference',0),('reference_repeat',1),('zero',2),('enabled',3)])
        subprocess.run([PYTHON,'-B','-u',str(ROOT/'engineering.py'),'--mode','compare'],cwd=ROOT,check=True)
        wave([('rans_matched',0),('rans_zero',1),('rans_enabled',2)])
        event('ENGINEERING_FORWARD_GRADIENT_PASS',formal_joint_updates=0)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
