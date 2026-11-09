from common_s2 import *

def call(name,args):
    env();log=ROOT/'logs'/f'{name}.log'
    with log.open('ab') as f:
        p=subprocess.Popen([PYTHON,'-B','-u',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        dump(ROOT/f'audits/process_{name}.json',dict(pid=p.pid,args=list(map(str,args)),log=str(log),time=time.time()))
        return p

def main():
    env()
    with lock('locks/engineering_batch.lock'):
        p=call('unmodified_reference',[ROOT/'reference_probe.py']);assert p.wait()==0
        ps=[(g,call('engineering_'+g,[ROOT/'engineering.py','--group',g,'--rank',str(i)])) for i,g in enumerate(GROUPS)]
        passed=[]
        for g,p in ps:
            if p.wait()==0:passed.append(g)
        ps=[(g,call('RANS_'+g,[ROOT/'probe_rans.py','--group',g,'--rank',str(GROUPS.index(g))])) for g in passed]
        for g,p in ps:
            if p.wait()!=0:passed.remove(g)
        p=call('calibration',[ROOT/'calibrate.py']);calibration_ok=p.wait()==0
        dump(ROOT/'audits/engineering_batch.json',dict(status='PASS' if len(passed)==4 and calibration_ok else 'PARTIAL',passed_branches=passed,calibration_passed=calibration_ok,time=time.time()))

if __name__=='__main__':main()
