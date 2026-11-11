"""Task-local durable training supervisor and progress snapshot."""
from common_s2 import *
import traceback


class AdoptedTrainer:
    def __init__(self,pid,group):self.pid=pid;self.group=group;self.returncode=None
    def poll(self):
        p=Path(f'/proc/{self.pid}/cmdline')
        if p.exists() and ('train.py --group '+self.group) in p.read_bytes().replace(b'\0',b' ').decode():return None
        self.returncode=0 if (ROOT/self.group/'training/completion.json').exists() else 1
        return self.returncode


def main():
    env();children={};start=time.time()
    with lock('locks/supervisor.lock'):
        while True:
            checks=all((ROOT/f).exists() and load(ROOT/f).get('status')=='PASS'
                for f in ['audits/preflight.json','gradient_calibration.json'])
            updates={}
            for group in GROUPS:
                p=ROOT/group/'training/progress.json';updates[group]=load(p)['committed_updates'] if p.exists() else 0
                if checks and updates[group]<100 and group not in children:
                    old=ROOT/'queue/train_PIDs.json'
                    previous=load(old).get(group,{}) if old.exists() else {}
                    pid=previous.get('PID');cmd=Path(f'/proc/{pid}/cmdline') if pid else None
                    if cmd and cmd.exists() and ('train.py --group '+group) in cmd.read_bytes().replace(b'\0',b' ').decode():
                        child=AdoptedTrainer(pid,group);log=(ROOT/f'logs/train_{group}.log').open('ab');children[group]=(child,log)
                        event('existing_training_adopted',group=group,PID=pid)
                        continue
                    log=(ROOT/f'logs/train_{group}.log').open('ab')
                    child=subprocess.Popen([PYTHON,'-B','-u','train.py','--group',group],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GROUPS.index(group))),
                        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    children[group]=(child,log);event('training_started',group=group,PID=child.pid,physical_GPU=GROUPS.index(group))
                if group in children:
                    child,log=children[group]
                    if child.poll() not in (None,0):
                        dump(ROOT/'supervisor_failure.json',dict(group=group,returncode=child.returncode,committed_updates=updates[group],time=time.time()))
                        raise RuntimeError(('TRAIN_PROCESS_FAILED',group,child.returncode))
            evaluations={}
            for model,s in [('Original',0),('B300',0),*[(g,s) for g in GROUPS for s in (50,100)]]:
                evaluations[f'{model}_step{s}']=len(list((ROOT/f'evaluation/development/{model}/step_{s:04d}/points').glob('*.json')))
            stage='training' if checks else 'calibration_and_preflight'
            if all(u==100 for u in updates.values()):stage='development_evaluation'
            dump(ROOT/'status.json',dict(status='RUNNING',stage=stage,experiment_directory=str(ROOT),
                committed_updates=updates,evaluation_completed_points=evaluations,expected_points_per_state=96,
                cache_h_windows=len(list((ROOT/'cache/h').glob('*.pt'))),
                processes={g:dict(PID=p.pid,physical_GPU=GROUPS.index(g),exitcode=p.poll()) for g,(p,log) in children.items()},
                supervisor_PID=os.getpid(),elapsed_seconds=time.time()-start,time=time.time(),complete=False))
            dump(ROOT/'queue/train_PIDs.json',{g:dict(PID=p.pid,physical_GPU=GROUPS.index(g),exitcode=p.poll()) for g,(p,log) in children.items()})
            if all(u==100 for u in updates.values()) and all(p.poll()==0 for p,log in children.values()):
                dump(ROOT/'training_completion.json',dict(status='PASS',committed_updates=updates,windows_per_group=400,time=time.time()));break
            time.sleep(5)

if __name__=='__main__':
    try:main()
    except Exception as e:env();dump(ROOT/'audits/supervisor_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
