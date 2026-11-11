"""Restore missing task-owned workers; trainers reload optimizer/RNG/data cursor."""
from common_s2 import *


def alive(pid,expected):
    try:
        p=Path(f'/proc/{pid}')
        return expected in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode() and (p/'cwd').resolve()==ROOT
    except (FileNotFoundError,ProcessLookupError):return False


def start(name,script,gpu=None):
    log=(ROOT/f'logs/{name}.log').open('ab');e=dict(os.environ)
    if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    p=subprocess.Popen([PYTHON,'-B','-u',script],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    return dict(PID=p.pid,physical_GPU=gpu,command=script,log=str(log.name),time=time.time())


def repair_commit_metadata(group):
    """Recover a milestone/receipt if interrupted after atomic latest publication."""
    import torch
    root=ROOT/group/'training';latest=root/'latest.pt'
    if not latest.exists():return
    registered=load(ROOT/'queue/train_PIDs.json') if (ROOT/'queue/train_PIDs.json').exists() else {}
    if alive(registered.get(group,{}).get('PID',0),'train.py --group '+group):return
    saved=torch.load(latest,map_location='cpu',weights_only=True)
    assert saved['group']==group and saved['protocol_SHA256']==sha(ROOT/'protocol.json')
    step=saved['committed_updates'];assert saved['next_cursor']==4*step
    if step in (0,25,50,75,100):
        target=root/f'checkpoints/step_{step:04d}.pt'
        if not target.exists():atomic_torch(target,saved)
    if step:dump(root/f'receipts/{step:04d}.json',saved['receipt'])
    dump(root/'progress.json',dict(status='RESUMABLE',committed_updates=step,next_cursor=4*step,
        physical_GPU=GROUPS.index(group),recovered_from_atomic_latest=True,time=time.time()))


def main():
    env();assert Path.cwd()==ROOT
    if (ROOT/'queue/STOP').exists():raise RuntimeError('TASK_MARKED_STOPPED_OR_COMPLETE; inspect status before resuming')
    for group in GROUPS:repair_commit_metadata(group)
    caches=load(ROOT/'queue/cache_PIDs.json') if (ROOT/'queue/cache_PIDs.json').exists() else dict(children={})
    done=(ROOT/'training_completion.json').exists()
    if not done:
        for gpu in (2,3):
            old=caches['children'].get(str(gpu),{})
            if not alive(old.get('PID',0),'cache_pipeline.py'):caches['children'][str(gpu)]=start(f'cache_pipeline_GPU{gpu}','cache_pipeline.py',gpu)
        dump(ROOT/'queue/cache_PIDs.json',caches)
        if not (ROOT/'gradient_calibration.json').exists():
            old=load(ROOT/'queue/check_PIDs.json');j=old.get('calibrate',{})
            if not alive(j.get('PID',0),'calibrate.py'):old['calibrate']=start('calibrate','calibrate.py',1);dump(ROOT/'queue/check_PIDs.json',old)
        if not (ROOT/'audits/preflight.json').exists():
            old=load(ROOT/'queue/check_PIDs.json');j=old.get('preflight',{})
            if not alive(j.get('PID',0),'preflight.py'):old['preflight']=start('preflight','preflight.py',0);dump(ROOT/'queue/check_PIDs.json',old)
        old=load(ROOT/'supervisor_PID.json')
        if not alive(old.get('PID',0),'supervisor.py'):dump(ROOT/'supervisor_PID.json',start('supervisor','supervisor.py'))
    if not (ROOT/'evaluation_completion.json').exists():
        old=load(ROOT/'queue/evaluation_PID.json') if (ROOT/'queue/evaluation_PID.json').exists() else {}
        if not alive(old.get('PID',0),'evaluation_queue.py'):dump(ROOT/'queue/evaluation_PID.json',start('evaluation_queue','evaluation_queue.py','2,3'))
    event('resume_checked',training_complete=done)

if __name__=='__main__':main()
