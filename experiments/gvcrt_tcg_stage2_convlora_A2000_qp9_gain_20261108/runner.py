"""Persistent QP9-only inference runner, source audit -> smoke ->32points -> publish."""
from common_s2 import *
from audit import *
import signal,traceback

def update(stage,state='RUNNING',**extra):
    points=len(list((ROOT/'evaluation/s05/points').glob('*.json')))
    dump(ROOT/'status.json',dict(status=state,complete=state=='COMPLETE',current_stage=stage,PID=os.getpid(),formal_completed_points=points,
        expected_points=32,external_QP=9,inference_gain=.5,training_updates=0,checkpoint_sha256=CHECKPOINT_SHA,time=time.time(),**extra))

def jobs(items,stage,lease):
    update(stage);children=[]
    for label,cmd in items:
        f=(ROOT/f'logs/{label}.log').open('a');p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7'),
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
        children.append((label,p,f));event('CHILD_STARTED',stage=stage,label=label,PID=p.pid,command=cmd)
    dump(ROOT/'active_processes.json',dict(stage=stage,runner_PID=os.getpid(),children=[dict(label=n,PID=p.pid) for n,p,f in children],time=time.time()))
    heartbeat=0
    while any(p.poll() is None for n,p,f in children):
        if any(p.poll() is not None and p.returncode for n,p,f in children):
            for n,p,f in children:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            break
        if time.time()-heartbeat>20:update(stage);heartbeat=time.time()
        time.sleep(2)
    for n,p,f in children:
        try:code=p.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        f.close();event('CHILD_EXIT',stage=stage,label=n,PID=p.pid,exit_code=code)
    assert all(p.returncode==0 for n,p,f in children),('CHILD_FAILED',stage,[(n,p.returncode) for n,p,f in children])

def run(lease):
    env();verify_sources();disk()
    if not (ROOT/'final_integrity.json').exists() or load(ROOT/'final_integrity.json').get('status')!='PASS':
        jobs([('resource_preflight',[PYTHON,'-B','-u',str(ROOT/'resource_preflight.py')])],'GPU_preflight',lease)
        if not (ROOT/'audits/smoke.json').exists():
            tasks=[]
            for rank,g in enumerate([0.,1.,.5]):tasks.append((f'smoke_s{g}',[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--rank',str(rank),'--gain',str(g),'--smoke']))
            jobs(tasks,'endpoint_smoke',lease);smoke_check()
        assert load(ROOT/'audits/smoke.json')['status']=='PASS'
        reuse_endpoints()
        jobs([(f'formal_rank{rank}',[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--rank',str(rank),'--gain','0.5']) for rank in range(4)],'formal_s05_QP9',lease)
        jobs([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')])],'final_integrity_and_plots',lease)
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    if not (ROOT/'publication_status.json').exists() or load(ROOT/'publication_status.json')['status']!='PUBLISHED':
        jobs([('publish',[PYTHON,'-B','-u',str(ROOT/'publish.py')])],'publication',lease)
    assert load(ROOT/'publication_status.json')['status']=='PUBLISHED'
    update('complete','COMPLETE',integrity_status='PASS',publication_status='PUBLISHED',remote_verified=True)
    dump(ROOT/'active_processes.json',dict(stage='COMPLETE',runner_PID=os.getpid(),children=[],time=time.time()))

def main():
    env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],time=time.time()))
        try:run(lease)
        except Exception:
            error=traceback.format_exc();stage=load(ROOT/'status.json').get('current_stage','source_preflight') if (ROOT/'status.json').exists() else 'source_preflight'
            update(stage,'FAILED',failure_stage=stage,error=error,local_evaluation_complete=(ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS',
                resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            event('RUNNER_FAILED',stage=stage,error=error);raise

if __name__=='__main__':main()
