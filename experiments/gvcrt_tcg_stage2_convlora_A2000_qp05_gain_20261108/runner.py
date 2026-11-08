"""Persistent64-point QP0/5 runner, audited QP9 reuse, local integrity and publication."""
from common_s2 import *
from audit import *
import signal,traceback

def update(stage,state='RUNNING',**extra):
    counts={q:0 for q in [0,5]}
    for path in (ROOT/'evaluation/s05/points').glob('*.json'):
        p=load(path);q=int(p['result']['QP']);assert q in counts;counts[q]+=1
    dump(ROOT/'status.json',dict(status=state,complete=state=='COMPLETE',current_stage=stage,PID=os.getpid(),runner_running=state=='RUNNING',
        formal_completed_points=sum(counts.values()),QP0_completed=counts[0],QP5_completed=counts[5],expected_points=64,new_external_QPs=[0,5],
        inference_gain=.5,training_updates=0,checkpoint_SHA256=CHECKPOINT_SHA,time=time.time(),**extra))

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
        if time.time()-heartbeat>20:
            dump(ROOT/'heartbeat.json',dict(stage=stage,runner_PID=os.getpid(),time=time.time()))
            if stage!='publication':update(stage)
            heartbeat=time.time()
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
            tasks=[(0,0.),(0,.5),(0,1.),(5,0.),(5,.5),(5,1.)]
            for offset in range(0,6,4):
                jobs([(f'smoke_qp{q}_s{g:g}',[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--rank',str(i),'--gain',str(g),'--smoke-qp',str(q)])
                    for i,(q,g) in enumerate(tasks[offset:offset+4])],'endpoint_smoke_round'+str(offset//4+1),lease)
            smoke_check()
        assert load(ROOT/'audits/smoke.json')['status']=='PASS'
        reuse_sources()
        jobs([(f'formal_rank{rank}',[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--rank',str(rank),'--gain','0.5']) for rank in range(4)],'formal_s05_QP0_QP5',lease)
        jobs([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')])],'final_integrity_and_plots',lease)
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    pub=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
    if pub.get('status')!='PUBLISHED' or not (ROOT/'publication_remote_receipt.json').exists():
        jobs([('publish',[PYTHON,'-B','-u',str(ROOT/'publish.py')])],'publication',lease)
    assert load(ROOT/'publication_status.json')['status']=='PUBLISHED' and load(ROOT/'publication_remote_receipt.json')['remote_verified']
    assert load(ROOT/'status.json')['status']=='COMPLETE'

def main():
    env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],time=time.time()))
        try:run(lease)
        except Exception:
            error=traceback.format_exc();stage=load(ROOT/'status.json').get('current_stage','source_preflight') if (ROOT/'status.json').exists() else 'source_preflight'
            update(stage,'FAILED',failure_stage=stage,error=error,local_evaluation_complete=(ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS',
                resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            event('RUNNER_FAILED',stage=stage,error=error);raise

if __name__=='__main__':main()
