"""Resumable evaluator for the fixed v6.34a screen; does not train models."""
from io32 import *
import fcntl, subprocess, time, os

POOL=(6,7)
RESERVE_MIB=8000
RETRY_LIMIT=2

def gpu_memory():
    out=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):{'free_MiB':int(b.strip()),'used_MiB':int(c.strip()),'utilization_percent':int(d.strip())} for a,b,c,d in (line.split(',') for line in out.strip().splitlines())}

def task_key(t): return f"{t['stage']}_{t['dataset']}_{t['video']}_{t['method']}_{t['qp']}"

def main():
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=load(ROOT/'task_manifest.json')['tasks']
    from evaluate32 import valid
    state_path=ROOT/'driver_state.json'
    state=load(state_path) if state_path.exists() else {'attempts':{},'failed':{},'completed':[]}
    state.setdefault('attempts',{});state.setdefault('failed',{})
    done={task_key(t) for t in tasks if valid(t)}
    running={};schedule=ROOT/'gpu_schedule.jsonl'
    # Adopt a worker if the controller was interrupted while its child kept
    # running. Check both PID liveness and the exact command task payload.
    for key,rec in list(state.get('running',{}).items()):
        pid=int(rec.get('pid',-1));cmd=process_command(pid)
        payload=json.dumps(rec.get('task',{}),separators=(',',':'))
        if cmd and str(ROOT/'evaluate32.py') in cmd and payload in cmd:
            running[key]={**rec,'process':None,'log_handle':None,'external':True}
    pending=[t for t in tasks if task_key(t) not in done]
    if not (ROOT/'audits/sanity.json').exists() and pending:
        t=pending[0];key=task_key(t)
        if key not in running:
            pending.pop(0);mem=gpu_memory();gpu=max(POOL,key=lambda g:mem[g]['free_MiB'])
            launch(t,gpu,running,schedule,state_path,state,done)
        while key in running:
            reap(running,schedule,state_path,state,done);time.sleep(2)
        if key not in done:
            state.update(status='FAILED_SANITY',expected=90,completed=sorted(done),updated=time.time());dump(state_path,state);return
        dump(ROOT/'audits/sanity.json',dict(status='PASS',task=t,receipt=str(ROOT/'tasks'/(task_key(t)+'.json')),time=time.time(),real_RANS=True,independent_decode=True,source_frame_hash_checks=True))
    while True:
        # Recompute pending from the frozen task list and durable retry counts.
        running_keys=set(running)
        pending=[t for t in tasks if task_key(t) not in done and task_key(t) not in running_keys and state['attempts'].get(task_key(t),0)<=RETRY_LIMIT]
        mem=gpu_memory()
        for gpu in POOL:
            reserved=sum(v['reserve_MiB'] for v in running.values() if v['gpu']==gpu)
            while pending and mem[gpu]['free_MiB']-reserved>=RESERVE_MIB:
                t=pending.pop(0);key=task_key(t);launch(t,gpu,running,schedule,state_path,state,done);reserved+=RESERVE_MIB
        reap(running,schedule,state_path,state,done)
        pending=[t for t in tasks if task_key(t) not in done and task_key(t) not in running and state['attempts'].get(task_key(t),0)<=RETRY_LIMIT]
        state.update(status='RUNNING',completed=sorted(done),expected=90,pending=len(pending),active=len(running),updated=time.time());dump(state_path,state)
        if not running and not pending: break
        time.sleep(3 if running else 15)
    failed={k:v for k,v in state['failed'].items() if state['attempts'].get(k,0)>RETRY_LIMIT}
    state.update(status='PASS' if len(done)==90 and not failed else 'FAILED',completed=sorted(done),expected=90,failed=failed,updated=time.time());dump(state_path,state)
    if state['status']=='PASS': subprocess.check_call([PYTHON,str(ROOT/'aggregate.py')],cwd=REPO)

def launch(t,gpu,running,schedule,state_path,state,done):
    key=task_key(t);attempt=state['attempts'].get(key,0);log=ROOT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    cmd=[PYTHON,'-B','-u',str(ROOT/'evaluate32.py'),'--task',json.dumps(t,separators=(',',':')),'--gpu',str(gpu)]
    f=log.open('ab');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rec={'task':t,'gpu':gpu,'pid':p.pid,'reserve_MiB':RESERVE_MIB,'attempt':attempt,'started':time.time(),'process':p,'log_handle':f,'log':str(log)};running[key]=rec
    append(schedule,dict(event='start',task_id=key,physical_gpu=gpu,pid=p.pid,command=cmd,started=rec['started'],free_before_MiB=gpu_memory()[gpu]['free_MiB'],reserved_MiB=RESERVE_MIB,retry=attempt,log=str(log)))
    state.setdefault('running',{})[key]={k:v for k,v in rec.items() if k not in ('process','log_handle')};dump(state_path,state)

def reap(running,schedule,state_path,state,done):
    from evaluate32 import valid
    for key,rec in list(running.items()):
        if rec.get('process') is None:
            if process_command(rec['pid']): continue
            rc=0 if valid(rec['task']) else 255
        else: rc=rec['process'].poll()
        if rc is None: continue
        if rec.get('log_handle') is not None:rec['log_handle'].close()
        ok=rc==0 and valid(rec['task'])
        append(schedule,dict(event='finish',task_id=key,physical_gpu=rec['gpu'],pid=rec['pid'],ended=time.time(),exit_code=rc,status='PASS' if ok else 'FAIL',retry=rec['attempt'],reserved_MiB=rec['reserve_MiB']))
        state.setdefault('running',{}).pop(key,None)
        if ok:done.add(key);state['failed'].pop(key,None)
        else:
            state['attempts'][key]=rec['attempt']+1
            state['failed'][key]=dict(attempts=rec['attempt']+1,exit_code=rc,log=rec['log'],reason='worker exit or artifact signature validation failure')
        running.pop(key)
    state.update(completed=sorted(done),updated=time.time());dump(state_path,state)

def process_command(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2]=='Z':return None
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except OSError:return None

if __name__=='__main__': main()
