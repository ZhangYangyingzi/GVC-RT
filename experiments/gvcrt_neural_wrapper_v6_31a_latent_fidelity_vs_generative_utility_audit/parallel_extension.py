"""Parallelize future MCL videos without modifying the frozen evaluator.

Only this experiment's original MCL worker is suspended. Its current video
remains its exclusive responsibility. An independent guardian resumes it if
the extension exits unexpectedly. The original driver retains publication.
"""
import argparse
import fcntl
import gc
import signal
import subprocess
import time
import traceback
from audit_io import *

def identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return dict(state=text[0],start_ticks=text[19],command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
    except OSError:return None

def same(pid,ticks):
    p=identity(pid)
    return p is not None and p['state']!='Z' and p['start_ticks']==ticks

def schedule(event):
    with (ROOT/'logs/parallel_gpu_schedule.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**event))+'\n')

def snapshot():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):float(x.split(',')[1]) for x in lines.splitlines() if int(x.split(',')[0]) in (4,5,6,7)}
    used={}
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for x in lines.splitlines():
        try:a,b=x.split(',');used[int(a)]=float(b)
        except ValueError:pass
    return free,used

def worker_main(gpu,video):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    from proxy_runtime import backend,teacher_targets,runtime
    from worker import execute,valid
    backend()
    import torch
    lock=(ROOT/'parallel_locks'/f'mcljcv_{video}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[t for t in load(ROOT/'task_manifest.json')['tasks'] if t['dataset']=='mcljcv' and t['video']==video]
    assert len(tasks)==30
    v=next(v for v in load(ROOT/'dataset_manifest.json')['datasets']['mcljcv'] if v['video_index']==video)
    torch.cuda.reset_peak_memory_stats();targets,th=teacher_targets(v)
    for method in ('Original','B1','B_covstats'):
        mt=[t for t in tasks if t['method']==method]
        if all(valid(t) for t in mt):continue
        eng,rt=runtime(method,targets)
        for t in mt:
            if valid(t):continue
            result=execute(t,v,targets,th,eng,rt)
            result.update(execution_physical_gpu=gpu,execution_pid=os.getpid(),
                          execution_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                          task_gpu_is_original_group_assignment=True)
            from worker import key
            dump('parts/'+key(t)+'/point.json',result)
            dump(f'worker_status/parallel_video{video}.json',dict(status='RUNNING',physical_gpu=gpu,pid=os.getpid(),
                 completed=sum(valid(t) for t in tasks),total=30,peak_reserved_MiB=result['execution_peak_reserved_MiB']))
        del eng,rt;gc.collect();torch.cuda.empty_cache()
    assert all(valid(t) for t in tasks)
    dump(f'worker_status/parallel_video{video}.json',dict(status='PASS',physical_gpu=gpu,pid=os.getpid(),completed=30,total=30,
         peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))

def guardian(owner,ticks):
    while True:
        state=load(ROOT/'parallel_extension_status.json')
        if state.get('original_resumed'):return
        if not same(owner,ticks):
            while any(same(a['pid'],a['start_ticks']) for a in state.get('active',[])):
                time.sleep(10)
            if same(state['original_pid'],state['original_start_ticks']):
                os.kill(state['original_pid'],signal.SIGCONT)
            dump('parallel_guardian_recovery.json',dict(owner_pid=owner,original_pid=state['original_pid'],
                 restored_after_extension_exit=True,time=time.time()))
            return
        time.sleep(10)

def main():
    from worker import valid
    lock=(ROOT/'parallel_extension.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert all(sha(ROOT/n)==h for n,h in load(ROOT/'implementation_hashes.json').items())
    assert load(ROOT/'sanity/proxy_sanity.json')['status']=='PASS'
    old=load(ROOT/'worker_status/gpu7.json');pid=old['pid'];p=identity(pid)
    assert p and str(ROOT/'worker.py') in p['command'] and '--gpu 7' in p['command']
    assert p['state']!='T', 'Original worker already suspended; inspect existing extension.'
    # Suspension precedes the snapshot: no current-video race or output collision.
    state=dict(status='INITIALIZING',pid=os.getpid(),original_pid=pid,original_start_ticks=p['start_ticks'],
               original_resumed=False,scope='future MCL videos only; evaluator/protocol/checkpoints unchanged',
               started=time.time(),active=[],completed_videos=[],blocked=[])
    dump('parallel_extension_status.json',state)
    owner=identity(os.getpid())
    with (ROOT/'logs/parallel_guardian.log').open('a') as f:
        guard=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'parallel_extension.py'),'--guardian',str(os.getpid()),
                               '--owner-ticks',owner['start_ticks']],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state['guardian_pid']=guard.pid;dump('parallel_extension_status.json',state)
    active=[]
    try:
        os.kill(pid,signal.SIGSTOP)
        # The paused worker may have moved to the next video before its last receipt.
        # Exclude every video with an existing output directory, even if incomplete.
        tasks=load(ROOT/'task_manifest.json')['tasks']
        vids=sorted({t['video'] for t in tasks if t['dataset']=='mcljcv'},reverse=True)
        remaining=[]
        for video in vids:
            vt=[t for t in tasks if t['dataset']=='mcljcv' and t['video']==video]
            if (ROOT/'parts/mcljcv'/f'{video:02d}').exists():continue
            if video<=old['last_task']['video']+1:continue
            if not all(valid(t) for t in vt):remaining.append(video)
        dump('parallel_extension_plan.json',dict(videos=remaining,excluded_original_current_and_next=True,
             protocol_sha256=sha(ROOT/'protocol.json'),evaluator_hashes=load(ROOT/'implementation_hashes.json'),
             physical_gpus=[4,5,6,7],initial_task_reservation_MiB=12000,margin_MiB=1800,
             nominal_task_gpu_unchanged=True,actual_gpu_recorded_in_receipts=True))
        state.update(status='RUNNING',remaining_videos=remaining,original_suspended=True);dump('parallel_extension_status.json',state)
        pending=remaining[:];retries={};reserve=12000.0;measured_peak=0.0;oom_floor=0.0
        while pending or active:
            for a in active[:]:
                if a['process'].poll() is None:continue
                active.remove(a);video=a['video'];vt=[t for t in tasks if t['dataset']=='mcljcv' and t['video']==video]
                code=a['process'].returncode;okay=code==0 and all(valid(t) for t in vt)
                schedule(dict(event='finished',video=video,gpu=a['gpu'],pid=a['process'].pid,exit_code=code,complete=okay))
                if okay:state['completed_videos'].append(video)
                else:
                    log=Path(a['log']).read_text(errors='replace');retries[video]=retries.get(video,0)+1
                    if 'out of memory' in log.lower():oom_floor=max(reserve*1.3,16000);reserve=oom_floor
                    if retries[video]<3:pending.append(video)
                    else:state['blocked'].append(dict(video=video,log=a['log'],error=log[-4000:]))
            free,used=snapshot();reserved={g:0.0 for g in free}
            for a in active:
                reserved[a['gpu']]+=max(0,a['reserved_MiB']-used.get(a['process'].pid,0))
                status=ROOT/f"worker_status/parallel_video{a['video']}.json"
                if status.exists():
                    peak=load(status).get('peak_reserved_MiB',0)
                    if peak:
                        measured_peak=max(measured_peak,peak)
                        reserve=max(9500,measured_peak*1.3+1500,oom_floor)
            mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
            while pending and mem>16000 and len(active)<max(4,(os.cpu_count() or 16)//4):
                gpu=max(free,key=lambda g:free[g]-reserved[g])
                if free[gpu]-reserved[gpu]<reserve+1800:break
                video=pending.pop(0);log=ROOT/'logs'/f'parallel_mcl{video}_gpu{gpu}_try{retries.get(video,0)}.log'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with log.open('a') as f:
                    child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'parallel_extension.py'),'--worker','--gpu',str(gpu),'--video',str(video)],
                        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                a=dict(video=video,gpu=gpu,process=child,start_ticks=identity(child.pid)['start_ticks'],reserved_MiB=reserve,log=str(log));active.append(a);reserved[gpu]+=reserve;mem-=4000
                schedule(dict(event='started',video=video,gpu=gpu,pid=child.pid,free_MiB=free[gpu],reserved_MiB=reserve,retry=retries.get(video,0)))
                state['active']=[dict(video=a['video'],gpu=a['gpu'],pid=a['process'].pid,start_ticks=a['start_ticks'],reserved_MiB=a['reserved_MiB'],log=a['log']) for a in active]
                dump('parallel_extension_status.json',state)
            state.update(active=[dict(video=a['video'],gpu=a['gpu'],pid=a['process'].pid,start_ticks=a['start_ticks'],reserved_MiB=a['reserved_MiB'],log=a['log']) for a in active],
                         pending_videos=pending,retries=retries,updated=time.time())
            dump('parallel_extension_status.json',state)
            time.sleep(10)
        state['status']='PASS' if not state['blocked'] else 'PARTIAL_RESTORE_ORIGINAL'
    except Exception:
        state.update(status='FAILED_RESTORE_ORIGINAL',error=traceback.format_exc())
        # Preserve active auxiliary workers; wait before restoring the sequential
        # writer, so canonical parts can never have simultaneous writers.
        for a in active:a['process'].wait()
        raise
    finally:
        if same(pid,state['original_start_ticks']):os.kill(pid,signal.SIGCONT)
        state.update(original_resumed=True,finished=time.time());dump('parallel_extension_status.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--gpu',type=int,choices=(4,5,6,7));parser.add_argument('--video',type=int)
    parser.add_argument('--guardian',type=int);parser.add_argument('--owner-ticks');args=parser.parse_args()
    (ROOT/'parallel_locks').mkdir(exist_ok=True)
    if args.guardian is not None:guardian(args.guardian,args.owner_ticks)
    elif args.worker:worker_main(args.gpu,args.video)
    else:main()
