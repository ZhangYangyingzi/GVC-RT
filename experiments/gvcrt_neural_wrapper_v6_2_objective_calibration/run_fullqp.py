"""Persistent scoped queue: GPUs 4-7 only, no preemption of other jobs."""
import argparse
import fcntl
import traceback
from fullqp_common import *
from fullqp_evaluate import point_path

def snapshot_old():
    result={}
    for folder in sorted(ROOT.parent.glob('gvcrt_neural_wrapper_v*')):
        if folder==ROOT or not folder.name.startswith(('gvcrt_neural_wrapper_v4','gvcrt_neural_wrapper_v5','gvcrt_neural_wrapper_v6_1')):continue
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.lock':continue
            st=p.stat();result[str(p)]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
    return result

def free_gpus():
    raw=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(line.split(',')[0]) for line in raw.splitlines() if int(line.split(',')[1].strip())<128]

def held(path):
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True

def train_done(branch):
    p=ROOT/'branches'/branch/'training_status.json'
    return p.exists() and load(p).get('status')=='PASS' and load(p).get('update')==3000

def point_group_done(branch,step,split):
    return all(point_path(branch,step,split,i,q).exists() for i in range(32) for q in QPS)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--status',action='store_true');args=parser.parse_args()
    statusfile=ROOT/'fullqp_pipeline_status.json'
    if args.status:
        print(json.dumps(load(statusfile) if statusfile.exists() else {'status':'NOT_STARTED'},indent=2));return
    if args.launch:
        if held(ROOT/'parts'/'fullqp_pipeline.lock'):print('ALREADY RUNNING');return
        log=(ROOT/'logs'/'fullqp_pipeline.log').open('a')
        child=subprocess.Popen([PYTHON,'-B','-u',str(Path(__file__).resolve())],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        print(json.dumps(dict(status='LAUNCHED',pid=child.pid,log=str(ROOT/'logs'/'fullqp_pipeline.log'))));return
    lock=(ROOT/'parts'/'fullqp_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_gate();assert_old_unchanged()
    if not (ROOT/'old_experiment_inventory_before.json').exists():dump(ROOT/'old_experiment_inventory_before.json',snapshot_old())
    dump(ROOT/'fullqp_execution_sources.json',dict(started_unix=time.time(),sources=source_signature()))
    jobs={};failed={};cpu=None;last_training_report=0;finalized=False
    def launch(key,command,gpu=None):
        logfile=ROOT/'logs'/(key+'.log');log=logfile.open('a')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        else:env['CUDA_VISIBLE_DEVICES']=''
        proc=subprocess.Popen([PYTHON,'-B','-u',*command],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);log.close()
        jobs[key]=dict(process=proc,gpu=gpu,log=str(logfile),started_unix=time.time(),command=command)
        print('START',key,'GPU',gpu,'PID',proc.pid,flush=True)
    def evaluation_tasks():
        for branch in ['original']+list(BRANCHES):
            for step in ([0] if branch=='original' else STEPS):
                if branch!='original':
                    hp=ROOT/'branches'/branch/'checkpoint_hashes.json'
                    if not hp.exists() or str(step) not in load(hp):continue
                for split in ('normal','hard'):
                    for shard in range(4):
                        if all(point_path(branch,step,split,i,q).exists() for i in range(shard,32,4) for q in QPS):continue
                        key=f'eval_{branch}_{step}_{split}_{shard}'
                        if key not in jobs and key not in failed:yield key,branch,step,split,shard
    while True:
        for key,job in list(jobs.items()):
            rc=job['process'].poll()
            if rc is None:continue
            print('EXIT',key,rc,flush=True)
            if rc:
                failed[key]=dict(returncode=rc,log=job['log'],command=job['command'],ended_unix=time.time())
                dump(ROOT/'fullqp_failures.json',dict(status='FAIL',jobs=failed))
            if key=='finalize' and rc==0:finalized=True
            del jobs[key]
        if finalized:
            before=load(ROOT/'old_experiment_inventory_before.json');after=snapshot_old()
            changed={p:dict(before=v,after=after.get(p)) for p,v in before.items() if after.get(p)!=v}
            added=sorted(set(after)-set(before))
            dump(ROOT/'old_experiment_inventory_audit.json',dict(status='PASS' if not changed and not added else 'FAIL',changed=changed,added=added,
                method='All preexisting V4/V5/V6.1 artifact sizes and mtime_ns; consumed scientific source/checkpoint files additionally SHA256-verified'))
            if changed or added:raise RuntimeError('Old experiment inventory changed; final integrity must not claim success')
            integrity=load(ROOT/'final_integrity.json');integrity['old_experiments_untouched']=True;dump(ROOT/'final_integrity.json',integrity)
            dump(statusfile,dict(status='PASS',pid=os.getpid(),updated_unix=time.time(),gpus_allowed=[4,5,6,7]));print('FULLQP COMPLETE',flush=True);return
        available=[g for g in free_gpus() if all(j['gpu']!=g for j in jobs.values())]
        for gpu in available:
            pending_train=next((b for b in BRANCHES if not train_done(b) and 'train_'+b not in jobs and 'train_'+b not in failed and not held(ROOT/'branches'/b/'parts'/'fullqp_train.lock')),None)
            if pending_train:
                launch('train_'+pending_train,[str(ROOT/'fullqp_train.py'),'--branch',pending_train,'--gpu',str(gpu)],gpu);continue
            task=next(evaluation_tasks(),None)
            if task:
                key,b,s,split,shard=task
                launch(key,[str(ROOT/'fullqp_evaluate.py'),'--branch',b,'--step',str(s),'--split',split,'--shard',str(shard),'--gpu',str(gpu)],gpu)
        if not any(j['gpu'] is None for j in jobs.values()):
            report_task=None
            for b in ['original']+list(BRANCHES):
                for s in ([0] if b=='original' else STEPS):
                    for split in ('normal','hard'):
                        key=f'report_{b}_{s}_{split}';out=ROOT/'results'/'fullqp'/b/f'step_{s:04d}'/split
                        anchor=ROOT/'results'/'fullqp'/'original'/'step_0000'/split/'report_integrity.json'
                        if key not in failed and not (out/'report_integrity.json').exists() and point_group_done(b,s,split) and (b=='original' or anchor.exists()):
                            report_task=(key,b,s,split);break
                    if report_task:break
                if report_task:break
            if report_task:
                key,b,s,split=report_task;launch(key,[str(ROOT/'fullqp_report.py'),'--branch',b,'--step',str(s),'--split',split])
            elif all(train_done(b) for b in BRANCHES) and all((ROOT/'results'/'fullqp'/b/f'step_{s:04d}'/split/'report_integrity.json').exists() for b in BRANCHES for s in STEPS for split in ('normal','hard')) and not failed:
                launch('finalize',[str(ROOT/'fullqp_report.py'),'--finalize'])
            elif time.time()-last_training_report>300 and 'training_report' not in failed:
                # Avoid reading step0 during its first atomic creation.
                if all((ROOT/'branches'/b/'checkpoint_hashes.json').exists() for b in BRANCHES):
                    launch('training_report',[str(ROOT/'fullqp_report.py'),'--training']);last_training_report=time.time()
        progress={b:(load(ROOT/'branches'/b/'training_status.json') if (ROOT/'branches'/b/'training_status.json').exists() else {'status':'QUEUED','update':0}) for b in BRANCHES}
        points=len(list((ROOT/'parts'/'fullqp').glob('*/*/*/video_*.json'))) if (ROOT/'parts'/'fullqp').exists() else 0
        dump(statusfile,dict(status='RUNNING_WITH_FAILURES' if failed else 'RUNNING',pid=os.getpid(),updated_unix=time.time(),
            gpus_allowed=[4,5,6,7],external_gpu_policy='Only allocate GPU with <128 MiB; never terminate unrelated processes',
            training=progress,completed_validation_points=points,required_validation_points=11904,
            active_jobs={k:dict(pid=j['process'].pid,gpu=j['gpu'],log=j['log'],started_unix=j['started_unix']) for k,j in jobs.items()},failed_jobs=failed))
        if failed and not jobs and not next(evaluation_tasks(),None):
            dump(ROOT/'final_integrity.json',dict(status='FAIL',revision=REVISION,failures=failed));return
        time.sleep(20)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'fullqp_pipeline_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time_unix=time.time()))
        dump(ROOT/'final_integrity.json',dict(status='FAIL',revision=REVISION,reason='Pipeline failure; see fullqp_pipeline_failure.json'))
        raise
