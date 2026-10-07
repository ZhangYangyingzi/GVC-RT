"""Resumable GPU 4–7 scheduler with live memory reservations and CPU FID."""
import math,traceback
from v68_io import *

def gpu_state():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(a):int(b) for a,b in (line.split(',') for line in text.splitlines()) if int(a) in (4,5,6,7)}
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    used={}
    for line in text.splitlines():
        try:
            pid,memory=line.split(',');used[int(pid)]=int(memory)
        except ValueError:pass
    return free,used

def identity(pid):
    try:
        folder=Path('/proc')/str(pid);stat=(folder/'stat').read_text().rsplit(') ',1)[1].split()
        if stat[0]=='Z':return None
        return dict(start=stat[19],command=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None

def commands(argv,log):
    with (ROOT/'logs/commands.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(argv=argv,cwd=str(REPO),log=str(log),unix=time.time()))+'\n')

def tasks():
    result={}
    for branch in BRANCHES:
        if branch=='F_frozen_core' and load(ROOT/'audits/objective_smoke.json')['reuse_F']:continue
        measured=load(ROOT/'audits/objective_smoke.json')['peak_reserved']/2**20
        result['train_'+branch]=dict(kind='train',script='train.py',args=['--branch',branch],branch=branch,memory=max(12000,math.ceil(measured)+3500),gpu=True)
    for method in METHODS:
        result['smoke_'+method]=dict(kind='smoke',script='evaluate.py',args=['--method',method,'--dataset','uvg','--video','0','--smoke'],method=method,memory=12000,gpu=True)
        result['heldout_'+method]=dict(kind='heldout',script='heldout.py',args=['--method',method],method=method,memory=7000,gpu=True)
        for dataset in DATASETS:
            for video in sources(dataset):
                result[f'eval_{dataset}_{method}_{video["video_index"]}']=dict(kind='eval',script='evaluate.py',args=['--method',method,'--dataset',dataset,'--video',str(video['video_index'])],method=method,dataset=dataset,video=video['video_index'],memory=12000,gpu=True)
            result[f'fid_{dataset}_{method}']=dict(kind='fid',script='fid_worker.py',args=['--method',method,'--dataset',dataset],method=method,dataset=dataset,memory=0,gpu=False)
    return result

def complete(task):
    kind=task['kind']
    if kind=='train':
        path=ROOT/'branches'/task['branch']/'training_status.json'
        return path.exists() and load(path)['status']=='PASS'
    if kind=='smoke':return all(point('uvg',task['method'],0,q).exists() for q in (0,9))
    if kind=='eval':return all(point(task['dataset'],task['method'],task['video'],qp).exists() for qp in range(10))
    if kind=='fid':return (ROOT/'parts'/f'fid_done_{task["dataset"]}_{task["method"]}.json').exists()
    return (ROOT/'parts'/f'heldout_done_{task["method"]}.json').exists()

def publish():
    path=ROOT/'evaluation/config.json';cfg=load(path);changed=False
    for branch in BRANCHES:
        metadata=ROOT/'branches'/branch/'checkpoint_hashes.json'
        if not metadata.exists():continue
        for step,cp in load(metadata).items():
            name=tag(branch)+step
            if int(step) in (1500,2000) and name not in cfg['checkpoints']:
                cp=cp['inference']
                assert sha(cp['path'])==cp['sha256'];cfg['checkpoints'][name]=cp;cfg['core_hashes'][name]=cp['compression_hash'];changed=True
    if changed:dump(path,cfg)
    initial=[ROOT/'branches'/branch/'initialization_audit.json' for branch in BRANCHES]
    if all(path.exists() for path in initial) and not load(ROOT/'audits/objective_smoke.json')['reuse_F']:
        assert all(load(p)==load(initial[0]) for p in initial)
        target=ROOT/'audits/branch_initialization_equivalence.json'
        if not target.exists():dump(target,dict(status='PASS',exact_initialization_optimizer_rng_equality=True))
    return cfg['checkpoints']

def ready(task,cps):
    if task['kind']=='train':return True
    if not (ROOT/'audits/evaluation_protocol_resolution.json').exists():return False
    if task['method']!='original' and task['method'] not in cps:return False
    if task['kind']=='eval':return complete(dict(kind='smoke',method=task['method']))
    if task['kind']=='fid':
        return all(point(task['dataset'],task['method'],video['video_index'],qp).exists()
                   for video in sources(task['dataset']) for qp in range(10))
    return True

def run():
    frozen();assert load(ROOT/'audits/objective_smoke.json')['status']=='PASS'
    jobs=tasks();active={};attempts={};failed_cards={}
    schedule=ROOT/'audits/gpu_schedule.json'
    events=load(schedule).get('events',[]) if schedule.exists() else []
    prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
    for key,worker in prior.get('active',{}).items():
        who=identity(worker['PID'])
        if key in jobs and who and who['start']==worker['start']:
            active[key]=dict(PID=worker['PID'],gpu=worker['gpu'],start=who['start'],proc=None,log=None)
    def status(phase='parallel_training_evaluation',**extra):
        counts={m:sum(len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for d in DATASETS) for m in METHODS}
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'FAIL' if phase=='failed' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),method_points=counts,raw_RD_points=sum(counts.values()),raw_RD_target=1860,FID_points=len(list((ROOT/'parts/fid').glob('*/*/qp*.json'))),FID_target=240,heldout_points=len(list((ROOT/'parts/heldout').glob('*/*.json'))),heldout_target=576,pending=sum(not complete(job) and key not in active for key,job in jobs.items()),active={key:{k:v for k,v in worker.items() if k not in ('proc','log')}|dict(kind=jobs[key]['kind'],log=str(ROOT/'logs'/f'{key}.log')) for key,worker in active.items()},attempts=attempts,**extra))
        dump(schedule,dict(status='PASS' if phase=='complete' else 'RUNNING',allowed_gpus=[4,5,6,7],multiple_processes_per_gpu=True,live_memory_reservation=True,events=events))
    def start(key,gpu):
        job=jobs[key];log_path=ROOT/'logs'/f'{key}.log';log=log_path.open('a')
        argv=[PYTHON,'-B','-u',str(ROOT/job['script']),*job['args']]
        if gpu is not None:argv+=['--gpu',str(gpu)]
        commands(argv,log_path)
        proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        who=identity(proc.pid)
        active[key]=dict(PID=proc.pid,gpu=gpu,start=who['start'] if who else None,proc=proc,log=log)
        attempts[key]=attempts.get(key,0)+1
        events.append(dict(event='start',task=key,gpu=gpu,PID=proc.pid,reserved_mib=job['memory'],unix=time.time()))
        status();print('START',key,'GPU',gpu,'PID',proc.pid,flush=True)
    try:
        while True:
            for key,worker in list(active.items()):
                if worker['proc'] is not None:
                    rc=worker['proc'].poll()
                    if rc is None:continue
                else:
                    if identity(worker['PID']):continue
                    rc=0 if complete(jobs[key]) else -1
                if worker['log']:worker['log'].close()
                del active[key];events.append(dict(event='finish',task=key,gpu=worker['gpu'],PID=worker['PID'],returncode=rc,unix=time.time()))
                if rc!=0 or not complete(jobs[key]):
                    failed_cards.setdefault(key,set()).add(worker['gpu'])
                    dump(ROOT/'logs/failures'/f'{key}_{attempts.get(key,1)}.json',dict(task=key,returncode=rc,gpu=worker['gpu'],log=str(ROOT/'logs'/f'{key}.log')))
                    assert attempts.get(key,1)<4,f'{key} failed four attempts'
                print('FINISH',key,'returncode',rc,flush=True)
            cps=publish();pending=[key for key,job in jobs.items() if key not in active and not complete(job)]
            if not pending and not active:break
            cpu=sum(worker['gpu'] is None for worker in active.values())
            for key in pending:
                job=jobs[key]
                if not job['gpu'] and ready(job,cps) and cpu<4:start(key,None);cpu+=1
            free,used=gpu_state();headroom=dict(free)
            for key,worker in active.items():
                if worker['gpu'] is not None:
                    headroom[worker['gpu']]-=max(0,jobs[key]['memory']-used.get(worker['PID'],0))
            priority={'train':0,'smoke':1,'eval':2,'heldout':3}
            for key in sorted((key for key in pending if jobs[key]['gpu']),key=lambda key:priority[jobs[key]['kind']]):
                job=jobs[key]
                if not ready(job,cps):continue
                eligible=[gpu for gpu in free if headroom[gpu]>=job['memory']
                          and sum(worker['gpu']==gpu for worker in active.values())<3
                          and gpu not in failed_cards.get(key,set())]
                if not eligible:continue
                preferred={'F_frozen_core':4,'J_joint_core':6}.get(job.get('branch'))
                gpu=preferred if preferred in eligible else max(eligible,key=lambda gpu:headroom[gpu]);start(key,gpu);headroom[gpu]-=job['memory']
            phase='parallel_training_evaluation' if (ROOT/'audits/evaluation_protocol_resolution.json').exists() else 'training_evaluation_frame_choice_pending'
            status(phase);time.sleep(10)
        status('raw_export')
        argv=[PYTHON,'-B','-u',str(ROOT/'report.py')];log_path=ROOT/'logs/raw_export.log'
        commands(argv,log_path)
        with log_path.open('a') as log:
            proc=subprocess.Popen(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:status('raw_export');time.sleep(5)
            assert proc.returncode==0,'Raw export failed; inspect logs/raw_export.log'
        assert load(ROOT/'final_integrity.json')['status']=='PASS'
        status('complete');print('EXPERIMENT PASS',flush=True)
    except Exception:
        error=traceback.format_exc();status('failed',traceback=error)
        dump(ROOT/'logs/failures'/f'pipeline_{time.time_ns()}.json',dict(traceback=error));raise
