"""Dependency-based continuation/evaluation/FID scheduler; only allowed GPUs, no signals."""
import math,traceback
from v68_io import *
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return None if s[0]=='Z' else dict(start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None
def gpu_state():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(a):int(b) for a,b in (l.split(',') for l in out.splitlines()) if int(a) in (4,5,6,7)}
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True);used={}
    for line in out.splitlines():
        try:p,m=line.split(',');used[int(p)]=int(m)
        except ValueError:pass
    return free,used
def jobs():
    smoke=load(ROOT/'audits/gpu_memory_smoke.json');tasks={}
    for b in BRANCHES:
        smoke_mem=max((v.get('max_memory_reserved',0) for v in smoke.values() if isinstance(v,dict)),default=0)
        mem=max(22000,math.ceil(smoke_mem/2**20)+3000)
        tasks['train_'+b]=dict(kind='train',script='train.py',args=['--branch',b],branch=b,gpu=True,memory=mem)
    for m in FRESH:
        tasks['drift_'+m]=dict(kind='drift',script='diagnostics.py',args=['--method',m],method=m,gpu=True,memory=16000)
        if m=='B1000':continue
        tasks['smoke_'+m]=dict(kind='smoke',script='evaluate.py',args=['--method',m,'--dataset','uvg','--video','0','--smoke'],method=m,gpu=True,memory=18000)
        tasks['heldout_'+m]=dict(kind='heldout',script='heldout.py',args=['--method',m],method=m,gpu=True,memory=7000)
        for d in DATASETS:
            for v in sources(d):
                tasks[f'eval_{d}_{m}_{v["video_index"]}']=dict(kind='eval',script='evaluate.py',args=['--method',m,'--dataset',d,'--video',str(v['video_index'])],method=m,dataset=d,video=v['video_index'],gpu=True,memory=18000)
            tasks[f'fid_{d}_{m}']=dict(kind='fid',script='fid_worker.py',args=['--method',m,'--dataset',d],method=m,dataset=d,gpu=False,memory=0)
    for m in DIAG_METHODS:
        tasks['semantic_'+m]=dict(kind='semantic',script='semantic_drift.py',args=['--method',m],method=m,gpu=True,memory=18000)
    return tasks
def complete(j):
    if j['kind']=='semantic':return (ROOT/'parts'/f'semantic_done_{j["method"]}.json').exists()
    if j['kind']=='train':
        p=ROOT/'branches'/j['branch']/'training_status.json';return p.exists() and load(p)['status']=='PASS'
    if j['kind']=='smoke':return (ROOT/'parts'/f'eval_done_uvg_{j["method"]}_0_smoke.json').exists()
    if j['kind']=='eval':return all(point(j['dataset'],j['method'],j['video'],q).exists() for q in range(10))
    if j['kind']=='fid':return (ROOT/'parts'/f'fid_done_{j["dataset"]}_{j["method"]}.json').exists()
    if j['kind']=='heldout':return (ROOT/'parts'/f'heldout_done_{j["method"]}.json').exists()
    return (ROOT/'parts'/f'diagnostic_done_{j["method"]}.json').exists()
def publish_checkpoints():
    p=ROOT/'evaluation/config.json';cfg=load(p);old=json.dumps(cfg,sort_keys=True)
    for b in BRANCHES:
        meta=ROOT/'branches'/b/'checkpoint_hashes.json'
        if not meta.exists():continue
        for step,cp in load(meta).items():
            if int(step)>1000:
                name=tag(b)+'_'+step
                if name not in cfg['checkpoints']:assert sha(cp['path'])==cp['sha256'];cfg['checkpoints'][name]=cp
    if json.dumps(cfg,sort_keys=True)!=old:dump(p,cfg)
    init=[ROOT/'branches'/b/'initialization_audit.json' for b in BRANCHES]
    if all(p.exists() for p in init):
        a,b=map(load,init);assert a==b
        target=ROOT/'audits/branch_initialization_equivalence.json'
        if not target.exists():dump(target,dict(status='PASS',all_parameter_tensors_identical=True,optimizer_state_identical=True,RNG_states_identical=True,compression_identical=True,initialization=a))
    return cfg['checkpoints']
def ready(j,cps):
    if j['kind']=='train':return True
    if j['method'] not in cps:return False
    if j['kind']=='eval':return complete(dict(kind='smoke',method=j['method']))
    if j['kind']=='fid':return all(point(j['dataset'],j['method'],v['video_index'],q).exists() for v in sources(j['dataset']) for q in range(10))
    return True
def run():
    tasks=jobs();active={};events=load(ROOT/'audits/gpu_schedule.json').get('events',[]) if (ROOT/'audits/gpu_schedule.json').exists() else [];attempts={};failed_cards={}
    prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
    # Retry budget is per controller launch; previous attempts remain in events/logs.
    for k,j in prior.get('active',{}).items():
        who=identity(j['PID'])
        if k in tasks and who and who['start']==j.get('start') and str(ROOT/tasks[k]['script']) in who['cmd']:active[k]=dict(PID=j['PID'],gpu=j['gpu'],start=who['start'],proc=None,log=None)
    def status(phase='parallel_train_evaluate_fid',**extra):
        counts={m:sum(len(list((ROOT/'parts'/d/m).glob('video_*_qp*.json'))) for d in DATASETS) for m in METHODS}
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'FAIL' if phase=='failed' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),method_points=counts,raw_RD_points=sum(counts.values()),raw_RD_target=2790,FID_points=len(list((ROOT/'parts/fid').glob('*/*/qp*.json'))),FID_target=360,heldout_points=len(list((ROOT/'parts/heldout').glob('*/*.json'))),heldout_target=672,active={k:{kk:vv for kk,vv in v.items() if kk not in ('proc','log')}|dict(kind=tasks[k]['kind'],log=str(ROOT/'logs'/f'{k}.log')) for k,v in active.items()},attempts=attempts,pending=sum(not complete(j) and k not in active for k,j in tasks.items()),max_CPU_FID_workers=4,**extra))
        dump(ROOT/'audits/gpu_schedule.json',dict(status='RUNNING' if phase!='complete' else 'PASS',allowed_gpus=[4,5,6,7],sharing_allowed=True,no_process_signals=True,events=events))
    def start(k,g,free=None):
        j=tasks[k];log=(ROOT/'logs'/f'{k}.log').open('a');cmd=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args']]
        if g is not None:cmd+=['--gpu',str(g)]
        proc=subprocess.Popen(cmd,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);who=identity(proc.pid)
        active[k]=dict(PID=proc.pid,gpu=g,start=who['start'] if who else None,proc=proc,log=log);attempts[k]=attempts.get(k,0)+1
        events.append(dict(event='start',task=k,GPU=g,free_memory_before=free,assigned_role=j['kind'],reserved_headroom_mib=j['memory'],PID=proc.pid,unix=time.time()));status();print('STARTED',k,'GPU',g,'PID',proc.pid,flush=True)
    try:
        frozen();assert load(ROOT/'audits/interface_alignment_calibration.json')['status']=='PASS'
        while True:
            for k,j in list(active.items()):
                if j['proc'] is not None:
                    rc=j['proc'].poll()
                    if rc is None:continue
                else:
                    who=identity(j['PID'])
                    if who and who['start']==j['start']:continue
                    rc=0 if complete(tasks[k]) else -1
                if j['log']:j['log'].close()
                del active[k];events.append(dict(event='finish',task=k,GPU=j['gpu'],PID=j['PID'],returncode=rc,unix=time.time()))
                if rc!=0 or not complete(tasks[k]):
                    failed_cards.setdefault(k,set()).add(j['gpu']);dump(ROOT/'logs/failures'/f'{k}_attempt{attempts.get(k,1)}.json',dict(task=k,returncode=rc,GPU=j['gpu'],PID=j['PID'],log=str(ROOT/'logs'/f'{k}.log'),completed_parts_preserved=True))
                    assert attempts.get(k,1)<4,f'{k} failed four isolated attempts; completed parts preserved'
                print('FINISHED',k,'rc',rc,flush=True)
            cps=publish_checkpoints();pending=[k for k,j in tasks.items() if k not in active and not complete(j)]
            if not pending and not active:break
            cpu_used=sum(j['gpu'] is None for j in active.values());meminfo=Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in meminfo.splitlines() if l.startswith('MemAvailable:')))//1024
            for k in pending:
                j=tasks[k]
                if not j['gpu'] and ready(j,cps) and cpu_used<4 and available>12000:start(k,None);cpu_used+=1;available-=3000
            free,used=gpu_state();headroom=dict(free)
            for k,j in active.items():
                if j['gpu'] is not None:headroom[j['gpu']]-=max(0,tasks[k]['memory']-used.get(j['PID'],0))
            # Initially one evaluation per card. Successful observed evaluation peak
            # is mandatory before safely permitting a second small job.
            measurements=[load(p)['peak_reserved']/2**20 for p in (ROOT/'parts/memory').glob('eval_*.json')]
            priority={'train':0,'smoke':1,'heldout':2,'eval':3,'drift':4,'semantic':4}
            for k in sorted((k for k in pending if tasks[k]['gpu']),key=lambda k:priority[tasks[k]['kind']]):
                j=tasks[k]
                if not ready(j,cps):continue
                if j['kind']=='eval' and measurements:j['memory']=max(18000,math.ceil(max(measurements)*1.15)+2000)
                eligible=[]
                for g in free:
                    occupants=[tasks[key] for key,val in active.items() if val['gpu']==g]
                    cap=2 if measurements and j['kind'] in ('eval','heldout') and all(x['kind'] in ('eval','heldout') for x in occupants) else 1
                    if len(occupants)<cap and headroom[g]>=j['memory'] and g not in failed_cards.get(k,set()):eligible.append(g)
                if not eligible:continue
                g=max(eligible,key=lambda g:headroom[g]);start(k,g,free[g]);headroom[g]-=j['memory']
            status();time.sleep(10)
        status('report')
        with (ROOT/'logs/report.log').open('a') as f:
            p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'report.py')],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            while p.poll() is None:status('report');time.sleep(5)
            assert p.returncode==0,'Report failed: logs/report.log'
        assert load(ROOT/'final_integrity.json')['status']=='PASS';status('complete');print('PIPELINE PASS',flush=True)
    except Exception:
        error=traceback.format_exc();status('failed',traceback=error);dump(ROOT/'logs/failures'/f'scheduler_{time.time_ns()}.json',dict(traceback=error));raise
