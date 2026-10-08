"""Resumable single-GPU training and parallel GPU4–7 evaluation controller."""
import argparse,fcntl,traceback
from mixed_io import *
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return None if s[0]=='Z' else dict(start=s[19])
    except (FileNotFoundError,ProcessLookupError):return None
def gpu_free():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(i):int(m) for i,m in (x.split(',') for x in out.splitlines()) if int(i) in (4,5,6,7)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':print('ALREADY PASS');return
    from publish_checkpoints import publish
    from adapter import validate
    if not (ROOT/'audits/baseline_reuse.json').exists():
        from reuse import main as reuse
        reuse()
    jobs={}
    def ej(key,d,m,v,qs):jobs[key]=dict(kind='eval',script='evaluate.py',dataset=d,method=m,video=v['video_index'],qps=list(qs),args=['--dataset',d,'--method',m,'--video',str(v['video_index']),'--qps',','.join(map(str,qs))],memory=15000)
    jobs['init']=dict(kind='init',script='train.py',args=['--initialize-only'],memory=16000)
    jobs['train']=dict(kind='train',script='train.py',args=[],memory=22000)
    val=sources('uvg_validation')[0];ej('step0_codec','uvg_validation','mixed_0',val,(0,9))
    ej('step0_reference','uvg_validation','v62_initial',val,(0,9))
    # Fixed validation/model schedule; no data-dependent checkpoint selection.
    for m in ('mixed_1000','mixed_5000'):
        for d in ('uvg_validation','ulong_normal','ulong_hard'):
            for v in sources(d):ej(f'validation_{d}_{m}_{v["video_index"]}',d,m,v,range(10) if d=='uvg_validation' else (0,2,4,6,8,9))
    for m in METHODS:
        for d in DATASETS:
            for v in sources(d):ej(f'final_{d}_{m}_{v["video_index"]}',d,m,v,range(10))
    def trained():return (ROOT/'training_integrity.json').exists() and load(ROOT/'training_integrity.json')['status']=='PASS'
    def done(j):
        if j['kind']=='init':return (ROOT/'checkpoint_index.json').exists() and '0' in load(ROOT/'checkpoint_index.json')
        if j['kind']=='train':return trained()
        return all(point(j['dataset'],j['method'],j['video'],q).exists() for q in j['qps'])
    old=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};attempts=old.get('attempts',{});events=old.get('events',[]);active={}
    for key,w in old.get('active',{}).items():
        who=identity(w['PID'])
        if key in jobs and who and who['start']==w['start']:active[key]=dict(**w,proc=None,stream=None)
    def status(phase):
        st=load(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {}
        dump(ROOT/'pipeline_status.json',dict(status='PASS' if phase=='complete' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),adaptation_step=st.get('adaptation_step',0),training_target=5000,completed_jobs=sum(done(j) for j in jobs.values()),expected_jobs=len(jobs),unique_RD_points=len(list((ROOT/'parts').glob('*/*/video_*_qp*.json'))),active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events))
    gate=ROOT/'audits/step0_codec_equivalence.json'
    while True:
        for key,w in list(active.items()):
            if w['proc'] is not None:
                rc=w['proc'].poll()
                if rc is None:continue
            else:
                if identity(w['PID']):continue
                rc=0 if done(jobs[key]) else -1
            if w['stream']:w['stream'].close()
            del active[key];events.append(dict(event='finish',task=key,returncode=rc,gpu=w['gpu'],unix=time.time()))
            if rc or not done(jobs[key]):assert attempts.get(key,0)<2,f'{key} failed: {w["log"]}'
            print('FINISH',key,rc,flush=True)
        cfg=publish()
        if not gate.exists() and done(jobs['step0_codec']) and done(jobs['step0_reference']):
            evidence=[]
            for q in (0,9):
                aa=load(point('uvg_validation','mixed_0',val['video_index'],q));bb=load(point('uvg_validation','v62_initial',val['video_index'],q))
                validate(aa,val,q,'mixed_0');validate(bb,val,q,'v62_initial')
                keys=('real_bytes','bitstream_sha256','reconstruction_sha256','source_rgb_sha256','actual_qps','loaded_receiver_module_hashes')
                assert all(aa[k]==bb[k] for k in keys),('step0 mismatch',q)
                evidence.append(dict(QP=q,**{k:aa[k] for k in keys}))
            dump(gate,dict(status='PASS',sequence='Bosphorus',source_step=1000,mixed_step=0,exact_equality=evidence))
        if trained() and not active and all(done(j) for j in jobs.values()):break
        free=gpu_free();counts={g:sum(w['gpu']==g for w in active.values()) for g in free};training_gpus={w['gpu'] for k,w in active.items() if jobs[k]['kind'] in ('train','init')}
        for key,j in jobs.items():
            if key in active or done(j):continue
            if j['kind']=='train' and not gate.exists():continue
            if j['kind']=='eval':
                if j['method']!='original' and j['method'] not in cfg['checkpoints']:continue
                if not key.startswith('step0') and not gate.exists():continue
                if key.startswith('final_') and j['method'].startswith('mixed_') and not trained():continue
            choices=[g for g,n in free.items() if n>=j['memory'] and (counts[g]==0 if j['kind'] in ('train','init') else counts[g]<2 and g not in training_gpus)]
            if not choices:continue
            gpu=max(choices,key=lambda g:free[g]);argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)]
            log=ROOT/'logs'/f'{key}.log';stream=log.open('a');command(argv);proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);who=identity(proc.pid)
            active[key]=dict(PID=proc.pid,start=who['start'] if who else None,gpu=gpu,log=str(log),proc=proc,stream=stream);counts[gpu]+=1;free[gpu]-=j['memory'];training_gpus.update([gpu] if j['kind'] in ('train','init') else []);attempts[key]=attempts.get(key,0)+1
            events.append(dict(event='start',task=key,PID=proc.pid,gpu=gpu,unix=time.time()));print('START',key,'GPU',gpu,flush=True)
        status('evaluation' if trained() else 'training_and_validation');time.sleep(10)
    status('aggregate');argv=[PYTHON,'-B','-u',str(ROOT/'report.py')];command(argv)
    with (ROOT/'logs/report.log').open('a') as f:
        proc=subprocess.Popen(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        while proc.poll() is None:status('aggregate');time.sleep(10)
    assert proc.returncode==0,'report.py failed';assert load(ROOT/'final_integrity.json')['status']=='PASS';status('complete');print('EXPERIMENT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};prior.update(status='FAIL',error=error,updated_unix=time.time());dump(ROOT/'pipeline_status.json',prior);raise
