"""Restartable smoke gate, four GPU workers, aggregation; no training."""
import argparse,fcntl,traceback
from retention_io import *
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return None if s[0]=='Z' else dict(start=s[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--status',action='store_true');a=parser.parse_args()
    if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');(ROOT/'logs').mkdir(exist_ok=True)
    if a.launch:
        with (ROOT/'pipeline.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:print('ALREADY RUNNING');return
        argv=[PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')];command(argv)
        with (ROOT/'logs/pipeline.log').open('a') as f:p=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='LAUNCHED',PID=p.pid,log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen();assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':print('ALREADY PASS');return
    from adapter import validate
    if not (ROOT/'audits/baseline_reuse.json').exists():
        from reuse import main as reuse
        reuse()
    dump(ROOT/'audits/executed_source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*.py')})
    jobs={}
    for d in DATASETS:
        for m in METHODS[1:]:jobs[f'smoke_{d}_{m}']=dict(d=d,m=m,i=0,smoke=True,qs=[0,9])
    for d in DATASETS:
        for m in METHODS:
            for v in sources(d):jobs[f'eval_{d}_{m}_{v["video_index"]}']=dict(d=d,m=m,i=v['video_index'],smoke=False,qs=list(range(10)))
    def done(j):
        v=next(v for v in sources(j['d']) if v['video_index']==j['i'])
        for q in j['qs']:
            p=point(j['d'],j['m'],j['i'],q)
            if not p.exists():return False
            validate(load(p),v,q,j['m'])
        return True
    complete={k for k,j in jobs.items() if done(j)}
    old=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};attempts=old.get('attempts',{});events=old.get('events',[]);active={}
    for k,w in old.get('active',{}).items():
        who=identity(w['PID'])
        if who and who['start']==w['start'] and str(ROOT/'evaluate.py') in who['command']:active[k]=dict(**w,proc=None,stream=None)
    def status(phase):
        dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),completed_RD=len(list((ROOT/'parts').glob('*/*/video_*_qp*.json'))),expected_RD=520,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events))
    while True:
        for k,w in list(active.items()):
            rc=w['proc'].poll() if w['proc'] is not None else (None if identity(w['PID']) else 0)
            if rc is None:continue
            if w['stream']:w['stream'].close()
            del active[k];ok=done(jobs[k]);events.append(dict(task=k,returncode=rc,verified=ok,unix=time.time()))
            if ok:complete.add(k)
            else:assert attempts[k]<3,('repeated worker failure',k,w['log'])
            print('FINISH',k,rc,ok,flush=True)
        gate=ROOT/'audits/smoke_audit.json'
        if not gate.exists() and all(k in complete for k in jobs if k.startswith('smoke_')):
            records=[]
            for d in DATASETS:
                for m in METHODS[1:]:
                    for q in (0,9):
                        p=point(d,m,0,q);r=load(p);assert not r.get('reused')
                        if m=='v62_initial':
                            oldr=load(oldpoint(d,m,0,q));keys=('real_bytes','bitstream_sha256','reconstruction_sha256','actual_qps','loaded_receiver_module_hashes')
                            assert all(r[k]==oldr[k] for k in keys),('baseline smoke mismatch',d,q)
                        records.append(dict(dataset=d,method=m,QP=q,point_sha256=sha(p),real_bytes=r['real_bytes'],runtime_audit=r['runtime_audit'],independent_decode_pass=r['independent_decode_pass'],state_sync_pass=r['state_sync_pass'],crop=r['metric_crop']))
            dump(gate,dict(status='PASS',points=12,baseline_exact_byte_and_reconstruction_equality=True,records=records));print('SMOKE PASS',flush=True)
        if len(complete)==len(jobs) and not active:break
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(i):int(n) for i,n in (x.split(',') for x in output.splitlines()) if int(i) in (4,5,6,7)};busy={w['gpu'] for w in active.values()}
        for k,j in jobs.items():
            if k in complete or k in active or (not j['smoke'] and not gate.exists()):continue
            if any((jobs[x]['d'],jobs[x]['m'],jobs[x]['i'])==(j['d'],j['m'],j['i']) for x in active):continue
            # Full jobs may already be complete after their smoke subset finished.
            if done(j):complete.add(k);continue
            cards=[g for g,n in free.items() if n>=14000 and g not in busy]
            if not cards:break
            gpu=max(cards,key=lambda g:free[g]);argv=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--dataset',j['d'],'--method',j['m'],'--video',str(j['i']),'--gpu',str(gpu)]+(['--smoke'] if j['smoke'] else [])
            command(argv);log=ROOT/'logs'/f'{k}.log';f=log.open('a');p=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);who=identity(p.pid)
            active[k]=dict(PID=p.pid,start=who['start'],gpu=gpu,log=str(log),proc=p,stream=f);busy.add(gpu);attempts[k]=attempts.get(k,0)+1;print('START',k,gpu,p.pid,flush=True)
        status('evaluation' if gate.exists() else 'smoke');time.sleep(10)
    status('aggregate');argv=[PYTHON,'-B','-u',str(ROOT/'report.py')];command(argv)
    with (ROOT/'logs/report.log').open('a') as f:
        p=subprocess.Popen(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:status('aggregate');time.sleep(10)
    assert p.returncode==0,'report failed';assert load(ROOT/'final_integrity.json')['status']=='PASS'
    status('complete');s=load(ROOT/'pipeline_status.json');s['status']='PASS';dump(ROOT/'pipeline_status.json',s);print('EXPERIMENT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();s=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};s.update(status='FAIL',error=error);dump(ROOT/'pipeline_status.json',s);dump(ROOT/'logs'/f'failure_pipeline_{time.time_ns()}.json',s);raise
