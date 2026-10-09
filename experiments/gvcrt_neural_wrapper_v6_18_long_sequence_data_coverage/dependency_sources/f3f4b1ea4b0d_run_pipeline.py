"""Persistent weight-control pipeline; memory-only GPU4–7 scheduling."""
import argparse,fcntl,traceback
from io17 import *
def identity(pid):
    try:
        parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        return None if parts[0]=='Z' else parts[19]
    except (FileNotFoundError,ProcessLookupError):return None
def idle_gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    minimum=load(ROOT/'gpu_schedule.json')['minimum_free_memory_MiB']
    return {int(i):int(free) for i,free in (r.split(',') for r in rows) if int(i) in (4,5,6,7) and int(free)>=minimum}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'final_integrity.json').exists() and (ROOT/'publication_status.json').exists():
        if load(ROOT/'final_integrity.json')['status']=='PASS' and load(ROOT/'publication_status.json')['status']=='PASS':
            print('ALREADY COMPLETE AND PUBLISHED',flush=True);return
    frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    assert load(ROOT/'audits/baseline_reuse.json')['status']=='PASS'
    from publish_checkpoints import publish
    from adapter import validate
    active={};jobs={};attempts={};events=[];verified={}
    def add(key,kind,script,args):jobs[key]=dict(kind=kind,script=script,args=args)
    add('weight_check','probe','weight_check.py',[]);add('initialize','init','train.py',['--initialize-only']);add('train','train','train.py',[])
    def ej(key,d,m,v,qs):
        add(key,'eval','evaluate.py',['--dataset',d,'--method',m,'--video',str(v['video_index']),'--qps',','.join(map(str,qs))]);jobs[key].update(dataset=d,method=m,video=v['video_index'],qps=list(qs))
    val=sources('uvg_validation')[0];ej('step0_codec','uvg_validation','dists_w05_0',val,(0,9))
    for d in DATASETS:
        for m in (METHODS[-1],):
            for v in sources(d):ej(f'final_{d}_{m}_{v["video_index"]}',d,m,v,range(10))
    def trained():return (ROOT/'training_integrity.json').exists() and load(ROOT/'training_integrity.json')['status']=='PASS'
    def done(j):
        if j['kind']=='probe':return (ROOT/'audits/weight_check.json').exists() and load(ROOT/'audits/weight_check.json')['status']=='PASS'
        if j['kind']=='init':return (ROOT/'checkpoint_index.json').exists() and '0' in load(ROOT/'checkpoint_index.json')
        if j['kind']=='train':return trained()
        paths=[point(j['dataset'],j['method'],j['video'],q) for q in j['qps']]
        if not all(p.exists() for p in paths):return False
        v=next(v for v in sources(j['dataset']) if v['video_index']==j['video'])
        for q,p in zip(j['qps'],paths):
            stamp=(p.stat().st_mtime_ns,p.stat().st_size)
            if verified.get(str(p))!=stamp:validate(load(p),v,q,j['method']);verified[str(p)]=stamp
        return True
    gatefile=ROOT/'audits/step0_codec_equivalence.json'
    previous=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
    attempts=previous.get('attempts',{});events=previous.get('events',[])
    for key,w in previous.get('active',{}).items():
        if key in jobs and w.get('start_identity') and identity(w['PID'])==w['start_identity']:active[key]=dict(**w,proc=None,stream=None)
    def status(phase):
        st=load(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {}
        ga=load(ROOT/'audits/weight_check.json') if (ROOT/'audits/weight_check.json').exists() else {}
        dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),gradient_status=ga.get('status','PENDING'),lambda_dists=0.5,adaptation_step=st.get('adaptation_step',0),training_target=1000,completed_RD=len([p for p in (ROOT/'parts').glob('*/*/video_*_qp*.json') if 'dists_w05_0' not in p.parts]),expected_RD=800,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events))
    while True:
        # A crash after aggregation can resume publication without re-running
        # training, encoding, or overwriting already checked result tables.
        if not active and (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':break
        for key,w in list(active.items()):
            if w['proc'] is None:
                if identity(w['PID'])==w.get('start_identity'):continue
                rc=0 if done(jobs[key]) else -1
            else:rc=w['proc'].poll()
            if rc is None:continue
            if w['stream']:w['stream'].close()
            del active[key];events.append(dict(event='finish',task=key,returncode=rc,gpu=w['gpu'],unix=time.time()))
            print('FINISH',key,rc,flush=True)
            if rc or not done(jobs[key]):
                with open(w['log']) as failure_log:
                    failure_log.seek(w.get('log_offset',0));content=failure_log.read()
                if 'out of memory' in content.lower() and attempts.get(key,0)<6:
                    events.append(dict(event='OOM_REQUEUE',task=key,gpu=w['gpu'],unix=time.time()))
                    # Exited experiment process releases its own CUDA allocations.
                    # Other processes are never signalled. Retry from latest state.
                    print('OOM REQUEUE',key,flush=True);continue
                raise RuntimeError(f'{key} failed, returncode={rc}; log={w["log"]}')
        ga=load(ROOT/'audits/weight_check.json') if (ROOT/'audits/weight_check.json').exists() else None
        cfg=publish()
        if not gatefile.exists() and done(jobs['step0_codec']):
            evidence=[]
            for q in (0,9):
                aa=load(point('uvg_validation','dists_w05_0',val['video_index'],q));bb=load(point('uvg_validation','v62_initial',val['video_index'],q))
                validate(aa,val,q,'dists_w05_0');validate(bb,val,q,'v62_initial')
                keys=('real_bytes','bitstream_sha256','reconstruction_sha256','source_rgb_sha256','actual_qps','loaded_receiver_module_hashes')
                assert all(aa[k]==bb[k] for k in keys),('step0 codec mismatch',q)
                evidence.append(dict(QP=q,**{k:aa[k] for k in keys}))
            dump(gatefile,dict(status='PASS',sequence='Bosphorus',source_step=1000,adaptation_step=0,exact_equality=evidence))
        if trained() and not active and all(done(j) for j in jobs.values()):
            status('aggregate');argv=[PYTHON,'-B','-u',str(ROOT/'report.py')];command(argv)
            with (ROOT/'logs/report.log').open('a') as f:subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert load(ROOT/'final_integrity.json')['status']=='PASS';break
        idle=idle_gpus();assigned={w['gpu'] for w in active.values()};choices=[g for g in idle if g not in assigned]
        for key,j in jobs.items():
            if key in active or done(j):continue
            if j['kind']!='probe' and (not ga or ga['status']!='PASS'):continue
            if j['kind']=='train' and not gatefile.exists():continue
            if j['kind']=='eval':
                if j['method']!='original' and j['method'] not in cfg['checkpoints']:continue
                if key!='step0_codec' and not gatefile.exists():continue
                if j['method']=='dists_w05_1000' and not trained():continue
            if not choices:break
            gpu=max(choices,key=lambda g:idle[g]);choices.remove(gpu)
            argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)];command(argv)
            log=ROOT/'logs'/f'{key}.log';offset=log.stat().st_size if log.exists() else 0;stream=log.open('a');proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active[key]=dict(PID=proc.pid,start_identity=identity(proc.pid),gpu=gpu,log=str(log),log_offset=offset,proc=proc,stream=stream);attempts[key]=attempts.get(key,0)+1
            events.append(dict(event='start',task=key,PID=proc.pid,gpu=gpu,unix=time.time()));print('START',key,'GPU',gpu,flush=True)
        phase='evaluation' if trained() else 'training' if 'train' in active else 'weight_verification' if 'weight_check' in active else 'WAITING_GPU_MEMORY' if not active else 'step0_verification'
        status(phase);time.sleep(10)
    status('publication');argv=[PYTHON,'-B','-u',str(ROOT/'publish_github.py')];command(argv)
    with (ROOT/'logs/publication.log').open('a') as f:subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    state=load(ROOT/'pipeline_status.json');state.update(status=load(ROOT/'final_integrity.json')['status'],phase='complete',publication=load(ROOT/'publication_status.json'));dump(ROOT/'pipeline_status.json',state)
    print('PIPELINE COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        old=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};old.update(status='FAIL',error=traceback.format_exc());dump(ROOT/'pipeline_status.json',old)
        dump(ROOT/'blocking_error.json',dict(status='FAIL',error=old['error'],adaptation_step=old.get('adaptation_step',0),completed_RD=old.get('completed_RD',0),expected_RD=800,updated_unix=time.time()));raise
