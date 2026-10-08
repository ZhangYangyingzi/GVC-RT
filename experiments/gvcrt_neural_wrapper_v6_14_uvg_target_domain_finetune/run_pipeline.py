"""Memory-aware GPU4-7 training/history/final evaluation controller."""
import argparse,fcntl,traceback
from v614_io import *
def identity(pid):
    try:
        folder=Path('/proc')/str(pid);s=(folder/'stat').read_text().rsplit(') ',1)[1].split()
        return None if s[0]=='Z' else dict(start=s[19],cmd=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None
def gpu_free():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(i):int(m) for i,m in (x.split(',') for x in out.splitlines()) if int(i) in (4,5,6,7)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');(ROOT/'logs').mkdir(exist_ok=True)
    if a.launch:
        with (ROOT/'pipeline.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:print(json.dumps(load(ROOT/'pipeline_status.json')));return
        argv=[PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')];command(argv)
        with (ROOT/'logs/pipeline.log').open('a') as f:proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pipeline.pid').write_text(str(proc.pid)+'\n');print(json.dumps(dict(status='LAUNCHED',PID=proc.pid,log=str(ROOT/'logs/pipeline.log'))));return
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen();assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':print('ALREADY PASS');return
    from eval_adapter import publish,validate
    from select_checkpoint import select,final_methods
    if not (ROOT/'audits/baseline_reuse.json').exists():
        from reuse import main as reuse
        reuse()
    vs=sources();val=next(v for v in vs if split(v)=='validation');active={};old=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};attempts=old.get('attempts',{});events=old.get('events',[])
    def trained():return (ROOT/'training_status.json').exists() and load(ROOT/'training_status.json')['status']=='PASS' and (ROOT/'frozen_checkpoint_index.json').exists()
    def done(job):
        if job['kind']=='init':return (ROOT/'checkpoint_index.json').exists() and '0' in load(ROOT/'checkpoint_index.json')
        if job['kind']=='train':return trained()
        return all(point('uvg',job['method'],job['video'],q).exists() for q in job['qps'])
    def evaljob(m,v,qs):return dict(kind='eval',script='evaluate.py',method=m,video=v['video_index'],qps=list(qs),args=['--dataset','uvg','--method',m,'--video',str(v['video_index']),'--qps',','.join(map(str,qs))],memory=14000)
    def jobs_now():
        jobs={'init':dict(kind='init',script='train.py',args=['--initialize-only'],memory=14000),'train':dict(kind='train',script='train.py',args=[],memory=20000)}
        for m in ('original','v62_initial'):
            for v in vs:jobs[f'baseline_{m}_{v["video_index"]}']=evaljob(m,v,range(10))
        jobs['step0_codec']=evaljob(name(0),val,(0,9))
        for s in STEPS:
            jobs[f'validation_{s}']=evaljob(name(s),val,range(10))
            if s in (0,1000,3000,5000):
                for v in vs:
                    if split(v)=='train_fit':jobs[f'fit_{s}_{v["video_index"]}']=evaljob(name(s),v,(0,4,9))
        if (ROOT/'validation_selection.json').exists() and trained():
            for m in final_methods():
                for v in vs:jobs[f'final_{m}_{v["video_index"]}']=evaljob(m,v,range(10))
        return jobs
    jobs=jobs_now()
    for key,w in old.get('active',{}).items():
        who=identity(w['PID'])
        if key in jobs and who and who['start']==w['start']:active[key]=dict(**w,proc=None,stream=None)
    def status(phase):
        st=load(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {}
        dump(ROOT/'pipeline_status.json',dict(status=load(ROOT/'final_integrity.json')['status'] if phase=='complete' else 'RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),adaptation_step=st.get('adaptation_step',0),training_target=5000,unique_RD_points=len(list((ROOT/'parts').glob('*/video_*_qp*.json'))),active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events))
    def ready(key,j,cps):
        if j['kind']=='init':return not any(jobs[k]['kind'] in ('init','train') for k in active)
        if j['kind']=='train':return (ROOT/'audits/step0_codec_equivalence.json').exists() and not any(jobs[k]['kind']=='init' for k in active)
        if j['method']!='original' and j['method'] not in cps:return False
        # Keep source-method/video exclusive even when QP subsets differ.
        if any(jobs[k]['kind']=='eval' and jobs[k]['method']==j['method'] and jobs[k]['video']==j['video'] for k in active):return False
        if key.startswith('baseline_') or key=='step0_codec':return True
        return (ROOT/'audits/step0_codec_equivalence.json').exists()
    while True:
        jobs=jobs_now()
        for key,w in list(active.items()):
            if w['proc'] is not None:
                rc=w['proc'].poll()
                if rc is None:continue
            else:
                if identity(w['PID']):continue
                rc=0 if done(jobs[key]) else -1
            if w['stream']:w['stream'].close()
            del active[key];events.append(dict(event='finish',task=key,returncode=rc,gpu=w['gpu'],unix=time.time()))
            if rc or not done(jobs[key]):
                dump(ROOT/'logs'/f'failure_{key}_{time.time_ns()}.json',dict(returncode=rc,log=w['log']))
                assert attempts.get(key,0)<3,f'{key} failed repeatedly; inspect {w["log"]}'
            print('FINISH',key,rc,flush=True)
        cfg=publish();cps=cfg['checkpoints']
        gate=ROOT/'audits/step0_codec_equivalence.json'
        if not gate.exists() and done(jobs['step0_codec']) and all(point('uvg','v62_initial',val['video_index'],q).exists() for q in (0,9)):
            evidence=[]
            for q in (0,9):
                a=load(point('uvg',name(0),val['video_index'],q));b=load(point('uvg','v62_initial',val['video_index'],q));validate(a,val,q,name(0));validate(b,val,q,'v62_initial')
                keys=('real_bytes','bitstream_sha256','reconstruction_sha256','source_rgb_sha256','actual_qps','loaded_receiver_module_hashes')
                assert all(a[k]==b[k] for k in keys),('step0 mismatch',q)
                evidence.append(dict(QP=q,**{k:a[k] for k in keys}))
            assert load(ROOT/'checkpoint_index.json')['0']['module_hashes']==cfg['checkpoints']['v62_initial']['module_hashes']
            dump(gate,dict(status='PASS',source_step=1000,adaptation_step=0,sequence='Bosphorus',real_RANS_independent_decode=True,exact_equality=evidence));print('STEP0 CODEC EQUALITY PASS',flush=True)
        if trained() and all(done(jobs[f'validation_{s}']) for s in STEPS) and not (ROOT/'validation_selection.json').exists():
            select();jobs=jobs_now();print('VALIDATION SELECTION FROZEN',flush=True)
        if trained() and (ROOT/'validation_selection.json').exists() and not active and all(done(j) for j in jobs.values()):break
        free=gpu_free();busy={w['gpu'] for w in active.values()}
        # Use separate GPUs for training and evaluation; unrelated processes may coexist.
        for key,j in jobs.items():
            if key in active or done(j) or not ready(key,j,cps):continue
            available=[g for g,n in free.items() if g not in busy and n>=j['memory']]
            if not available:continue
            gpu=max(available,key=lambda g:free[g]);argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)]
            log=ROOT/'logs'/f'{key}.log';stream=log.open('a');command(argv);proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);who=identity(proc.pid)
            active[key]=dict(PID=proc.pid,start=who['start'] if who else None,gpu=gpu,log=str(log),proc=proc,stream=stream);busy.add(gpu);attempts[key]=attempts.get(key,0)+1
            events.append(dict(event='start',task=key,PID=proc.pid,gpu=gpu,unix=time.time()));print('START',key,'GPU',gpu,'PID',proc.pid,flush=True)
        status('final_evaluation' if trained() else 'training_and_history');time.sleep(10)
    status('aggregate');argv=[PYTHON,'-B','-u',str(ROOT/'report.py')];command(argv)
    with (ROOT/'logs/report.log').open('a') as stream:
        p=subprocess.Popen(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        while p.poll() is None:status('aggregate');time.sleep(10)
    assert p.returncode==0,'report.py failed';assert load(ROOT/'final_integrity.json')['status']=='PASS';status('complete');print('EXPERIMENT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};prior.update(status='FAIL',error=error,updated_unix=time.time());dump(ROOT/'pipeline_status.json',prior);dump(ROOT/'logs'/f'failure_pipeline_{time.time_ns()}.json',dict(status='FAIL',traceback=error));raise
