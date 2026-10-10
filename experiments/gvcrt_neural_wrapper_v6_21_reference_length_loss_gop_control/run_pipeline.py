"""Recoverable, memory-based GPU 4-7 scheduler, then verified publication."""
import argparse,fcntl,traceback
from io21 import *
def alive(pid,start):
    try:
        a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        return a[0]!='Z' and a[19]==start
    except (FileNotFoundError,ProcessLookupError):return False
def identity(pid):return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
def passed(p):return p.exists() and load(p).get('status')=='PASS'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');args=ap.parse_args()
    if args.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
    os.chdir(REPO);os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    (ROOT/'logs').mkdir(exist_ok=True);lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    def cpu(script):
        cmd=[PYTHON,'-B','-u',str(ROOT/script)];command(cmd)
        with (ROOT/'logs'/f'{Path(script).stem}.log').open('a') as f:subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not passed(ROOT/'preflight_audit.json') or not (ROOT/'qp_semantics_audit.json').exists() or 'qp_semantics_audit.json' not in load(ROOT/'audits/protocol_hashes.json'):
        dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase='preparation',PID=os.getpid()));cpu('prepare.py')
    frozen()
    if not (ROOT/'audits/baseline_reuse.json').exists():cpu('reuse.py')
    from adapter import validate
    from publish_checkpoints import publish
    prior=load(ROOT/'pipeline_status.json');active={};jobs={};checked={};attempts=prior.get('attempts',{});events=prior.get('events',[]);memory=prior.get('memory',{});max_workers=16
    def add(key,kind,script,args,**kw):jobs[key]=dict(kind=kind,script=script,args=args,**kw)
    add('implementation','probe','implementation_check.py',[])
    for b in BRANCHES:
        add('init_'+b,'init','train.py',['--branch',b,'--initialize-only'],branch=b);add('train_'+b,'train','train.py',['--branch',b],branch=b)
    val=sources('uvg_validation')[0]
    def ev(key,ip,d,m,v,qs,smoke=False):
        add(key,'eval','evaluate.py',['--ip',str(ip),'--dataset',d,'--method',m,'--video',str(v['video_index']),'--qps',','.join(map(str,qs))],ip=ip,dataset=d,method=m,video=v['video_index'],qps=list(qs),smoke=smoke)
    for b in BRANCHES:ev('step0_'+b,-1,'uvg_validation',b+'_0',val,(0,9),True)
    for ip in (31,32):ev(f'gop_smoke_{ip}',ip,'uvg_validation','native_initial',val,(0,9),True)
    for ip in IPS:
        for d in DATASETS:
            for m in METHODS:
                for v in sources(d):ev(f'RD_{ip}_{d}_{m}_{v["video_index"]}',ip,d,m,v,range(10))
    def trained(b):return passed(ROOT/'branches'/b/'training_integrity.json')
    def done(j):
        if j['kind']=='probe':return passed(ROOT/'audits/implementation_check.json')
        if j['kind']=='init':
            p=ROOT/'branches'/j['branch']/'checkpoint_index.json';return p.exists() and '0' in load(p)
        if j['kind']=='train':return trained(j['branch'])
        v=next(v for v in sources(j['dataset']) if v['video_index']==j['video'])
        for q in j['qps']:
            p=point(j['dataset'],j['method'],j['video'],q,j['ip'])
            if not p.exists():return False
            stamp=(p.stat().st_size,p.stat().st_mtime_ns)
            if checked.get(str(p))!=stamp:
                try:validate(load(p),v,q,j['method'],j['ip'])
                except (AssertionError,ValueError,KeyError,OSError):return False
                checked[str(p)]=stamp
        return True
    for k,w in prior.get('active',{}).items():
        if k in jobs and alive(w['PID'],w['identity']):active[k]=dict(w,proc=None,stream=None)
    gate=ROOT/'audits/step0_codec_equivalence.json'
    def status(phase):
        steps={b:load(ROOT/'branches'/b/'training_status.json').get('adaptation_step',0) if (ROOT/'branches'/b/'training_status.json').exists() else 0 for b in BRANCHES}
        count=sum(point(d,m,v['video_index'],q,ip).exists() for ip in IPS for d in DATASETS for m in METHODS for v in sources(d) for q in range(10))
        dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),training_steps=steps,completed_RD=count,expected_RD=2400,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events,memory=memory,max_workers=max_workers,updated_unix=time.time()))
    while not passed(ROOT/'final_integrity.json'):
        for k,w in list(active.items()):
            rc=(None if alive(w['PID'],w['identity']) else 0 if done(jobs[k]) else -1) if w['proc'] is None else w['proc'].poll()
            if rc is None:continue
            if w['stream']:w['stream'].close()
            del active[k];events.append(dict(event='finish',task=k,rc=rc,gpu=w['gpu'],unix=time.time()));print('FINISH',k,rc,flush=True)
            if rc or not done(jobs[k]):
                with open(w['log']) as f:f.seek(w['offset']);error=f.read()
                oom='out of memory' in error.lower();retry=(oom and attempts[k]<8) or (jobs[k]['kind']=='eval' and attempts[k]<3)
                events.append(dict(event='failure',task=k,error=error[-5000:],retry=retry,OOM=oom,unix=time.time()))
                if retry:
                    if oom:max_workers=max(1,max_workers-1);memory[jobs[k]['kind']]=max(memory.get(jobs[k]['kind'],0),w['required_MiB'])
                    continue
                raise RuntimeError(f'{k} rc={rc}\n'+error[-6000:])
        cfg=publish()
        if not gate.exists() and all(done(jobs['step0_'+b]) for b in BRANCHES) and all(done(jobs[f'gop_smoke_{ip}']) for ip in (31,32)):
            checks=[]
            for b in BRANCHES:
                assert load(ROOT/'branches'/b/'checkpoint_index.json')['0']['module_hashes']==load(ROOT/'config.json')['source_checkpoint']['module_hashes']
                for q in (0,9):
                    r=load(point('uvg_validation',b+'_0',val['video_index'],q));assert r['same_GPU_step0_verification']['status']=='PASS';checks.append(r['same_GPU_step0_verification'])
            dump(gate,dict(status='PASS',same_GPU_native_source=checks,periodic_I_smoke=True,IPs=IPS,branches=BRANCHES))
        if not active and all(done(j) for j in jobs.values()):status('aggregate');cpu('report.py');break
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        free={int(i):int(v) for i,v in (line.split(',') for line in lines) if int(i) in (4,5,6,7)}
        observed=set()
        for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
            try:pid,used=map(int,line.split(','))
            except ValueError:continue
            observed.add(pid)
            for k,w in active.items():
                if w['PID']==pid:memory[jobs[k]['kind']]=max(memory.get(jobs[k]['kind'],0),used)
        for w in active.values():
            if w['PID'] not in observed:free[w['gpu']]=max(0,free[w['gpu']]-w['required_MiB'])
        choices=list(free)
        for k,j in jobs.items():
            if k in active or done(j):continue
            if j['kind']!='probe' and not passed(ROOT/'audits/implementation_check.json'):continue
            if j['kind']=='train' and not passed(gate):continue
            if j['kind']=='eval':
                if j['method']!='original' and j['method'] not in cfg['checkpoints']:continue
                if not j['smoke'] and not passed(gate):continue
            peak=memory.get(j['kind'],0)
            if j['kind']=='train' and passed(ROOT/'audits/implementation_check.json'):peak=max(peak,load(ROOT/'audits/implementation_check.json')['peak_memory_MiB'])
            required=max(8000,int(peak*1.25+2048));valid=[g for g in choices if free[g]>=required and sum(w['gpu']==g for w in active.values())<4 and not (j['kind']=='eval' and any(k=='train_C' and w['gpu']==g for k,w in active.items()))]
            if not valid or len(active)>=max_workers:continue
            gpu=max(valid,key=lambda g:free[g]);free[gpu]-=required;argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)];command(argv)
            log=ROOT/'logs'/f'{k}.log';offset=log.stat().st_size if log.exists() else 0;f=log.open('a');p=subprocess.Popen(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[k]=dict(PID=p.pid,identity=identity(p.pid),gpu=gpu,log=str(log),offset=offset,required_MiB=required,proc=p,stream=f);attempts[k]=attempts.get(k,0)+1;print('START',k,'GPU',gpu,flush=True)
        phase='training' if any(jobs[k]['kind']=='train' for k in active) else 'evaluation' if all(trained(b) for b in BRANCHES) else 'pretraining_checks'
        status(phase if active else 'WAITING_GPU_MEMORY');time.sleep(10)
    assert passed(ROOT/'final_integrity.json');status('publication');cpu('publish_github.py');pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS'
    s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',publication=pub,active={});dump(ROOT/'pipeline_status.json',s);print('PIPELINE COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        s=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};s.update(status='FAIL',error=traceback.format_exc());dump(ROOT/'pipeline_status.json',s);dump(ROOT/'blocking_error.json',dict(status='FAIL',error=s['error'],unix=time.time()));raise
