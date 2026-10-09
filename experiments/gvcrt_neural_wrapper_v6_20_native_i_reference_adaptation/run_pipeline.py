"""Recoverable memory-only scheduler. Each physical GPU4–7 hosts <=1 worker."""
import argparse,fcntl,traceback
from io20 import *
def identity(pid):
    try:
        a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split();return None if a[0]=='Z' else a[19]
    except (FileNotFoundError,ProcessLookupError):return None
def passfile(p):return p.exists() and load(p).get('status')=='PASS'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:
        s=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {'status':'PREPARING'}
        if (ROOT/'preparation_status.json').exists():s['preparation']=load(ROOT/'preparation_status.json')
        print(json.dumps(s,indent=2));return
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');(ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    def cpu(script):
        argv=[PYTHON,'-B','-u',str(ROOT/script)];command(argv)
        with (ROOT/'logs'/(Path(script).stem+'.log')).open('a') as f:subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (ROOT/'audits/baseline_reuse.json').exists():cpu('prepare.py')
    frozen()
    from adapter import validate
    from publish_checkpoints import publish
    jobs={};active={};verified={};prior=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};events=prior.get('events',[]);attempts=prior.get('attempts',{});memory=prior.get('memory',{});concurrency=prior.get('max_concurrency',4)
    def add(k,kind,script,args,**data):jobs[k]=dict(kind=kind,script=script,args=args,**data)
    add('implementation','probe','implementation_check.py',[])
    for b in BRANCHES:
        add('init_'+b,'init','train.py',['--branch',b,'--initialize-only'],branch=b);add('train_'+b,'train','train.py',['--branch',b],branch=b)
    vals=[v for v in sources('uvg_holdout') if v['name'] in ('HoneyBee','Jockey')]
    def ej(key,d,m,v,qs,step0=False):add(key,'eval','evaluate.py',['--dataset',d,'--method',m,'--video',str(v['video_index']),'--qps',','.join(map(str,qs))],dataset=d,method=m,video=v['video_index'],qps=list(qs),step0=step0)
    for b in BRANCHES:
        for v in vals:ej(f'step0_{b}_{v["video_index"]}','uvg_holdout',b+'_0',v,(0,4),True)
    for d in DATASETS:
        for m in METHODS:
            for v in sources(d):ej(f'final_{d}_{m}_{v["video_index"]}',d,m,v,range(10))
    def trained(b):return passfile(ROOT/'branches'/b/'training_integrity.json')
    def done(j):
        if j['kind']=='probe':return passfile(ROOT/'audits/implementation_check.json')
        if j['kind']=='init':return (ROOT/'branches'/j['branch']/'checkpoint_index.json').exists() and '0' in load(ROOT/'branches'/j['branch']/'checkpoint_index.json')
        if j['kind']=='train':return trained(j['branch'])
        paths=[point(j['dataset'],j['method'],j['video'],q) for q in j['qps']]
        if not all(p.exists() for p in paths):return False
        v=next(v for v in sources(j['dataset']) if v['video_index']==j['video'])
        for q,p in zip(j['qps'],paths):
            stamp=(p.stat().st_size,p.stat().st_mtime_ns)
            if verified.get(str(p))!=stamp:validate(load(p),v,q,j['method']);verified[str(p)]=stamp
        return True
    for k,w in prior.get('active',{}).items():
        if k in jobs and w.get('start_identity') and identity(w['PID'])==w['start_identity']:active[k]=dict(**w,proc=None,stream=None)
    gate=ROOT/'audits/step0_codec_equivalence.json'
    def progress(phase):
        steps={b:load(ROOT/'branches'/b/'training_status.json').get('adaptation_step',0) if (ROOT/'branches'/b/'training_status.json').exists() else 0 for b in BRANCHES}
        points=sum(point(d,m,v['video_index'],q).exists() for d in DATASETS for m in METHODS for v in sources(d) for q in range(10))
        dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),updated_unix=time.time(),training_steps=steps,completed_points=points,expected_points=800,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,events=events,memory=memory,max_concurrency=concurrency))
        dump(ROOT/'gpu_schedule.json',dict(allowed_GPUs=[4,5,6,7],events=events,memory=memory,max_concurrency=concurrency,other_processes_untouched=True))
    while True:
        if not active and passfile(ROOT/'final_integrity.json'):break
        for k,w in list(active.items()):
            rc=(None if identity(w['PID'])==w['start_identity'] else 0 if done(jobs[k]) else -1) if w['proc'] is None else w['proc'].poll()
            if rc is None:continue
            if w['stream']:w['stream'].close()
            del active[k];events.append(dict(event='finish',task=k,returncode=rc,gpu=w['gpu'],unix=time.time()));print('FINISH',k,rc,flush=True)
            if rc or not done(jobs[k]):
                with open(w['log']) as f:f.seek(w.get('log_offset',0));err=f.read()
                if 'out of memory' in err.lower() and attempts.get(k,0)<8:
                    concurrency=max(1,concurrency-1);events.append(dict(event='OOM_REQUEUE',task=k,gpu=w['gpu'],concurrency=concurrency,unix=time.time()));continue
                raise RuntimeError(f'{k} failed rc={rc}; log={w["log"]}\n'+err[-6000:])
        cfg=publish()
        if not gate.exists() and all(done(j) for j in jobs.values() if j.get('step0')):
            evidence=[]
            for b in BRANCHES:
                assert load(ROOT/'branches'/b/'checkpoint_index.json')['0']['module_hashes']==load(ROOT/'config.json')['source_checkpoint']['module_hashes']
                for v in vals:
                    for q in (0,4):
                        r=load(point('uvg_holdout',b+'_0',v['video_index'],q));assert r['same_GPU_step0_verification']['status']=='PASS' and r['historical_reproduction_verified']
                        if b=='native_i_adapt':assert r['native_I_original_match']['status']=='PASS'
                        evidence.append(dict(branch=b,sequence=v['name'],QP=q,same_GPU=r['same_GPU_step0_verification'],native_I_match=r.get('native_I_original_match'),bitstream_sha256=r['bitstream_sha256'],reconstruction_sha256=r['reconstruction_sha256']))
            dump(gate,dict(status='PASS',source='v619',step0_weights_shared=True,continuous16=True,P_frames=15,reference_resets=[0],points=8,evidence=evidence));print('STEP0 PASS',flush=True)
        if not active and all(done(j) for j in jobs.values()):progress('aggregate');cpu('report.py');assert passfile(ROOT/'final_integrity.json');break
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(i):int(f) for i,f in (r.split(',') for r in rows) if int(i) in (4,5,6,7)}
        telemetry=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
        for line in telemetry:
            try:pid,used=map(int,line.split(','))
            except ValueError:continue
            for k,w in active.items():
                if w['PID']==pid:memory[jobs[k]['kind']]=max(memory.get(jobs[k]['kind'],0),used)
        choices=[g for g in free if g not in {w['gpu'] for w in active.values()}];probe_pass=passfile(ROOT/'audits/implementation_check.json')
        for k,j in jobs.items():
            if k in active or done(j):continue
            if j['kind']!='probe' and not probe_pass:continue
            if j['kind']=='train' and not gate.exists():continue
            if j['kind']=='init' and not probe_pass:continue
            if j['kind']=='eval':
                if j['method']!='original' and j['method'] not in cfg['checkpoints']:continue
                if not j['step0'] and j['method'] not in METHODS[:3] and not gate.exists():continue
                if j['method'] in METHODS[3:] and not trained(j['method'].removesuffix('_1000')):continue
            peak=memory.get(j['kind'],0)
            if j['kind'] in ('init','train') and probe_pass:peak=max(peak,load(ROOT/'audits/implementation_check.json')['peak_memory_MiB'])
            minimum=max(14000,int(peak*1.25+2048));valid=[g for g in choices if free[g]>=minimum]
            if not valid or len(active)>=concurrency:continue
            gpu=max(valid,key=lambda g:free[g]);choices.remove(gpu);argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)];command(argv);log=ROOT/'logs'/f'{k}.log';offset=log.stat().st_size if log.exists() else 0;stream=log.open('a');proc=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active[k]=dict(PID=proc.pid,start_identity=identity(proc.pid),gpu=gpu,log=str(log),log_offset=offset,required_free_MiB=minimum,proc=proc,stream=stream);attempts[k]=attempts.get(k,0)+1;events.append(dict(event='start',task=k,PID=proc.pid,gpu=gpu,unix=time.time()));print('START',k,'GPU',gpu,flush=True)
        phase='training' if any(jobs[k]['kind']=='train' for k in active) else 'evaluation' if gate.exists() else 'step0_or_implementation_verification';progress(phase if active else 'WAITING_GPU_MEMORY');time.sleep(10)
    progress('publication');cpu('publish_github.py');pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS';s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',publication=pub,active={});dump(ROOT/'pipeline_status.json',s);print('PIPELINE COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        s=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {};s.update(status='FAIL',traceback=traceback.format_exc());dump(ROOT/'pipeline_status.json',s);dump(ROOT/'blocking_error.json',dict(status='FAIL',traceback=s['traceback'],updated_unix=time.time()));raise
