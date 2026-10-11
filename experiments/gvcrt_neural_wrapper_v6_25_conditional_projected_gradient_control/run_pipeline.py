"""Persistent task queue with startup reservations, identity checks and staged publishing."""
import fcntl,traceback
from io21 import *

def identity(pid):return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
def alive(pid,start):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split();return s[0]!='Z' and s[19]==start
 except (OSError,ProcessLookupError):return False
def passed(p):return p.exists() and load(p).get('status')=='PASS'

def main():
 if '--status' in sys.argv:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
 os.chdir(REPO);os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);frozen()
 from adapter import validate
 from publish_checkpoints import publish
 prior=load(ROOT/'pipeline_status.json');active={};jobs={};attempts=prior.get('attempts',{});memory=prior.get('memory',{'eval':9000,'train':14000,'probe':14000,'init':5000});events=prior.get('events',[]);blocked=prior.get('blocked',{});checked={};stages=prior.get('published_stages',[])
 def add(k,kind,script,args,**kw):jobs[k]=dict(kind=kind,script=script,args=args,**kw)
 for b in BRANCHES:
  add('check_'+b,'probe','precheck.py',['--branch',b],branch=b)
  add('init_'+b,'init','train.py',['--branch',b,'--initialize-only'],branch=b)
  add('train_'+b,'train','train.py',['--branch',b],branch=b)
  v=sources('uvg_validation')[0];add('codec_'+b,'eval','evaluate.py',['--dataset','uvg_validation','--method',b+'_0','--video',str(v['video_index']),'--qps','0'],method=b+'_0',dataset='uvg_validation',video=v['video_index'],qps=[0],branch=b,smoke=True)
 history=[f'historical_{b}_{s}' for b in ('A_continue','B_projected') for s in (250,500)]
 for d in DATASETS:
  for m in (*METHODS,*history):
   for v in sources(d):
    for q in range(10):add(f'eval_{d}_{m}_{v["video_index"]}_{q}','eval','evaluate.py',['--dataset',d,'--method',m,'--video',str(v['video_index']),'--qps',str(q)],method=m,dataset=d,video=v['video_index'],qps=[q],smoke=False)
 def done(j):
  if j['kind']=='cpu':return passed(ROOT/'historical/final_integrity.json')
  if j['kind']=='probe':return passed(ROOT/'audits'/f'{j["branch"]}_pretraining.json')
  if j['kind']=='init':return (ROOT/'branches'/j['branch']/'checkpoint_index.json').exists() and '0' in load(ROOT/'branches'/j['branch']/'checkpoint_index.json')
  if j['kind']=='train':return passed(ROOT/'branches'/j['branch']/'training_integrity.json')
  v=next(v for v in sources(j['dataset']) if v['video_index']==j['video'])
  for q in j['qps']:
   p=point(j['dataset'],j['method'],j['video'],q)
   if not p.exists():return False
   stamp=(p.stat().st_size,p.stat().st_mtime_ns)
   if checked.get(str(p))!=stamp:
    try:validate(load(p),v,q,j['method'])
    except (AssertionError,ValueError,KeyError,OSError):return False
    checked[str(p)]=stamp
  return True
 add('historical_cpu','cpu','historical_report.py',[])
 for k,w in prior.get('active',{}).items():
  if k in jobs and alive(w['PID'],w['identity']):active[k]=dict(w,proc=None,stream=None)
 # Adopt independently launched checks, preserving their loading reservations.
 for w in load(ROOT/'precheck_launch.json') if (ROOT/'precheck_launch.json').exists() else []:
  k='check_'+w['argv'][-1]
  try:start=identity(w['PID'])
  except OSError:continue
  if k not in active and alive(w['PID'],start):
   log=ROOT/'logs'/('precheck_'+w['argv'][-1]+'.log');active[k]=dict(PID=w['PID'],identity=start,gpu=w['physical_gpu'],required_MiB=w['reservation_MiB'],log=str(log),offset=0,proc=None,stream=None);attempts[k]=attempts.get(k,1)
 def event(r):
  with (ROOT/'gpu_schedule.jsonl').open('a') as f:f.write(json.dumps(dict(r,unix=time.time()))+'\n')
 def state(phase):
  steps={b:load(ROOT/'branches'/b/'training_status.json').get('adaptation_step',0) if (ROOT/'branches'/b/'training_status.json').exists() else 0 for b in BRANCHES}
  maincount=sum(point(d,m,v['video_index'],q).exists() for d in DATASETS for m in METHODS for v in sources(d) for q in range(10));hc=sum(point(d,m,v['video_index'],q).exists() for d in DATASETS for m in history for v in sources(d) for q in range(10))
  dump(ROOT/'pipeline_status.json',dict(status='RUNNING' if active or not blocked else 'BLOCKED',phase=phase,PID=os.getpid(),training_steps=steps,completed_RD=maincount,expected_RD=800,completed_historical=hc,expected_historical=640,active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},attempts=attempts,memory=memory,events=events,blocked=blocked,published_stages=stages,updated_unix=time.time()))
 def cpu(script,phase=None):
  env=os.environ.copy()
  if phase:env['V625_PUBLICATION_PHASE']=phase
  with (ROOT/'logs'/f'{Path(script).stem}.log').open('a') as f:subprocess.run([PYTHON,'-B','-u',str(ROOT/script)],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 while True:
  for k,w in list(active.items()):
   rc=(None if alive(w['PID'],w['identity']) else 0 if done(jobs[k]) else -1) if w['proc'] is None else w['proc'].poll()
   if rc is None:continue
   if w['stream']:w['stream'].close()
   j=jobs[k];del active[k];error=''
   if rc or not done(j):
    error=Path(w['log']).read_text()[w['offset']:][-8000:];oom='out of memory' in error.lower();retry=attempts.get(k,1)<(8 if oom else 3)
    if oom:memory[j['kind']]=max(memory[j['kind']]*1.15,w['required_MiB']*.9)
    if not retry:blocked[k]=dict(error=error,log=w['log'],attempts=attempts.get(k,1))
   else:
    if j['kind']=='probe':peak=load(ROOT/'audits'/f'{j["branch"]}_pretraining.json')['peak_memory_MiB'];memory['train']=max(memory['train'],peak);memory['probe']=max(memory['probe'],peak)
    elif j['kind']=='eval':memory['eval']=max(memory['eval'],*(load(point(j['dataset'],j['method'],j['video'],q)).get('peak_memory_MiB',0) for q in j['qps']))
   event(dict(event='exit',task=k,physical_gpu=w['gpu'],logical_gpu=0,PID=w['PID'],process_start=w['identity'],exit_code=rc,retry_count=attempts.get(k,1)-1,peak_memory_MiB=memory.get(j['kind'],0),error=error));print('FINISH',k,rc,flush=True)
  cfg=publish()
  for b in BRANCHES:
   if done(jobs['codec_'+b]):
    p=ROOT/'audits'/f'{b}_codec_equivalence.json'
    if not p.exists():
     j=jobs['codec_'+b];r=load(point(j['dataset'],j['method'],j['video'],0));assert r['same_GPU_step0_verification']['status']=='PASS';dump(p,dict(status='PASS',branch=b,real_RANS=True,independent_decode=True,source_reproduction=r['same_GPU_step0_verification'],peak_memory_MiB=r['peak_memory_MiB']))
  # CPU decomposition runs beside GPU work and resumes completed dataset/method pools.
  cpu_key='historical_cpu'
  if cpu_key not in active and 'historical_report.py' in [p.name for p in ROOT.glob('*.py')] and not passed(ROOT/'historical/final_integrity.json') and cpu_key not in blocked:
   jobs[cpu_key]=dict(kind='cpu',script='historical_report.py',args=[])
   # CPU worker handled separately to avoid GPU reservation.
   log=ROOT/'logs/historical_report.log';f=log.open('a');p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'historical_report.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);active[cpu_key]=dict(PID=p.pid,identity=identity(p.pid),gpu=None,required_MiB=0,log=str(log),offset=log.stat().st_size,proc=p,stream=f);attempts[cpu_key]=attempts.get(cpu_key,0)+1
  # Publish each training stage without stopping independent workers.
  for phase in ['code_protocol_ready',*[b+'_trained' for b in BRANCHES],'historical_diagnosis_complete']:
   ready=phase=='code_protocol_ready' or (phase.endswith('_trained') and done(jobs['train_'+phase[:-8]])) or (phase=='historical_diagnosis_complete' and passed(ROOT/'historical/final_integrity.json'))
   if ready and phase not in stages:
    state('stage_publication');cpu('publish_github.py',phase);stages.append(phase)
  if not active and all(done(j) for j in jobs.values()) and passed(ROOT/'historical/final_integrity.json'):
   state('aggregate');cpu('report.py');cpu('finish.py');cpu('publish_github.py','final_results');cpu('verify_publication.py');state('complete');s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',publication=load(ROOT/'publication_status.json'));dump(ROOT/'pipeline_status.json',s);print('PIPELINE COMPLETE',flush=True);return
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(i):int(v) for i,v in (l.split(',') for l in lines) if int(i) in (4,5,6,7)};used={}
  for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
   try:pid,mi=map(int,l.split(','));used[pid]=mi
   except ValueError:continue
  for w in active.values():
   if w['gpu'] is not None:free[w['gpu']]-=max(0,w['required_MiB']-used.get(w['PID'],0))
  for k,j in jobs.items():
   if k in active or k in blocked or j['kind']=='cpu' or done(j):continue
   if j['kind']=='train' and not (passed(ROOT/'audits'/f'{j["branch"]}_pretraining.json') and passed(ROOT/'audits'/f'{j["branch"]}_codec_equivalence.json')):continue
   if j['kind']=='eval' and j['method']!='original' and j['method'] not in cfg['checkpoints']:continue
   required=int(memory.get(j['kind'],14000)*1.2+2048);valid=[g for g in free if free[g]>=required]
   if not valid:continue
   gpu=max(valid,key=lambda g:free[g]);before=free[gpu];free[gpu]-=required;argv=[PYTHON,'-B','-u',str(ROOT/j['script']),*j['args'],'--gpu',str(gpu)];command(argv);log=ROOT/'logs'/f'{k}.log';offset=log.stat().st_size if log.exists() else 0;f=log.open('a');p=subprocess.Popen(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);active[k]=dict(PID=p.pid,identity=identity(p.pid),gpu=gpu,required_MiB=required,log=str(log),offset=offset,proc=p,stream=f);attempts[k]=attempts.get(k,0)+1;event(dict(event='start',task=k,kind=j['kind'],method=j.get('method'),branch=j.get('branch'),dataset=j.get('dataset'),video=j.get('video'),qps=j.get('qps'),physical_gpu=gpu,logical_gpu=0,CUDA_VISIBLE_DEVICES=str(gpu),PID=p.pid,process_start=identity(p.pid),free_before_MiB=before,reservation_MiB=required,retry_count=attempts[k]-1));print('START',k,'GPU',gpu,flush=True)
  state('training_and_evaluation')
  if blocked and not active and not any(k not in blocked and not done(j) for k,j in jobs.items()):raise RuntimeError('Blocked tasks: '+json.dumps(blocked))
  time.sleep(10)
if __name__=='__main__':
 try:main()
 except Exception:
  s=load(ROOT/'pipeline_status.json');s.update(status='BLOCKED',error=traceback.format_exc());dump(ROOT/'pipeline_status.json',s);dump(ROOT/'blocking_error.json',dict(status='BLOCKED',error=s['error']));raise
