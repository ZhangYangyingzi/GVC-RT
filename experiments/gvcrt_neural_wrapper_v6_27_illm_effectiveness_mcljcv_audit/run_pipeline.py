"""Persistent no-training queue with PID identities and immediate GPU loading reservations."""
from io21 import *
import fcntl,traceback,math

def identity(pid):return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
def alive(pid,start):
 try:s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split();return s[0]!='Z' and s[19]==start
 except OSError:return False
def passed(p):return p.exists() and load(p).get('status')=='PASS'
def main():
 os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prior=load(ROOT/'pipeline_status.json');active={};attempts=prior.get('attempts',{});blocked=prior.get('blocked',{});memory=prior.get('memory',dict(cache=6000,diagnostic=14000,eval=11000));measured=prior.get('measured',{});jobs={};stages=prior.get('stages',[]);checkcache={}
 def event(r):
  with (ROOT/'gpu_schedule.jsonl').open('a') as f:f.write(json.dumps(dict(r,unix=time.time()))+'\n')
 def addjobs():
  if passed(ROOT/'audits/diagnostic_preparation.json'):
   for r in load(ROOT/'diagnostic_plan.json')['updates']:
    panel=r['panel_id'];jobs['cache_'+panel]=dict(kind='cache',script='diagnose.py',args=['--panel',panel,'--cache-only'],panel=panel)
    for step in (0,250,500,1000):jobs[f'diag_{panel}_{step}']=dict(kind='diagnostic',script='diagnose.py',args=['--panel',panel,'--step',str(step)],panel=panel,step=step)
  if passed(ROOT/'audits/data_preparation.json'):
   for v in sources('mcljcv'):
    for m in METHODS:
     for q in range(10):jobs[f'eval_{m}_{v["video_index"]}_{q}']=dict(kind='eval',script='evaluate.py',args=['--dataset','mcljcv','--method',m,'--video',str(v['video_index']),'--qps',str(q)],method=m,video=v['video_index'],QP=q)
 def done(j):
  if j['kind']=='cache':return passed(ROOT/'diagnostics/cache_status'/f'{j["panel"]}.json')
  if j['kind']=='diagnostic':return passed(ROOT/'diagnostics/points'/f'{j["panel"]}_s{j["step"]}.json')
  p=point('mcljcv',j['method'],j['video'],j['QP'])
  if not p.exists():return False
  stamp=(p.stat().st_size,p.stat().st_mtime_ns)
  if checkcache.get(str(p))==stamp:return True
  from adapter import validate
  v=next(v for v in sources('mcljcv') if v['video_index']==j['video'])
  try:validate(load(p),v,j['QP'],j['method'])
  except (AssertionError,ValueError,KeyError,OSError):return False
  checkcache[str(p)]=stamp;return True
 def state(phase):
  ev=sum(done(j) for j in jobs.values() if j['kind']=='eval');dg=sum(done(j) for j in jobs.values() if j['kind']=='diagnostic');dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,no_training=True,PID=os.getpid(),completed_RD=ev,expected_RD=len([j for j in jobs.values() if j['kind']=='eval']),completed_diagnostics=dg,expected_diagnostics=len([j for j in jobs.values() if j['kind']=='diagnostic']),active={k:{x:y for x,y in w.items() if x not in ('proc','stream')} for k,w in active.items()},memory=memory,measured=measured,attempts=attempts,blocked=blocked,stages=stages,updated_unix=time.time()))
 addjobs()
 for k,w in prior.get('active',{}).items():
  if k in jobs and alive(w['PID'],w['identity']):active[k]=dict(w,proc=None,stream=None)
 def cpu(script,phase=None):
  env=os.environ.copy()
  if phase:env['V627_PUBLICATION_PHASE']=phase
  with (ROOT/'logs'/f'{Path(script).stem}.log').open('a') as log:subprocess.run([PYTHON,'-B','-u',str(ROOT/script)],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 while True:
  addjobs()
  for k,w in list(active.items()):
   rc=(None if alive(w['PID'],w['identity']) else 0 if done(jobs[k]) else -1) if w['proc'] is None else w['proc'].poll()
   if rc is None:continue
   if w['stream']:w['stream'].close()
   del active[k];j=jobs[k];ok=not rc and done(j);err='';peak=None
   if ok:
    p=ROOT/'diagnostics/cache_status'/f'{j["panel"]}.json' if j['kind']=='cache' else ROOT/'diagnostics/points'/f'{j["panel"]}_s{j["step"]}.json' if j['kind']=='diagnostic' else point('mcljcv',j['method'],j['video'],j['QP']);peak=load(p)['peak_memory_MiB'];memory[j['kind']]=max(peak,memory[j['kind']]);measured[j['kind']]=True
   else:
    err=Path(w['log']).read_text()[w['offset']:][-12000:];oom='out of memory' in err.lower();memory[j['kind']]*=1.25 if oom else 1.
    if attempts.get(k,0)>=(6 if oom else 2):blocked[k]=dict(error=err,log=w['log'],attempts=attempts[k])
   event(dict(event='exit',task_id=k,category=j['kind'],physical_gpu=w['gpu'],logical_gpu=0,PID=w['PID'],process_start=w['identity'],reserved_MiB=w['required_MiB'],measured_peak_MiB=peak,exit_code=rc,retry_count=attempts[k]-1,error=err));print('FINISH',k,rc,flush=True)
  if passed(ROOT/'audits/diagnostic_preparation.json') and 'code_protocol' not in stages and (ROOT/'summarize.py').exists():
   state('code_protocol_publication');cpu('publish_github.py','code_protocol');stages.append('code_protocol')
  if jobs and passed(ROOT/'audits/data_preparation.json') and all(done(j) or k in blocked for k,j in jobs.items()) and not active:
   if blocked:
    state('blocked');s=load(ROOT/'pipeline_status.json');s['status']='BLOCKED';dump(ROOT/'pipeline_status.json',s);print('BLOCKED',list(blocked),flush=True);return
   if not (ROOT/'summarize.py').exists():state('waiting_for_finalizer');time.sleep(15);continue
   state('summarizing');cpu('summarize.py');cpu('plots.py');cpu('visuals.py');cpu('finalize.py');state('final_publication');cpu('publish_github.py','final_results');cpu('verify_final.py');s=load(ROOT/'pipeline_status.json');s.update(status='PASS',phase='complete',active={});dump(ROOT/'pipeline_status.json',s);print('PIPELINE COMPLETE',flush=True);return
  try:
   lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(x.split(',')[0]):float(x.split(',')[1]) for x in lines if int(x.split(',')[0]) in (4,5,6,7)}
   processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();allocated={int(x.split(',')[0]):float(x.split(',')[1]) for x in processes if x.split(',')[1].strip().isdigit()}
  except Exception as e:event(dict(event='GPU_QUERY_ERROR',error=repr(e)));time.sleep(15);continue
  remaining={g:free[g]-sum(max(0,w['required_MiB']-allocated.get(w['PID'],0)) for w in active.values() if w['gpu']==g) for g in free}
  memavailable=float(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1])/1024;maxcpu=max(2,(os.cpu_count() or 8)//3)
  caches_ready=passed(ROOT/'audits/diagnostic_preparation.json') and all(done(j) for j in jobs.values() if j['kind']=='cache')
  ordered=sorted(jobs.items(),key=lambda kv:(0 if kv[1]['kind']=='diagnostic' and not measured.get('diagnostic') else 1 if kv[1]['kind']=='eval' and not measured.get('eval') else 2 if kv[1]['kind']=='cache' else 3 if kv[1]['kind']=='diagnostic' else 4,kv[0]))
  for k,j in ordered:
   if k in active or k in blocked or done(j):continue
   if j['kind']=='diagnostic' and not caches_ready:continue
   if not measured.get(j['kind']) and any(jobs[a]['kind']==j['kind'] for a in active):continue
   if len(active)*2>=maxcpu or (len(active)+1)*3500>memavailable:continue
   req=memory[j['kind']]*1.2+2048;gpu=max(remaining,key=remaining.get)
   if remaining[gpu]<req:continue
   log=ROOT/'logs'/f'{k}.log';offset=log.stat().st_size if log.exists() else 0;stream=log.open('a');argv=[PYTHON,'-B','-u',str(ROOT/j['script']),'--gpu',str(gpu),*j['args']];env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu));proc=subprocess.Popen(argv,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);attempts[k]=attempts.get(k,0)+1;active[k]=dict(PID=proc.pid,identity=identity(proc.pid),gpu=gpu,required_MiB=req,log=str(log),offset=offset,proc=proc,stream=stream,argv=argv);remaining[gpu]-=req
   event(dict(event='launch',task_id=k,category=j['kind'],physical_gpu=gpu,logical_gpu=0,PID=proc.pid,process_start=active[k]['identity'],free_before_MiB=free[gpu],reserved_MiB=req,peak_estimate_MiB=memory[j['kind']],retry_count=attempts[k]-1,argv=argv));print('START',k,'GPU',gpu,flush=True)
  state('running');time.sleep(15)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'pipeline_failure_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
