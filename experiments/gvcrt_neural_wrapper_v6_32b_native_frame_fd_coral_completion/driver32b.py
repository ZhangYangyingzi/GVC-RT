"""Persistent, singleton, reservation-aware pipeline. No LLM or external agent calls."""
from io32 import *
import fcntl,subprocess,traceback,signal

def alive(pid,start=None):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().split();return s[2]!='Z' and (start is None or s[21]==str(start))
 except OSError:return False
def identity(pid):
 try:return dict(PID=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
 except OSError:return dict(PID=pid,start_ticks=None,command='')
def resources():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(x.split(',')[0]):float(x.split(',')[1]) for x in lines if int(x.split(',')[0]) in (4,5,6,7)}
 mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()};return free,mem['MemAvailable']/1024

def main():
 lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from evaluate32 import valid,receipt,task_id
 tasks=load(ROOT/'task_manifest.json')['tasks'];running=[];failed={};attempts={};done=set();tic=time.time();measured=dict(calibration=20000.,train=22000.,evaluation=12500.,statistics=8000.,aggregate=5000.,cpu=0.)
 p0=ROOT/'calibration/receipts/00.json'
 if p0.exists():measured['calibration']=load(p0)['peak_memory_MiB']
 stp=ROOT/'driver_state.json'
 if stp.exists():
  state=load(stp);failed=state.get('failed',{});attempts=state.get('attempts',{});measured.update(state.get('measured',{}))
  for w in state.get('running',[]):
   if alive(w['PID'],w.get('start_ticks')):
    ident=identity(w['PID']);assert str(ROOT) in ident['command'];w['adopted']=True;running.append(w)
   else:append(ROOT/'gpu_schedule.jsonl',dict(time=time.time(),event='RECOVER_DEAD_WORKER',task=w['id'],PID=w['PID']))
 def passed(path):
  try:return load(path).get('status') in ('PASS','PUBLISHED')
  except (OSError,ValueError):return False
 def completed(w):
  category=w['category'];n=w['id']
  if category=='calibration':return passed(ROOT/'calibration/receipts'/f'{w["index"]:02d}.json')
  if category=='train':return passed(ROOT/'branches'/w['branch']/'training_integrity.json')
  if category=='evaluation':return valid(w['task'])
  if n=='aggregate':return passed(ROOT/'reference_stats/index.json')
  if n=='calibration_aggregate':return passed(ROOT/'calibration/final.json')
  if n=='report':return passed(ROOT/'computation_integrity.json')
  if n=='publish':return passed(ROOT/'publication_status.json')
  if n=='reuse':return (ROOT/'reuse_manifest.json').exists()
  if n=='cache_import':return passed(ROOT/'audits/prefill_reuse.json')
  return False
 # Reconcile start records that were fsynced before a controller stopped but before
 # its next state snapshot. Live workers keep their full pending reservations.
 known={w['PID'] for w in running};events=ROOT/'gpu_schedule.jsonl'
 if events.exists():
  for line in events.read_text().splitlines():
   ev=json.loads(line)
   if ev.get('event')=='START' and ev.get('PID') not in known and alive(ev['PID'],ev.get('start_ticks')):
    assert str(ROOT) in identity(ev['PID'])['command'];ev=dict(ev,adopted=True);running.append(ev);known.add(ev['PID'])
 procs={};logs={} 
 def launch(w,gpu):
  argv=w['argv'];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID')
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  path=ROOT/'logs'/f'{w["id"]}.log';f=path.open('a');p=subprocess.Popen(argv,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();procs[p.pid]=p
  reserve=(measured[w['category']]*1.2+2048) if gpu is not None else 0.;ident=identity(p.pid);r=dict(w,**ident,gpu=gpu,physical_GPU=gpu,process_device=0 if gpu is not None else None,reserved_MiB=reserve,started=time.time(),log=str(path),attempt=attempts.get(w['id'],0))
  running.append(r);append(ROOT/'gpu_schedule.jsonl',dict(event='START',**r));return r
 def command(script,*args):return [PYTHON,'-B','-u',str(ROOT/script),*map(str,args)]
 while True:
  for w in list(running):
   pid=w['PID'];p=procs.get(pid);rc=p.poll() if p else (None if alive(pid,w.get('start_ticks')) else -999)
   if rc is None:continue
   running.remove(w);ok=completed(w);peak=0.
   if ok:
    done.add(w['id'])
    path=(ROOT/'calibration/receipts'/f'{w["index"]:02d}.json') if w['category']=='calibration' else (receipt(w['task']) if w['category']=='evaluation' else (ROOT/'branches'/w['branch']/'training_status.json' if w['category']=='train' else None))
    if path and path.exists():peak=load(path).get('peak_memory_MiB',0.);measured[w['category']]=max(peak,measured[w['category']] if w['category'] not in ('calibration',) else peak)
   else:
    tail=Path(w['log']).read_text(errors='replace')[-10000:];attempts[w['id']]=attempts.get(w['id'],0)+1
    oom='out of memory' in tail.lower()
    if oom:measured[w['category']]=max(measured[w['category']]*1.3,24000.)
    if not oom or attempts[w['id']]>3:failed[w['id']]=dict(error=tail,log=w['log'],returncode=rc,task=w)
    error(dict(stage='driver_worker',task=w['id'],returncode=rc,OOM=oom,tail=tail))
   append(ROOT/'gpu_schedule.jsonl',dict(event='END',task=w['id'],PID=pid,physical_GPU=w['gpu'],time=time.time(),exit_code=rc,completed=ok,peak_memory_MiB=peak,retry=attempts.get(w['id'],0)))
  pending=[]
  def enqueue(id,category,argv,**kw):
   w=dict(id=id,category=category,argv=argv,**kw)
   if id not in done and id not in failed and id not in {r['id'] for r in running}:
    if completed(w):done.add(id)
    else:pending.append(w)
  if not passed(ROOT/'audits/prefill_reuse.json'):enqueue('cache_import','cpu',command('statistics32b.py','--import-cache'))
  elif not passed(ROOT/'reference_stats/index.json'):enqueue('aggregate','aggregate',command('statistics32b.py','--aggregate','--gpu',4))
  if not (ROOT/'reuse_manifest.json').exists():enqueue('reuse','cpu',command('reuse32b.py'))
  if passed(ROOT/'reference_stats/index.json'):
   if not passed(ROOT/'calibration/final.json'):
    for i in range(40):enqueue(f'cal_{i:02d}','calibration',command('calibration32b.py','--index',i),index=i)
    if all(passed(ROOT/'calibration/receipts'/f'{i:02d}.json') for i in range(40)):enqueue('calibration_aggregate','cpu',command('calibration32b.py','--aggregate'))
   else:
    for b in BRANCHES[2:]:enqueue('train_'+b,'train',command('train.py','--branch',b),branch=b)
  if (ROOT/'reuse_manifest.json').exists():
   for t in tasks:
    if valid(t):continue
    if t['stage']=='new':
     ix=ROOT/'branches'/t['branch']/'checkpoint_index.json'
     if not ix.exists() or str(t['step']) not in load(ix):continue
    enqueue('eval_'+task_id(t),'evaluation',command('evaluate32.py','--task',json.dumps(t)),task=t)
  # Complete network calibration before admitting additional full evaluation workers.
  free,host=resources()
  # Count actual residency once, and retain the unallocated part of every live
  # reservation. Model-loading tasks therefore reserve their full predicted peak.
  resident={}
  try:
   lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
   for line in lines:
    a=line.split(',')
    if a[0].strip().isdigit() and a[1].strip().isdigit():resident[int(a[0])]=float(a[1])
  except Exception:pass
  observed=[]
  for w in running:
   if w['category']=='train':
    tp=ROOT/'branches'/w['branch']/'training_status.json'
    if tp.exists():
     tr=load(tp)
     if tr.get('PID')==w['PID'] and tr.get('committed_updates',0)>=1:observed.append(tr['peak_memory_MiB'])
  if observed:measured['train']=max(observed)*1.1
  for w in running:
   if w['category']=='train' and observed:
    tp=ROOT/'branches'/w['branch']/'training_status.json'
    if tp.exists() and load(tp).get('PID')==w['PID']:
     reserve=measured['train']*1.2+2048
     if reserve!=w['reserved_MiB']:
      append(ROOT/'gpu_schedule.jsonl',dict(event='RESERVATION_MEASURED_UPDATE',time=time.time(),task=w['id'],PID=w['PID'],physical_GPU=w['gpu'],previous_reserved_MiB=w['reserved_MiB'],reserved_MiB=reserve,measured_full_update_peak_MiB=load(tp)['peak_memory_MiB']))
      w['reserved_MiB']=reserve
  inflight={g:sum(max(0.,w['reserved_MiB']-resident.get(w['PID'],0.)) for w in running if w['gpu']==g) for g in (4,5,6,7)}
  for w in pending:
   cpu=w['category']=='cpu'
   if cpu:
    if sum(x['gpu'] is None for x in running)>=2:continue
    launch(w,None);continue
   if w['category']=='calibration' and 'cal_00' not in done and w['id']!='cal_00':continue
   need=measured[w['category']]*1.2+2048
   # Free memory deducts residency. Subtract only unrealized reservation plus margin;
   # every loading worker with no allocation still reserves its entire peak.
   possible=[g for g in (4,5,6,7) if free[g]-inflight.get(g,0)>=need and host>=12000+sum(9000 for x in running)]
   if not possible:continue
   preferred=4 if w.get('branch')==BRANCHES[2] else 5 if w.get('branch')==BRANCHES[3] else None
   g=preferred if preferred in possible else max(possible,key=lambda k:free[k]-inflight.get(k,0))
   if w['category']!='aggregate':w['argv']+=['--gpu',str(g)]
   else:
    w['argv'][-1]=str(g)
   launch(w,g);inflight[g]=inflight.get(g,0)+need
  evalcount=sum(valid(t) for t in tasks);calcount=sum(passed(ROOT/'calibration/receipts'/f'{i:02d}.json') for i in range(40));updates={b:(load(ROOT/'branches'/b/'training_status.json').get('committed_updates',load(ROOT/'branches'/b/'training_status.json').get('adaptation_step',0)) if (ROOT/'branches'/b/'training_status.json').exists() else 0) for b in BRANCHES}
  calculation=evalcount==3312 and calcount==40 and all(n==1000 for n in updates.values())
  if calculation and not running and not failed:
   if not passed(ROOT/'computation_integrity.json'):launch(dict(id='report',category='cpu',argv=command('completion_report32b.py')),None)
   elif not passed(ROOT/'publication_status.json'):launch(dict(id='publish',category='cpu',argv=command('publish32b.py')),None)
  published=passed(ROOT/'publication_status.json');terminal=published or (not running and bool(failed) and not pending)
  if not running and pending and not failed and time.time()-tic>120:
   impossible=[w for w in pending if measured[w['category']]*1.2+2048>40960]
   if impossible:
    for w in impossible:failed[w['id']]=dict(error='Measured task peak plus required safety reserve exceeds all allowed GPUs; protocol unchanged',task=w)
  state=dict(status='PASS' if published else 'FAILED' if terminal else 'RUNNING',calculation='COMPLETE' if calculation else 'FAILED' if terminal else 'RUNNING',report='COMPLETE' if passed(ROOT/'computation_integrity.json') else 'PENDING',publication='PUBLISHED' if published else 'FAILED' if 'publish' in failed else 'PENDING',heartbeat_unix=time.time(),driver=identity(os.getpid()),training_updates=updates,prefill=1000 if passed(ROOT/'reference_stats/index.json') else sum((ROOT/'reference_stats/receipts'/f'{i:04d}.json').exists() for i in range(1000)),calibration=calcount,evaluation_completed=evalcount,evaluation_expected=3312,running=running,remaining_jobs=len(pending),failures=failed,terminal=terminal,measured_peaks_MiB=measured)
  dump(ROOT/'pipeline_status.json',state);dump(stp,dict(running=running,failed=failed,attempts=attempts,measured=measured,done=list(done)))
  if terminal:return
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except BlockingIOError:
  print('An active v6.32b driver holds the singleton lock; no state modified.',flush=True)
 except Exception:
  error(dict(stage='driver',error=traceback.format_exc()));dump(ROOT/'pipeline_status.json',dict(status='FAILED',calculation='FAILED_DRIVER',report='PENDING',publication='PENDING',terminal=True,error=traceback.format_exc(),heartbeat_unix=time.time()));raise
