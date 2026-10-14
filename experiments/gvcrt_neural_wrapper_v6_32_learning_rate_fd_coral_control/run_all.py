"""Recoverable v6.32 driver, independent work despite explicit full-view blockers."""
from io32 import *
import subprocess,fcntl,traceback
def resources():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 free={int(r.split(',')[0]):float(r.split(',')[1]) for r in rows if int(r.split(',')[0]) in (4,5,6,7)}
 mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()};return free,mem['MemAvailable']/1024
def main():
 lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=load(ROOT/'launch.json') if (ROOT/'launch.json').exists() else {};launch.update(PID=os.getpid());dump(ROOT/'launch.json',launch)
 assert load(ROOT/'audits/common.json')['status']=='PASS'
 bad=load(ROOT/'audits/full_resolution_conflicts.json')['conflicts']
 assert bad,'Complete-resolution path must implement and pass full calibration gate before admitting C/D'
 for b in BRANCHES[2:]:dump(ROOT/'branches'/b/'training_status.json',dict(status='BLOCKED',adaptation_step=0,committed_updates=0,reason='FULL_RESOLUTION_MISMATCH in exact required shared plan; no substitution',affected_plan_indices=[x['plan_index'] for x in bad]))
 dump(ROOT/'calibration/final.json',dict(status='BLOCKED',completed_windows=0,expected_windows=40,reason='Required shared GT/EMA seed cannot be complete for four native-resolution windows',missing_plan_indices=[x['plan_index'] for x in bad]))
 # No new coefficients or training of distribution branches with incomplete references.
 pending=[dict(kind='train',branch=BRANCHES[0],preferred=4),dict(kind='train',branch=BRANCHES[1],preferred=5),dict(kind='prefill',shard=0,preferred=6),dict(kind='prefill',shard=1,preferred=7)]
 running=[];failed=[];last_hour=None;prefill_aggregated=(ROOT/'prefill_integrity.json').exists();evaluation_prepared=False;eval_queue=[];completed_eval=set()
 # Existing logs/receipts are never considered complete solely because they exist.
 from evaluate32 import task_id,valid,prepare_baselines
 prepare_baselines();tasks=load(ROOT/'task_manifest.json')['tasks'];active_tasks=[t for t in tasks if t['stage']=='baseline' or t.get('branch') in BRANCHES[:2]]
 attempts={};error_count=0
 while True:
  now=time.time()
  for w in list(running):
   rc=w['proc'].poll()
   if rc is None:continue
   w['log'].close();running.remove(w);j=w['job']
   if rc:
    error_count+=1;logtext=Path(w['logpath']).read_text(errors='replace')[-6000:];error(dict(job=j,returncode=rc,tail=logtext));key=json.dumps(j,sort_keys=True);attempts[key]=attempts.get(key,0)+1
    if 'out of memory' in logtext.lower() and attempts[key]<=1:pending.insert(0,j);continue
    failed.append(dict(job=j,returncode=rc,error=logtext));continue
   if j['kind']=='evaluation':completed_eval.add(task_id(j['task']))
  trainstate={b:load(ROOT/'branches'/b/'training_status.json') if (ROOT/'branches'/b/'training_status.json').exists() else dict(status='PENDING',adaptation_step=0) for b in BRANCHES}
  if not prefill_aggregated and not any(j['kind']=='prefill' for j in pending) and not any(w['job']['kind']=='prefill' for w in running):
   if not any(x['job']['kind']=='prefill' for x in failed):
    subprocess.run([PYTHON,'-B',str(ROOT/'statistics32.py'),'--aggregate'],cwd=REPO,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True);prefill_aggregated=True
  queued={task_id(j['task']) for j in pending if j['kind']=='evaluation'}|{task_id(w['job']['task']) for w in running if w['job']['kind']=='evaluation'}
  failedids={task_id(x['job']['task']) for x in failed if x['job']['kind']=='evaluation'}
  for t in active_tasks:
   tid=task_id(t)
   if tid in completed_eval or tid in queued or tid in failedids:continue
   if valid(t):completed_eval.add(tid);continue
   if t['stage']=='new':
    ix=ROOT/'branches'/t['branch']/'checkpoint_index.json'
    if not ix.exists() or str(t['step']) not in load(ix):continue
   pending.append(dict(kind='evaluation',task=t,preferred=6))
  free,host=resources();occupied={w['gpu'] for w in running}
  # One large measured worker per device. Increase only after measured resource margin exists.
  for j in list(pending):
   gpus=[g for g in (j['preferred'],4,5,6,7) if g not in occupied]
   common=load(ROOT/'audits/common.json');measured=common['peak_memory_MiB']
   # Admission baseline is the historical measured full-resolution protocol, not GPU utilization.
   if j['kind'] in ('train','prefill'):need=max(18000.,measured*1.25)+2048
   else:need=14000.
   gs=[g for g in gpus if free[g]>=need and host>=32768]
   if not gs:continue
   g=gs[0]
   if j['kind']=='train':
    if trainstate[j['branch']]['adaptation_step']>=1000:pending.remove(j);continue
    args=[str(ROOT/'train.py'),'--branch',j['branch'],'--gpu',str(g)]
   elif j['kind']=='prefill':args=[str(ROOT/'statistics32.py'),'--shard',str(j['shard']),'--shards','2','--gpu',str(g)]
   else:args=[str(ROOT/'evaluate32.py'),'--task',json.dumps(j['task']),'--gpu',str(g)]
   tag=j.get('branch',str(j.get('shard',task_id(j['task']) if 'task' in j else '')));lp=ROOT/'logs'/f'{j["kind"]}_{tag}.log';log=lp.open('a')
   argv=[PYTHON,'-B','-u',*args];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   append(ROOT/'logs/commands.jsonl',dict(argv=argv,cwd=str(REPO),time=now,gpu=g));p=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);running.append(dict(proc=p,job=j,gpu=g,log=log,logpath=str(lp)));pending.remove(j);occupied.add(g)
  counts=dict(logical_completed=len(completed_eval),expected=3312,new_completed=sum(t['stage']=='new' and task_id(t) in completed_eval for t in active_tasks),new_expected=2392,baseline_completed=sum(t['stage']=='baseline' and task_id(t) in completed_eval for t in active_tasks),baseline_expected=920)
  status=dict(status='RUNNING_WITH_C_D_BLOCKED',terminal=False,driver_PID=os.getpid(),heartbeat_unix=now,training={b:s.get('adaptation_step',0) for b,s in trainstate.items()},evaluation=counts,running=[dict(PID=w['proc'].pid,gpu=w['gpu'],job=w['job']) for w in running],pending=len(pending),failed=len(failed),error_count=error_count,prefill_aggregated=prefill_aggregated,calibration='BLOCKED',report='PENDING',publication='PENDING')
  dump(ROOT/'status.json',status)
  hour=int(now//3600)
  if hour!=last_hour:
   last_hour=hour
   with (ROOT/'hourly_summary.txt').open('a') as f:f.write(json.dumps(dict(time=now,training=status['training'],evaluation=counts,failed=len(failed)))+'\n')
  trainterminal=all(trainstate[b]['adaptation_step']>=1000 or any(x['job'].get('branch')==b and x['job']['kind']=='train' for x in failed) for b in BRANCHES[:2])
  if not running and not pending and trainterminal:
   # All possible tasks have completed or have recorded concrete failures/blockers.
   from report32 import finalize
   finalize(failed)
   status.update(report='COMPLETE',publication='RUNNING');dump(ROOT/'status.json',status)
   cp=subprocess.run([PYTHON,'-B',str(ROOT/'publish32.py')],cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
   pub=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else dict(status='FAIL',error='publisher returned '+str(cp.returncode))
   status.update(status='BLOCKED',terminal=True,report='COMPLETE',publication=pub['status'],unresolved=['Four fixed full-frame source windows are not 1920x1080; C/D and complete EMA/calibration blocked'],heartbeat_unix=time.time());dump(ROOT/'status.json',status);dump(ROOT/'BLOCKED.json',status);return
  time.sleep(60)
if __name__=='__main__':
 try:main()
 except Exception:
  error(dict(stage='driver',error=traceback.format_exc()));s=load(ROOT/'status.json') if (ROOT/'status.json').exists() else {};s.update(status='FAILED',terminal=True,driver_PID=os.getpid(),error=traceback.format_exc());dump(ROOT/'status.json',s);dump(ROOT/'FAILED.json',s);raise
