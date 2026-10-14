"""Additional memory-admitted formal evaluations; common per-point lock prevents duplication."""
from io32 import *
import argparse,ast,fcntl,subprocess,traceback
from evaluate32 import task_id,receipt,valid,eval_signature
def migrate():
 old=ROOT/'audits/scheduling_before/evaluate32.py';new=ROOT/'evaluate32.py'
 a=old.read_text();b=new.read_text()
 assert a.replace('fcntl.LOCK_EX|fcntl.LOCK_NB','fcntl.LOCK_EX')==b,'Only the scheduling lock wait may change'
 oldhash=sha(old);newhash=sha(new);changed=[]
 for t in load(ROOT/'task_manifest.json')['tasks']:
  p=receipt(t)
  if not p.exists():continue
  r=load(p)
  if r['signature']==eval_signature(t):continue
  x=dict(task=t,config=sha(ROOT/'config.json'),protocol=sha(ROOT/'protocol.json'),manifest=sha(ROOT/'eval_manifest.json'),sources=sha(ROOT/'source_hashes.json'),code=code_signature(['io32.py','evaluate32.py','adapter32.py']))
  x['code']['evaluate32.py']=oldhash
  if t['stage']=='new':x['checkpoint']=load(ROOT/'branches'/t['branch']/'checkpoint_index.json')[str(t['step'])]['inference']
  previous=hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();assert r['signature']==previous
  assert r['status']=='PASS'
  for k,v in r['artifact_metadata'].items():assert sha(k)==v['sha256']
  r.update(scheduling_only_signature_migration=dict(previous_signature=previous,old_source_sha256=oldhash,new_source_sha256=newhash,numerical_code_identical=True,change='blocking per-point lock; no additional encoding or altered metric path'),signature=eval_signature(t));dump(p,r);changed.append(task_id(t))
 dump(ROOT/'audits/parallel_evaluation_admission.json',dict(status='PASS',numeric_code_identical=True,old_sha256=oldhash,new_sha256=newhash,revalidated_receipts=changed,extra_formal_points=0,extra_optimizer_updates=0))
def resources():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(x.split(',')[0]):float(x.split(',')[1]) for x in raw.splitlines() if int(x.split(',')[0]) in (4,5,6,7)}
 mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
 return free,mem['MemAvailable']/1024
def task_lock_available(t):
 p=receipt(t).with_suffix('.lock')
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return True
  except BlockingIOError:return False
def main():
 lock=(ROOT/'parallel_evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=[t for t in load(ROOT/'task_manifest.json')['tasks'] if t['stage']=='baseline' or t.get('branch') in BRANCHES[:2]]
 running=[];failed={};done=set();measured={'baseline':dict(gpu=6000.,rss=4000.),'new':dict(gpu=10000.,rss=6000.)};attempts={}
 while True:
  for w in list(running):
   rc=w['p'].poll()
   if rc is None:continue
   w['log'].close();running.remove(w);tid=task_id(w['task'])
   if rc:
    tail=Path(w['path']).read_text(errors='replace')[-5000:];attempts[tid]=attempts.get(tid,0)+1;error(dict(stage='parallel_evaluation',task=w['task'],returncode=rc,tail=tail))
    if 'out of memory' not in tail.lower() or attempts[tid]>1:failed[tid]=dict(task=w['task'],error=tail)
    else:measured[w['task']['stage']]['gpu']=max(measured[w['task']['stage']]['gpu'],16000.)
   elif valid(w['task']):
    rr=load(receipt(w['task']));done.add(tid)
    if 'peak_memory_MiB' in rr:measured[w['task']['stage']]['gpu']=max(measured[w['task']['stage']]['gpu'],rr['peak_memory_MiB']);measured[w['task']['stage']]['rss']=max(measured[w['task']['stage']]['rss'],rr['peak_RSS_MiB'])
  base=load(ROOT/'status.json');occupied={w['gpu'] for w in running};free,host=resources();inflight={task_id(w['task']) for w in running}
  # Primary queue uses forward order. Reverse order here minimizes overlapping admission.
  for t in reversed(tasks):
   tid=task_id(t)
   if tid in done or tid in inflight or tid in failed:continue
   if valid(t):done.add(tid);continue
   if t['stage']=='new':
    p=ROOT/'branches'/t['branch']/'checkpoint_index.json'
    if not p.exists() or str(t['step']) not in load(p):continue
   if not task_lock_available(t):continue
   if len(running)>=4:break
   reserve=measured[t['stage']];gpus=[]
   for g in (4,5,6,7):
    if g in occupied:continue
    # Remaining free memory already excludes resident allocations. Add the 25% growth
    # margin for both the new worker and measured resident workers, plus 2 GiB/device.
    resident_margin=2500.;need=reserve['gpu']*1.25+resident_margin+2048
    if free[g]>=need and host>=reserve['rss']*1.25+8192:gpus.append(g)
   if not gpus:continue
   g=max(gpus,key=free.get);path=ROOT/'logs'/f'parallel_{tid}.log';log=path.open('a');argv=[PYTHON,'-B','-u',str(ROOT/'evaluate32.py'),'--task',json.dumps(t),'--gpu',str(g)]
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   append(ROOT/'logs/commands.jsonl',dict(argv=argv,cwd=str(REPO),time=time.time(),gpu=g,admission=dict(measured_peak=reserve,remaining_free_MiB=free[g],reserved_MiB=reserve['gpu']*1.25+2500+2048,host_free_MiB=host)))
   p=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);running.append(dict(p=p,task=t,gpu=g,log=log,path=str(path)));occupied.add(g)
  ready_left=[]
  for t in tasks:
   tid=task_id(t)
   if tid in done or tid in failed:continue
   if valid(t):done.add(tid);continue
   ready_left.append(tid)
  terminal=not running and (not ready_left or base.get('terminal',False))
  dump(ROOT/'parallel_evaluation_status.json',dict(status='COMPLETE' if terminal else 'RUNNING',PID=os.getpid(),heartbeat_unix=time.time(),running=[dict(PID=w['p'].pid,gpu=w['gpu'],task=w['task']) for w in running],validated_completed=len(done),active_expected=len(tasks),failures=failed,terminal=terminal,measured_peaks=measured))
  if terminal:return
  time.sleep(60)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--migrate',action='store_true');a=ap.parse_args()
 try:migrate() if a.migrate else main()
 except Exception:error(dict(stage='parallel_evaluation_driver',error=traceback.format_exc()));raise
