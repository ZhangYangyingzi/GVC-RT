"""Restartable bounded GPU queue. Smoke gates precede all full100 endpoints."""
from io32 import *
import subprocess,fcntl

def events():return [json.loads(x) for x in (ROOT/'logs/task_events.jsonl').read_text().splitlines()] if (ROOT/'logs/task_events.jsonl').exists() else []
def done(key):
 if key=="pretrain_objective_audit":return (ROOT/"audits/objective_gradient_step0.json").exists()
 if key.startswith('smoke_'):
  b=key[6:];p=ROOT/f'branches/{b}/training_status.json';return p.exists() and load(p)['committed_updates']>=2
 if key.startswith('train_'):
  p=ROOT/f'branches/{key[6:]}/final_integrity.json';return p.exists() and load(p)['status']=='PASS' and load(p)['committed_updates']==100
 return (ROOT/f'tasks/{key}.json').exists() and load(ROOT/f'tasks/{key}.json')['status']=='PASS'
def active():
 ev=events();out={}
 for e in ev:
  if e['event']=='START':out[e['PID']]=e
  else:out.pop(e['PID'],None)
 return [e for pid,e in out.items() if Path(f'/proc/{pid}/cmdline').exists() and str(ROOT/'worker.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes() and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z']
def queue(jobs):
 failed=set()
 while True:
  running=active();occupied={e['gpu'] for e in running};pending=[(k,c) for k,c in jobs if not done(k) and k not in failed]
  if not pending and not any(e['key'] in dict(jobs) for e in running):break
  snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  append(ROOT/'logs/gpu_snapshots.jsonl',dict(time=time.time(),snapshot=snap,active=running))
  quarantined={e['gpu'] for e in events() if e['event']=='EXIT' and e['exit_code']!=0 and (ROOT/f"logs/{e['key']}.log").exists() and 'OutOfMemoryError' in (ROOT/f"logs/{e['key']}.log").read_text()}
  if quarantined:dump(ROOT/'audits/gpu_quarantine.json',dict(physical_GPUs=sorted(quarantined),reason='external GPU occupancy increased after launch and caused OOM; preserve original precision/resolution/frames and retry on another eligible GPU',time=time.time()))
  available=[int(line.split(',')[0]) for line in snap.splitlines() if int(line.split(',')[0]) in (4,5,6,7) and int(line.split(',')[1])>10000 and int(line.split(',')[0]) not in occupied|quarantined]
  for key,cmd in pending:
   if any(e['key']==key for e in running):continue
   if cmd[0]=='evaluate.py' and cmd[cmd.index('--method')+1]!='O_original':
    method=cmd[cmd.index('--method')+1]
    if not (ROOT/f'branches/{method}/final_integrity.json').exists():
     if f'train_{method}' in failed:failed.add(key)
     continue
   ev=events();repairs=load(ROOT/'audits/retry_after_repair.json') if (ROOT/'audits/retry_after_repair.json').exists() else {};after=repairs.get(key,{}).get('time',0);n=sum(e['event']=='START' and e['key']==key and e['start']>=after for e in ev)
   if n>=3:failed.add(key);continue
   if not available:break
   gpu=available.pop();log=(ROOT/f'logs/supervise_{key}.log').open('a')
   p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'worker.py'),'--key',key,'--gpu',str(gpu),*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
   print('LAUNCH',key,'GPU',gpu,'PID',p.pid,flush=True)
  dump(ROOT/'run_status.json',dict(status='RUNNING',phase=CURRENT_PHASE,complete=sum(done(k) for k,c in jobs),planned=len(jobs),active=running,failed=sorted(failed),time=time.time()))
  time.sleep(10)
 if failed:append(ROOT/'errors.jsonl',dict(stage=CURRENT_PHASE,exhausted_tasks=sorted(failed),time=time.time()))
 return failed
if __name__=='__main__':
 lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':
  print('ALREADY_COMPLETE; no training or evaluation repeated');sys.exit(0)
 CURRENT_PHASE='pretrain_objective_audit'
 failed=queue([('pretrain_objective_audit',['train.py','--branch','G_dists','--stop','0'])])
 if failed:raise RuntimeError('step0 diagnostic failed; formal optimizer updates prohibited')
 CURRENT_PHASE='smoke_and_stream_probe';smokes=[(f'smoke_{b}',['train.py','--branch',b,'--stop','2']) for b in ('G_basic','G_dists')];probe=('O_original_v0_q0',['evaluate.py','--method','O_original','--video','0','--qp','0'])
 failed=queue(smokes+[probe])
 if failed:
  dump(ROOT/'run_status.json',dict(status='BLOCKED_PREFLIGHT',failed=sorted(failed)));sys.exit(1)
 audits={b:load(ROOT/f'branches/{b}/step0_audit.json') for b in ('G_basic','G_dists')}
 assert len({json.dumps(x['initial_module_hashes'],sort_keys=True) for x in audits.values()})==1
 assert len({x['plan_sha256'] for x in audits.values()})==1
 assert len({x['source_RNG_hash'] for x in audits.values()})==1
 dump(ROOT/'audits/smoke_and_step0.json',dict(status='PASS',step0=audits,formal_first2_updates_reused=True,extra_optimizer_updates=0,stream_probe='O_original_v0_q0',identical_initial_modules=True))
 CURRENT_PHASE='train100_and_evaluation'
 jobs=[(f'train_{b}',['train.py','--branch',b,'--stop','100']) for b in ('G_basic','G_dists')]
 evaljobs=[]
 for m in ('O_original','G_basic','G_dists'):
  for v in (0,1):
   for q in (0,5,9):evaljobs.append((f'{m}_v{v}_q{q}',['evaluate.py','--method',m,'--video',str(v),'--qp',str(q)]))
 failed=queue(jobs+evaljobs)
 dump(ROOT/'run_status.json',dict(status='COMPUTE_COMPLETE' if not failed else 'PARTIAL',failed=sorted(failed),RD_complete=sum(done(k) for k,c in evaljobs),RD_planned=18,time=time.time()))
 print('COMPUTE_FINISHED',flush=True)
