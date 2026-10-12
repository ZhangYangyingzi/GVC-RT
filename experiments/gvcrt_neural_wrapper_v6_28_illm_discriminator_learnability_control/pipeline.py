"""Persistent D-only DAG, startup reservations, atomic outputs and isolated publication."""
from io21 import *
import fcntl,traceback,signal
BR=('A_raw_current','B_rms_current','C_zero_current','D_raw_official_optimizer')
def log(x):
 with (ROOT/'gpu_schedule.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),**x))+'\n')
def gpu_state():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);return {int(z[0]):dict(free=float(z[1]),total=float(z[2]),used=float(z[3]),util=float(z[4])) for line in out.splitlines() if (z:=[s.strip() for s in line.split(',')]) and int(z[0]) in (4,5,6,7)}
def observed():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True);d={}
 for line in out.splitlines():
  try:p,v=line.split(',');d[int(p)]=d.get(int(p),0)+float(v)
  except ValueError:pass
 return d

def tasks():
 cfg=load(ROOT/'config.json');plan=load(ROOT/'plans/shared.json')['updates'];val=load(ROOT/'plans/validation.json')['updates'];tiny=[w['row_id'] for w in load(ROOT/'plans/tiny.json')['windows']];cachekeys=list(dict.fromkeys(tiny+[f'main_{r["adaptation_step"]:04d}' for r in plan[:64]]+['val_'+r['panel_id'] for r in val]+[f'main_{r["adaptation_step"]:04d}' for r in plan[64:]]));jobs=[]
 # Cache producers use batches: enough pipeline depth for all MAIN consumers.
 for i in range(0,len(cachekeys),8):
  keys=cachekeys[i:i+8];jobs.append(dict(id='cache_'+hashlib.sha256('|'.join(keys).encode()).hexdigest()[:12],kind='cache',keys=keys,argv=['cache_worker.py','--keys',*keys],priority=0 if i<48 else 4,complete=all((ROOT/'cache/status'/f'{k}.json').exists() for k in keys)))
 for b in BR:
  for stage in ['TINY','MAIN']:
   base=ROOT/'branches'/b/stage;st=load(base/'status.json') if (base/'status.json').exists() else {};ready=all((ROOT/'cache/status'/f'{k}.json').exists() for k in (tiny if stage=='TINY' else ['main_0001']));jobs.append(dict(id='train_'+b+'_'+stage,kind='train',argv=['train_D.py','--branch',b,'--stage',stage],priority=1,ready=ready,complete=(base/'integrity.json').exists() and load(base/'integrity.json')['status']=='PASS'))
   idx=load(base/'checkpoint_index.json') if (base/'checkpoint_index.json').exists() else {}
   for step in cfg['stages'][stage]['checkpoints']:
    panels=['tiny_fit']+['val_'+r['panel_id'] for r in val if r['panel']=='unseen'] if stage=='TINY' else ['val_'+r['panel_id'] for r in val]
    for off in range(0,len(panels),10):
     batch=panels[off:off+10];dests=[ROOT/'validation'/b/stage/f'step_{step:05d}'/panel/'integrity.json' for panel in batch];needed=[k for panel in batch for k in (tiny if panel=='tiny_fit' else [panel])];available=all((ROOT/'cache/status'/f'{k}.json').exists() for k in needed);jobs.append(dict(id=f'val_{b}_{stage}_{step}_batch{off}',kind='validation',argv=['validate_D.py','--branch',b,'--stage',stage,'--step',str(step),'--panel',*batch],priority=3 if step==0 else 5,ready=str(step) in idx and available,complete=all(dest.exists() and load(dest)['status']=='PASS' for dest in dests)))
 return jobs

def main():
 lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=load(ROOT/'config.json');pre=load(ROOT/'audits/preflight.json');mem={k:max(648.,pre['cache_peak_MiB' if k=='cache' else 'D_peak_MiB'])*1.15+1024 for k in ('cache','train','validation')};active={};attempts={};blocked={};statefile=ROOT/'scheduler_state.json'
 if statefile.exists():
  old=load(statefile);mem.update({k:max(v,mem[k]) for k,v in old['memory'].items()});attempts=old.get('attempts',{});blocked=old.get('blocked',{})
  for tid,s in old.get('active',{}).items():
   try:
    st=Path('/proc')/str(s['PID'])/'stat';assert st.read_text().split()[21]==s['start_ticks'];assert str(ROOT) in (Path('/proc')/str(s['PID'])/'cmdline').read_text();active[tid]=s
   except (FileNotFoundError,AssertionError):pass
 while True:
  for tid,s in list(active.items()):
   done=ROOT/'task_exit'/f'{tid}.json'
   if done.exists():
    rc=load(done)['exit_code'];log(dict(event='finish',task=tid,**s,exit_code=rc));active.pop(tid)
    if rc!=0:
     data=Path(s['log']).read_text();oom='out of memory' in data.lower();attempts[tid]=attempts.get(tid,0)+1
     if oom:mem[s['kind']]*=1.3
     if attempts[tid]>=3:blocked[tid]=dict(error=data[-6000:],log=s['log'],attempts=attempts[tid])
     log(dict(event='retry' if tid not in blocked else 'blocked',task=tid,OOM=oom,attempts=attempts[tid]))
   elif not (Path('/proc')/str(s['PID'])).exists():active.pop(tid);attempts[tid]=attempts.get(tid,0)+1
  jj=tasks();pending=[j for j in jj if not j['complete']];gs=gpu_state();obs=observed();available={g:v['free']-sum(max(0,s['reserved_MiB']-obs.get(s['PID'],0)) for s in active.values() if s['gpu']==g)-1024 for g,v in gs.items()};max_active=min(32,max(8,os.cpu_count()//4))
  for j in sorted(pending,key=lambda x:x['priority']):
   if j['id'] in active or j['id'] in blocked or not j.get('ready',True) or len(active)>=max_active:continue
   # Limit cache producer host-RAM/I/O pressure; no fixed GPU task count.
   if j['kind']=='cache' and sum(s['kind']=='cache' for s in active.values())>=6:continue
   g=max(available,key=available.get);reserve=mem[j['kind']]
   if available[g]<reserve:continue
   tid=j['id'];done=ROOT/'task_exit'/f'{tid}.json';done.unlink(missing_ok=True);lp=ROOT/'logs/tasks'/f'{tid}_attempt{attempts.get(tid,0)}.log';lp.parent.mkdir(parents=True,exist_ok=True);f=lp.open('a');argv=[PYTHON,str(ROOT/'task_wrapper.py'),'--id',tid,'--gpu',str(g),'--',*j['argv']];command(argv);pr=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();s=dict(PID=pr.pid,start_ticks=(Path('/proc')/str(pr.pid)/'stat').read_text().split()[21],kind=j['kind'],gpu=g,reserved_MiB=reserve,log=str(lp),started_unix=time.time(),argv=argv,attempt=attempts.get(tid,0));active[tid]=s;available[g]-=reserve;log(dict(event='start',task=tid,free_before_MiB=gs[g]['free'],logical_gpu=0,**s))
  dump(statefile,dict(active=active,memory=mem,attempts=attempts,blocked=blocked));training={b:{st:(load(ROOT/'branches'/b/st/'status.json') if (ROOT/'branches'/b/st/'status.json').exists() else {'status':'PENDING','effective_updates':0}) for st in ('TINY','MAIN')} for b in BR};status=dict(status='RUNNING' if active or any(j['id'] not in blocked for j in pending) else 'BLOCKED' if blocked else 'RUNNING',phase='D_training_validation',no_G_updates=True,training=training,completed_tasks=len(jj)-len(pending),expected_tasks=len(jj),active=active,cache_completed=len(list((ROOT/'cache/status').glob('*.json'))) if (ROOT/'cache/status').exists() else 0,validation_completed=sum(j['complete'] for j in jj if j['kind']=='validation'),validation_expected=sum(j['kind']=='validation' for j in jj),blocked=blocked);dump(ROOT/'pipeline_status.json',status)
  if not pending:break
  if blocked and not active and not any(j['id'] not in blocked and j.get('ready',True) for j in pending):return
  time.sleep(5)
 # Summarize CPU-only, then publish and verify without changing any historical results.
 subprocess.check_call([PYTHON,str(ROOT/'summarize.py')],cwd=REPO)
 subprocess.check_call([PYTHON,str(ROOT/'verify_final.py')],cwd=REPO)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs/pipeline_failure.json',dict(status='BLOCKED',traceback=traceback.format_exc()));raise
