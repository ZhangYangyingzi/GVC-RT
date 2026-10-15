"""Memory-admitted independent A points; drain before final evaluation."""
from preflight import *
from data import load
import fcntl

def main():
 lock=(R/'early_original.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 methods=('B_rgb_P','C_latent_P');running=[];done=0;retried=set();excluded_gpu={};need=9458*1.05+512;max_workers=6;resource_deadline=time.time()+3600
 tasks=[dict(dataset=d,video=v,branch='A_original',q=q) for d,vs in load(R/'eval_manifest.json').items() for v in range(len(vs)) for q in reversed(load(R/'protocol.json')['q_grid'])]
 def snapshot(status,**extra):save('early_original_status.json',dict(status=status,PID=os.getpid(),completed=done,running=[dict(task=t,gpu=g,PID=p.pid) for p,l,t,g,tic in running],time=time.time(),**extra))
 def valid(t):
  name=f"{t['dataset']}_{t['video']}_A_original_{t['q']}";path=R/'results/points'/f'{name}.json'
  if not path.exists():return False
  x=load(path);return x.get('status')=='PASS' and x['signature']['evaluation']==sha(R/'evaluate.py') and x['signature']['runtime']==sha(R/'runtime.py') and x['signature']['manifest']==sha(R/'eval_manifest.json') and x['signature']['protocol']==sha(R/'protocol.json') and all(Path(p).exists() and sha(p)==h for p,h in x['artifacts'].items())
 try:
  while tasks or running:
   driver=load(R/'status.json');updates=[load(R/f'branches/{m}/status.json')['updates'] for m in methods]
   allow=driver.get('phase')=='training' and driver.get('status')=='RUNNING' and max(updates)<850
   if not allow:tasks=[]
   for w in list(running):
    p,l,t,g,tic=w
    if p.poll() is None:
     if time.time()-tic>900:
      p.terminate();p.wait(timeout=30);raise RuntimeError(f'Original point exceeded bounded 900s deadline: {t}')
     continue
    running.remove(w);l.close();name=f"{t['dataset']}_{t['video']}_A_original_{t['q']}"
    if p.returncode:
     tail=(R/'logs'/f'early_{name}.log').read_text(errors='replace')[-5000:]
     reference_failure='AssertionError' in tail and '==hashes[i]' in tail
     if ('out of memory' in tail.lower() or reference_failure) and name not in retried and allow:
      retried.add(name);excluded_gpu[name]=g;need=max(need,14000 if reference_failure else 12000);max_workers=max(1,max_workers-1);tasks.insert(0,t)
      save(f'audits/early_failure_{name}.json',dict(task=t,GPU=g,error=tail,policy='one retry on different GPU; strict reference checks unchanged',time=time.time()));snapshot('RUNNING',retry_task=name);continue
     raise RuntimeError(f'{name} exit {p.returncode}; logs/early_{name}.log')
    assert valid(t);point=load(R/'results/points'/f'{name}.json');need=max(need,point['result']['peak_MiB']*1.05+512);done+=1
    snapshot('RUNNING',admission_MiB=need)
   if tasks:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(a):float(c)-max(2048,float(b)*.1) for a,b,c in (x.split(',') for x in raw.splitlines()) if int(a) in (4,5,6,7)}
    # Reserve each admitted process's full measured peak even during model
    # loading/encoding, before the larger metric/decoder allocation is visible.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    allocated={int(a):float(b) for a,b in (x.split(',') for x in apps.splitlines()) if a.strip().isdigit() and b.strip().isdigit()}
    for p,l,t,g,tic in running:free[g]-=max(0,need-allocated.get(p.pid,0))
    host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
    for g in sorted(free,key=free.get,reverse=True)*2:
     if not tasks or len(running)>=max_workers or sum(w[3]==g for w in running)>=2 or free[g]<need or host<12000:continue
     idx=next((i for i,t in enumerate(tasks) if excluded_gpu.get(f"{t['dataset']}_{t['video']}_A_original_{t['q']}")!=g),None)
     if idx is None:continue
     t=tasks.pop(idx)
     while valid(t):
      idx=next((i for i,next_t in enumerate(tasks) if excluded_gpu.get(f"{next_t['dataset']}_{next_t['video']}_A_original_{next_t['q']}")!=g),None)
      if idx is None:t=None;break
      t=tasks.pop(idx)
     if t is None:break
     name=f"{t['dataset']}_{t['video']}_A_original_{t['q']}";argv=[sys.executable,'-B','-u',str(R/'evaluate.py'),'--task',json.dumps(t),'--gpu',str(g)]
     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
     with (R/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=argv,gpu=g,time=time.time(),phase='early_original'))+'\n')
     l=(R/'logs'/f'early_{name}.log').open('a');p=subprocess.Popen(argv,cwd=GLC,env=env,stdout=l,stderr=subprocess.STDOUT);running.append((p,l,t,g,time.time()));free[g]-=need;host-=5000;resource_deadline=time.time()+3600;snapshot('RUNNING',admission_MiB=need,max_workers=max_workers)
    if tasks and not running and time.time()>resource_deadline:raise RuntimeError('Early A admission timeout after 3600 seconds; main pipeline preserved')
   time.sleep(30)
  snapshot('COMPLETE',reason='all A tasks done or drained before B/C update850; main pipeline resumes remaining points')
 except Exception:
  for p,l,t,g,tic in running:
   if p.poll() is None:p.terminate()
   l.close()
  snapshot('BLOCKED',error=traceback.format_exc());raise
if __name__=='__main__':main()
