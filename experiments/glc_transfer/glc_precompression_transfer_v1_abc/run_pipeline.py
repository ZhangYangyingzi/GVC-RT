"""Single-instance, bounded resource admission and resumable fixed task pipeline."""
from preflight import *
from data import load
import fcntl
PY=str(Path(sys.executable));LOG=R/'logs';CHILDREN=[]
def command(name,argv,gpu=None):
 LOG.mkdir(exist_ok=True);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 if gpu is not None:env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 with (LOG/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=argv,gpu=gpu,time=time.time()))+'\n')
 log=(LOG/f'{name}.log').open('a');p=subprocess.Popen(argv,cwd=GLC,env=env,stdout=log,stderr=subprocess.STDOUT);CHILDREN.append(p);return p,log
def wait(p,log,name):
 while p.poll() is None:time.sleep(30)
 log.close()
 if p.returncode:raise RuntimeError(f'{name} failed with exit {p.returncode}; see logs/{name}.log')
def gpu_free():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):float(c)-max(2048,float(b)*.1) for a,b,c in (x.split(',') for x in raw.splitlines()) if int(a) in (4,5,6,7)}
def host_free():
 return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
def admit(need=9000,excluded=()):
 limit=time.time()+3600
 while time.time()<limit:
  free={g:v for g,v in gpu_free().items() if g not in excluded and v>need}
  if free and host_free()>8000:return max(free,key=free.get)
  time.sleep(30)
 raise RuntimeError('GPU4–7 resource admission timeout after 3600 seconds; other jobs preserved')
def state(phase,**extra):save('status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),time=time.time(),**extra))
def main():
 lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(R/'publication_status.json')['status']=='PUBLISHED'
 if not (R/'plans/shared.json').exists():
  state('data');p,l=command('data',[PY,'-B','-u',str(R/'data.py')]);wait(p,l,'data')
 assert load(R/'audits/gradient.json')['status']=='PASS' and load(R/'audits/causal64.json')['status']=='PASS'
 assert load(R/'audits/identity.json')['status']=='PASS'
 assert load(R/'results/points/ulong_0_A_original_0.json')['status']=='PASS'
 if not (R/'calibration/final.json').exists():
  state('calibration');g=admit();p,l=command('calibration',[PY,'-B','-u',str(R/'train.py'),'--calibrate','--gpu',str(g)],g);wait(p,l,'calibration')
 if load(R/'publication_status.json').get('phase') not in ('preparation-verified','training','final'):
  p,l=command('publish_ready',[PY,'-B',str(R/'publish.py'),'--phase','preparation-verified']);wait(p,l,'publish_ready')
 state('training');running=[]
 for b in ('B_rgb_P','C_latent_P'):
  if (R/f'branches/{b}/integrity.json').exists():continue
  g=admit(excluded=[x[3] for x in running]);p,l=command(b,[PY,'-B','-u',str(R/'train.py'),'--branch',b,'--gpu',str(g)],g);running.append((p,l,b,g))
 retried=set()
 while running:
  for w in list(running):
   p,l,b,g=w
   if p.poll() is None:continue
   l.close();running.remove(w)
   if p.returncode:
    tail=(LOG/f'{b}.log').read_text(errors='replace')[-5000:]
    if 'out of memory' not in tail.lower() or b in retried:raise RuntimeError(f'{b} failed; logs/{b}.log')
    retried.add(b);g=admit(need=11000,excluded=[x[3] for x in running]);p,l=command(b,[PY,'-B','-u',str(R/'train.py'),'--branch',b,'--gpu',str(g)],g);running.append((p,l,b,g))
  time.sleep(30)
 if load(R/'publication_status.json').get('phase') not in ('training','final'):
  p,l=command('publish_training',[PY,'-B',str(R/'publish.py'),'--phase','training']);wait(p,l,'publish_training')
 state('evaluation')
 from evaluation_stage import run as evaluate_all
 done,expected=evaluate_all(command,state,PY)
 state('report');p,l=command('report',[PY,'-B',str(R/'report.py')]);wait(p,l,'report')
 state('publication');p,l=command('publish_final',[PY,'-B',str(R/'publish.py'),'--phase','final']);wait(p,l,'publish_final')
 save('status.json',dict(status='COMPLETE',terminal=True,B_updates=1000,C_updates=1000,evaluation_completed=done,evaluation_expected=expected,publication=load(R/'publication_status.json')))
if __name__=='__main__':
 try:main()
 except Exception:
  for p in CHILDREN:
   if p.poll() is None:p.terminate()
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',error=e));save('status.json',dict(status='BLOCKED',terminal=True,error=e));print(e);sys.exit(1)
