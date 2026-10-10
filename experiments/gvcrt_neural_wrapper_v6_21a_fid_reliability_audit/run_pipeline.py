"""Resumable memory-aware evaluation and statistics supervisor; GPU4..7 only."""
from io21 import *
import fcntl,traceback

def gpu_state():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [dict(zip(['gpu','total_MiB','used_MiB','free_MiB','utilization'],map(int,x.split(','))))for x in s.splitlines()if int(x.split(',')[0])in (4,5,6,7)]
def completed():return len(list((ROOT/'parts').glob('*/*/video_*_qp*.json')))
def main():
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(ROOT/'logs').mkdir(exist_ok=True);events=load(ROOT/'gpu_schedule.json')['events']if(ROOT/'gpu_schedule.json').exists()else[];initial=gpu_state();active={};attempts={};peaks={'eval':14000,'stats':2048}
 if not(ROOT/'dataset_integrity.json').exists():raise RuntimeError('prepare.py must complete first')
 if not(ROOT/'fid_reproduction_integrity.json').exists():
  argv=[PYTHON,'-B','-u',str(ROOT/'statistics.py'),'--reproduce'];command(argv);subprocess.run(argv,stdout=(ROOT/'logs/reproduction.log').open('a'),stderr=subprocess.STDOUT,check=True)
 assert load(ROOT/'fid_reproduction_integrity.json')['status']=='PASS';assert load(ROOT/'kid_implementation_integrity.json')['status']=='PASS'
 jobs={}
 # UVG small pools are evaluated first; separate statistical and codec workers may share a GPU when memory permits.
 for d in ('uvg_validation','uvg_holdout','hevc_b','ulong'):
  for m in METHODS:
   for v in sources(d):jobs[f'eval_{d}_{m}_{v["video_index"]}']=dict(kind='eval',d=d,m=m,i=v['video_index'])
 for d in ('uvg_validation','uvg_holdout','hevc_b','ulong'):
  for q in range(10):jobs[f'stats_{d}_{q}']=dict(kind='stats',d=d,q=q)
 def done(j):return all(point(j['d'],j['m'],j['i'],q).exists()for q in range(10))if j['kind']=='eval'else(ROOT/'statistics'/j['d']/f'qp{j["q"]}.json').exists()
 def ready(j):return j['kind']=='eval'or all(point(j['d'],m,v['video_index'],j['q']).exists()for m in METHODS for v in sources(j['d']))
 class Adopted:
  def __init__(self,pid,job):self.pid=pid;self.job=job;self.returncode=None
  def poll(self):
   path=Path('/proc')/str(self.pid)
   try:
    alive=path.exists() and str(ROOT).encode() in (path/'cmdline').read_bytes() and (path/'stat').read_text().split()[2]!='Z'
   except (FileNotFoundError,ProcessLookupError):alive=False
   if alive:return None
   self.returncode=0 if done(self.job) else 1;return self.returncode
 if (ROOT/'gpu_schedule.json').exists():
  for name,previous in load(ROOT/'gpu_schedule.json').get('active',{}).items():
   if name not in jobs:continue
   proc=Adopted(previous['PID'],jobs[name])
   if proc.poll() is None:
    active[name]=dict(process=proc,gpu=previous['gpu'],log=previous['log'],log_handle=open(previous['log'],'a'));attempts[name]=1;events.append(dict(event='adopt',task=name,PID=proc.pid,gpu=previous['gpu'],unix=time.time()))
 while True:
  for name,a in list(active.items()):
   if a['process'].poll()is not None:
    code=a['process'].returncode;a['log_handle'].close();events.append(dict(event='finish',task=name,gpu=a['gpu'],returncode=code,unix=time.time()));del active[name]
    if code!=0:
     txt=Path(a['log']).read_text(errors='replace')[-20000:];oom='out of memory'in txt.lower();events.append(dict(event='retry',task=name,OOM=oom,error=txt[-5000:],unix=time.time()))
     if not oom or attempts[name]>=4:raise RuntimeError(f'{name} failed; see {a["log"]}: {txt[-3000:]}')
     peaks[jobs[name]['kind']]=max(peaks[jobs[name]['kind']]*1.3,16000)
  todo=[(n,j)for n,j in jobs.items()if not done(j)]
  if not todo and not active:break
  states=gpu_state()
  # Two codec and one statistics worker maximum per physical device; both are allowed with unrelated processes.
  for name,j in todo:
   if name in active or not ready(j):continue
   free=[]
   for s in states:
    own=[jobs[k]['kind']for k,a in active.items()if a['gpu']==s['gpu']]
    if own.count(j['kind']) < (2 if j['kind']=='eval' else 1) and s['free_MiB']>=peaks[j['kind']]:free.append(s)
   if not free:continue
   g=max(free,key=lambda x:x['free_MiB']);gpu=g['gpu'];g['free_MiB']-=peaks[j['kind']]
   if j['kind']=='eval':argv=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--gpu',str(gpu),'--dataset',j['d'],'--method',j['m'],'--video',str(j['i'])]
   else:argv=[PYTHON,'-B','-u',str(ROOT/'statistics.py'),'--gpu',str(gpu),'--dataset',j['d'],'--qp',str(j['q'])]
   attempts[name]=attempts.get(name,0)+1;log=ROOT/'logs'/f'{name}_attempt{attempts[name]}.log';f=log.open('a');command(argv);proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'));active[name]=dict(process=proc,gpu=gpu,log=str(log),log_handle=f);events.append(dict(event='start',task=name,PID=proc.pid,gpu=gpu,available_memory=g['free_MiB']+peaks[j['kind']],command=argv,unix=time.time()));print('START',name,gpu,flush=True)
  view={n:dict(PID=a['process'].pid,gpu=a['gpu'],log=a['log'])for n,a in active.items()};count=completed();stats_count=sum((ROOT/'statistics'/d/f'qp{q}.json').exists()for d in DATASETS for q in range(10));dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase='evaluation_and_statistics',PID=os.getpid(),completed_points=count,expected_points=480,completed_statistics=stats_count,expected_statistics=40,active=view,unix=time.time()));dump(ROOT/'gpu_schedule.json',dict(initial=initial,current=states,events=events,active=view,OOM_retries=sum(e.get('OOM',False)for e in events),gpu_ids=[4,5,6,7]));time.sleep(10)
 dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase='report',completed_points=480,completed_statistics=40,PID=os.getpid()))
 for script in ('report.py','publish_github.py'):
  argv=[PYTHON,'-B','-u',str(ROOT/script)];command(argv);subprocess.run(argv,stdout=(ROOT/'logs'/f'{script}.log').open('a'),stderr=subprocess.STDOUT,check=True)
 dump(ROOT/'pipeline_status.json',dict(status='PASS',phase='complete',completed_points=480,completed_statistics=40,PID=os.getpid()));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  previous=load(ROOT/'pipeline_status.json')if(ROOT/'pipeline_status.json').exists()else{};previous.update(status='FAIL',error=traceback.format_exc(),completed_points=completed());dump(ROOT/'pipeline_status.json',previous);raise
