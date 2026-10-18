import json,os,subprocess,sys,time,signal
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from evaluate35 import valid,task_id
P=Path('/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python')
GPUS=(4,5,6,7); RESERVE=8500; MAX_PER_GPU=2; MAX_RETRIES=2
manifest=json.load(open(R/'task_manifest.json'));pending=list(manifest['tasks']);logroot=R/'logs';logroot.mkdir(exist_ok=True)
statusp=R/'pipeline_status.json';sched=R/'gpu_schedule.jsonl'
def gpu_free():
 try:
  s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  return {int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in s.splitlines())}
 except Exception:return {}
def save_status(active,done,fail):
 x={'status':'RUNNING' if pending or active else ('FAILED' if fail else 'EVALUATIONS_COMPLETE'),'expected_new':len(manifest['tasks']),'completed_new':done,'pending':len(pending),'active':[{'task':task_id(t),'gpu':g,'pid':p.pid} for t,g,p,_ in active],'failures':fail,'updated_unix':time.time()}
 tmp=statusp.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(statusp)
def isdone(t):
 try:return valid(t)
 except Exception:return False
# Drop any already completed rows; idempotent on resume.
pending=[t for t in pending if not isdone(t)];done=42-len(pending);failed=[];active=[];attempts={}
with sched.open('a') as sf:
 while pending or active:
  free=gpu_free();counts={g:sum(1 for _,gg,_,_ in active if gg==g) for g in GPUS}
  # Reserve immediately for every launched process, before CUDA allocations become visible.
  for g in GPUS:
   while pending and counts[g]<MAX_PER_GPU:
    active_res=counts[g]*RESERVE
    if free.get(g,0)-active_res<RESERVE:break
    t=pending.pop(0);tid=task_id(t);attempt=attempts.get(tid,0)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    lf=(logroot/f'{tid}.log').open('ab')
    cmd=[str(P),'-u',str(R/'evaluate35.py'),'--task',json.dumps(t,separators=(',',':')),'--gpu',str(g)]
    p=subprocess.Popen(cmd,cwd=R.parents[1],stdout=lf,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    active.append((t,g,p,lf));counts[g]+=1
    sf.write(json.dumps({'task_id':tid,'task':t,'physical_gpu':g,'pid':p.pid,'started_unix':time.time(),'free_before_mib':free.get(g),'reserved_mib':RESERVE,'attempt':attempt+1})+'\n');sf.flush();os.fsync(sf.fileno())
  new_active=[]
  for t,g,p,lf in active:
   code=p.poll()
   if code is None:new_active.append((t,g,p,lf));continue
   lf.close();tid=task_id(t)
   ok=code==0 and isdone(t)
   sf.write(json.dumps({'task_id':tid,'physical_gpu':g,'pid':p.pid,'ended_unix':time.time(),'exit_code':code,'status':'PASS' if ok else 'FAIL','attempt':attempts.get(tid,0)+1})+'\n');sf.flush();os.fsync(sf.fileno())
   if ok:done+=1
   else:
    attempts[tid]=attempts.get(tid,0)+1
    if attempts[tid]<=MAX_RETRIES:pending.append(t)
    else:failed.append({'task':t,'exit_code':code,'log':str(logroot/f'{tid}.log')})
  active=new_active;save_status(active,done,failed)
  if pending or active:time.sleep(5)
 save_status([],done,failed)
 print(json.dumps({'completed':done,'expected':42,'failed':failed}))
