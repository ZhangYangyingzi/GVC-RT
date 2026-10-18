import json,os,subprocess,time,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
BRANCHES={'N_noP_BG_FD':6,'P_PBG_FD':7}
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def gpu():
 out=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(a):dict(total=int(b),used=int(c),free=int(d),util=int(e)) for a,b,c,d,e in (x.strip().split(',') for x in out.splitlines())}
def proc_alive(p):
 try:os.kill(p,0);return True
 except OSError:return False
def main():
 lock=ROOT/'driver.lock';fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o664)
 import fcntl
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('v6.34b driver already active')
 stp=ROOT/'pipeline_status.json';state=read(stp) if stp.exists() else {}
 state.update(computation='RUNNING',training='RUNNING',evaluation='PENDING',aggregation='PENDING',publication='PENDING',target_step=500,updated_unix=time.time());dump(stp,state)
 workers={};launched=set()
 # Recover only exact branch workers attached to this driver and its output directory.
 for b,g in BRANCHES.items():
  q=ROOT/'branches'/b/'training_status.json'
  s=read(q) if q.exists() else {}
  if s.get('status')=='PASS' and s.get('committed_updates')==500:continue
  ck=ROOT/'branches'/b/'checkpoints/latest.pt';cur=int(__import__('torch').load(ck,map_location='cpu',weights_only=True)['adaptation_step'])
  if cur>=500:continue
  log=(ROOT/'logs'/f'train_{b}.log').open('ab')
  env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  p=subprocess.Popen([PYTHON,'-u',str(ROOT/'train.py'),'--branch',b,'--gpu',str(g)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  workers[b]=[p,log,g];launched.add(b)
 dump(ROOT/'audits/gpu_admission.json',{'status':'PASS','initial_gpu_state':gpu(),'training_reservations_MiB':{'N_noP_BG_FD':14000,'P_PBG_FD':25000},'evaluation_reservation_MiB':8000,'training_gpu_assignment':BRANCHES,'evaluation_gpu_pool':[6,7],'source_measured_training_peaks_MiB':{'N':11556,'P':21034},'evaluation_measured_peak_MiB':5038,'external_processes_terminated':False})
 pending_eval=[];eval_running={};reported={}
 alltasks=read(ROOT/'task_manifest.json')['tasks']
 # The endpoint tasks must wait for their own branch's durable step-500 export.
 def evalvalid(t):
  p=ROOT/'tasks'/f"adapt_{t['dataset']}_{t['video']}_{t['method']}_{t['qp']}.json"
  if not p.exists():return False
  try:
   r=read(p);return r.get('status')=='PASS' and r.get('result',{}).get('method')==t['method'] and r.get('result',{}).get('adaptation_step')==500
  except Exception:return False
 while workers or eval_running or pending_eval:
  for b,(p,log,g) in list(workers.items()):
   rc=p.poll()
   if rc is not None:
    log.close();del workers[b]
    status=read(ROOT/'branches'/b/'training_status.json')
    if rc!=0 or status.get('committed_updates')!=500:
     state.update(training='FAILED',failed_branch=b,exit_code=rc,committed=status.get('committed_updates'),updated_unix=time.time());dump(stp,state)
     raise RuntimeError(f'{b} failed at {status.get("committed_updates")}; inspect logs/train_{b}.log')
    method=b+'_500'
    branch_tasks=[t for t in alltasks if t.get('method')==method]
    pending_eval.extend(t for t in branch_tasks if not evalvalid(t))
    reported[b]=500
  mem=gpu();active_gpu={v[2] for v in eval_running.values()}
  for g in (6,7):
   if pending_eval and g not in active_gpu and mem[g]['free']>=8000:
    t=pending_eval.pop(0);key=f"{t['method']}_{t['video']}_{t['qp']}";log=(ROOT/'logs'/f'eval_{key}.log').open('ab')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([PYTHON,'-u',str(ROOT/'evaluate32.py'),'--task',json.dumps(t,separators=(',',':')),'--gpu',str(g)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    eval_running[key]=[p,log,g,t]
    active_gpu.add(g)
  for key,(p,log,g,t) in list(eval_running.items()):
   rc=p.poll()
   if rc is not None:
    log.close();del eval_running[key]
    if rc!=0 or not evalvalid(t):pending_eval.insert(0,t)
  state.update(training='PASS' if len(reported)==2 else 'RUNNING',branches_complete=reported,evaluation='RUNNING' if eval_running or pending_eval else ('PASS' if len(reported)==2 else 'PENDING'),new_eval_completed=sum(evalvalid(t) for t in alltasks if t['step']==500),new_eval_expected=36,active_training=len(workers),active_evaluation=len(eval_running),pending_evaluation=len(pending_eval),updated_unix=time.time(),gpu_state=gpu())
  dump(stp,state)
  time.sleep(5)
 # Aggregate is a separate CPU stage.
 state.update(training='PASS',evaluation='PASS',aggregation='RUNNING',updated_unix=time.time());dump(stp,state)
 subprocess.check_call([PYTHON,str(ROOT/'aggregate34b.py')],cwd=ROOT)
 state=read(stp);state.update(computation='PASS',aggregation='PASS',updated_unix=time.time());dump(stp,state)
if __name__=='__main__':main()
