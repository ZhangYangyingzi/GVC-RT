"""Persistent locked queue; immediate peak reservations, no external process termination."""
from io21 import *
import fcntl,argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');a=ap.parse_args()
 if a.status:print(json.dumps(load(ROOT/'pipeline_status.json'),indent=2));return
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=load(ROOT/'pipeline_status.json');active=state.get('active',{});done=set(state.get('done',[]));attempts=state.get('attempts',{})
 def ticks(pid):
  try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
  except OSError:return None
 def persist():
  state.update(status='RUNNING',PID=os.getpid(),active=active,done=sorted(done),attempts=attempts,updated_unix=time.time(),completed_RD=sum(1 for p in (ROOT/'parts/ip_-1').glob('*/*/*.json') if p.parent.name in METHODS),expected_RD=1840)
  tp=ROOT/'branches/B_covstats/training_status.json'
  state['training_steps']={'A_continue':1000,'B_covstats':load(tp).get('adaptation_step',0) if tp.exists() else 0}
  state['FID_feature_completed']=len(list((ROOT/'audits/features').glob('*/*/qp*.json')))
  state['lambda_stat']=load(ROOT/'calibration/final.json')['lambda_stat'] if (ROOT/'calibration/final.json').exists() else None
  dump(ROOT/'pipeline_status.json',state)
 def log(v):
  with (ROOT/'gpu_schedule.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),**v))+'\n')
 def launch(name,argv,gpu,mem):
  out=ROOT/'logs'/f'{name}_attempt{attempts.get(name,0)+1}.log';env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
  p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'task_wrapper.py'),name,*argv],stdout=out.open('ab'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
  attempts[name]=attempts.get(name,0)+1;active[name]=dict(PID=p.pid,start_ticks=ticks(p.pid),gpu=gpu,reserve=mem,argv=argv,log=str(out));log(dict(event='start',task=name,**active[name]));persist()
 # Adopt externally launched checked tasks; start identity is recorded before any replacement.
 external=[('calibration','calibrate.py',4,16000),('precheck','precheck.py',5,16000),('feature_ulong','audit_features.py --gpu 6 --dataset ulong',6,21000),('feature_hevc','audit_features.py --gpu 7 --dataset hevc_b',7,21000)]
 for name,text,gpu,mem in external:
  if name in active or name in done:continue
  lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
  for line in lines:
   if str(ROOT)+'/' in line and text.split()[0] in line and 'task_wrapper' not in line and ('--dataset '+name.split('_')[-1] in line if name.startswith('feature_') else True):
    pid=int(line.split()[0]);active[name]=dict(PID=pid,start_ticks=ticks(pid),gpu=gpu,reserve=mem,argv=[],external=True);break
 tasks={
 'calibration':(['calibrate.py','--gpu','GPU'],16000),
 'precheck':(['precheck.py','--gpu','GPU'],16000),
 'feature_ulong':(['audit_features.py','--gpu','GPU','--dataset','ulong'],21000),
 'feature_hevc':(['audit_features.py','--gpu','GPU','--dataset','hevc_b'],21000),
 'feature_mcljcv':(['audit_features.py','--gpu','GPU','--dataset','mcljcv'],21000),
 'codec_step0':(['evaluate.py','--gpu','GPU','--dataset','uvg_holdout','--method','B_covstats_0','--video','2','--qps','0'],23000),
 'dists_probe':(['dists_bg_probe.py','--gpu','GPU'],5000),
 'train_B':(['train.py','--gpu','GPU','--branch','B_covstats'],18000)}
 def complete(n):
  if n=='calibration':p=ROOT/'calibration/final.json'
  elif n=='precheck':p=ROOT/'audits/B_covstats_pretraining.json'
  elif n.startswith('feature_'):
   d={'feature_hevc':'hevc_b'}.get(n,n[8:]);files=list((ROOT/'audits/features'/d).glob('*/qp*.json'));return len(files)==6 and all(load(x)['status']=='PASS' for x in files)
  elif n=='codec_step0':
   p=point('uvg_holdout','B_covstats_0',2,0);return p.exists() and load(p).get('same_GPU_step0_verification',{}).get('status')=='PASS'
  elif n=='dists_probe':p=ROOT/'audits/DISTS_deep_BG_probe.json'
  elif n=='train_B':p=ROOT/'branches/B_covstats/training_integrity.json'
  elif n.startswith('eval|'):
   _,d,i,q=n.split('|');return point(d,'B_covstats_1000',int(i),int(q)).exists()
  else:return False
  return p.exists() and load(p).get('status')=='PASS'
 while True:
  for n in list(active):
   t=active[n]
   if ticks(t['PID'])==t['start_ticks'] and Path(f'/proc/{t["PID"]}/stat').read_text().split()[2]!='Z':continue
   exitfile=ROOT/'task_exits'/f'{n}.json';rc=load(exitfile)['exit_code'] if exitfile.exists() else None
   log(dict(event='exit',task=n,exit_code=rc,**t));del active[n]
   if complete(n):done.add(n)
   else:
    error=Path(t['log']).read_text()[-12000:] if t.get('log') and Path(t['log']).exists() else 'external worker exited without complete integrity record'
    oom='out of memory' in error.lower()
    if oom:
     category='evaluation' if n.startswith('eval|') else n
     old=state.setdefault('peak_estimates',{}).get(category,t['reserve']);state['peak_estimates'][category]=int(max(old,t['reserve'])*1.35+1024)
     log(dict(event='OOM_retry',task=n,category=category,new_reserved_MiB=state['peak_estimates'][category],error=error))
    if attempts.get(n,0)>=3 and not oom:state.setdefault('blocked',{})[n]=dict(error=error,log=t.get('log'))
  feature_points=list((ROOT/'audits/features').glob('*/*/qp*.json'))
  if feature_points:
   measured=max(load(p)['peak_memory_MiB'] for p in feature_points);reserved=int(measured*1.25+1536)
   for nn in list(active):
    if nn.startswith('feature_'):active[nn]['reserve']=reserved
   for nn,(aa,mm) in list(tasks.items()):
    if nn.startswith('feature_'):tasks[nn]=(aa,reserved)
  for n in tasks:
   if complete(n):done.add(n)
  if 'codec_step0' in done:
   for b in BRANCHES:dump(ROOT/'audits'/f'{b}_codec_equivalence.json',dict(status='PASS',source=EXPECTED,point=str(point('uvg_holdout','B_covstats_0',2,0)),same_initial_modules=True))
  if 'train_B' in done:
   idx=load(ROOT/'branches/B_covstats/checkpoint_index.json')['1000'];ec=evalcfg()
   if 'B_covstats_1000' not in ec['checkpoints']:
    import torch
    cp=idx['inference'];ec['checkpoints']['B_covstats_1000']=cp
    # Deployment precision follows historical apply_joint: bridge/generator are FP16 in evaluation.
    ss=torch.load(cp['path'],map_location='cpu',weights_only=True);ec['deployment_receiver_hashes']['B_covstats_1000']={k:tensor_hash({n:t.half() if t.is_floating_point() else t for n,t in ss[k].items()}) for k in ('bridge','generator')};dump(ROOT/'evaluation/config.json',ec)
   for d in DATASETS:
    for v in sources(d):
     for q in range(10):
      n=f'eval|{d}|{v["video_index"]}|{q}'
      if point(d,'B_covstats_1000',v['video_index'],q).exists():done.add(n)
      else:tasks[n]=(['evaluate.py','--gpu','GPU','--dataset',d,'--method','B_covstats_1000','--video',str(v['video_index']),'--qps',str(q)],23000)
  # Full measured peaks update task admission; no fixed workers-per-card limit.
  for nn,(aa,mm) in list(tasks.items()):
   peak=None
   if nn=='train_B' and (ROOT/'calibration/final.json').exists():peak=load(ROOT/'calibration/final.json')['peak_memory_MiB']+1024
   elif nn.startswith('eval|') and 'codec_step0' in done:peak=load(point('uvg_holdout','B_covstats_0',2,0))['peak_memory_MiB']
   measured_reserve=int(peak*1.25+512) if peak is not None else mm
   category='evaluation' if nn.startswith('eval|') else nn
   tasks[nn]=(aa,max(measured_reserve,state.get('peak_estimates',{}).get(category,0)))
  gpurows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpurows.splitlines() if int(x.split(',')[0]) in (4,5,6,7)}
  # Conservative reserve includes loading workers, even before any cuda allocation.
  # Discount actual allocated memory already reflected by nvidia-smi; retain all pending reservations.
  allocated={}
  try:
   for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
    pid,mb=line.split(',');allocated[int(pid)]=int(mb)
  except (ValueError,subprocess.CalledProcessError):pass
  parents={}
  for proc in Path('/proc').iterdir():
   if proc.name.isdigit():
    try:parents[int(proc.name)]=int((proc/'stat').read_text().split()[3])
    except (OSError,ValueError,IndexError):pass
  def owned_mb(pid):
   total=0
   for pp,mb in allocated.items():
    pp0=pp;seen=set()
    while pp0 in parents and pp0 not in seen:
     if pp0==pid:total+=mb;break
     seen.add(pp0);pp0=parents[pp0]
   return total
  for n,(argv,mem) in list(tasks.items()):
   if n in done or n in active or n in state.get('blocked',{}):continue
   available_host_kib=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
   if available_host_kib<8192*1024 or len(active)>=max(1,(os.cpu_count() or 2)//2):break
   if n=='train_B' and not {'calibration','precheck','codec_step0','dists_probe'}<=done:continue
   if n=='codec_step0' and 'precheck' not in done:continue
   available={g:f-sum(max(0,x['reserve']-owned_mb(x['PID'])) for x in active.values() if x['gpu']==g)-1500 for g,f in free.items()}
   g=max(available,key=available.get)
   if available[g]>=mem:launch(n,[str(ROOT/x) if j==0 else str(g) if x=='GPU' else x for j,x in enumerate(argv)],g,mem)
  persist()
  if state['completed_RD']>=1840 and not active and {'calibration','precheck','codec_step0','feature_ulong','feature_hevc','feature_mcljcv','train_B'}<=done:
   subprocess.run([PYTHON,'-B',str(ROOT/'finalize.py')],check=True);return
  time.sleep(15)
if __name__=='__main__':main()
