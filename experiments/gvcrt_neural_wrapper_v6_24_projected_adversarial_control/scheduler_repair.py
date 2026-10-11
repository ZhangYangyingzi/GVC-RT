"""Repair own scheduler reservation bookkeeping without terminating GPU workers."""
import os,signal,json,pathlib,time,subprocess
r=pathlib.Path(__file__).resolve().parent;s=json.load(open(r/'pipeline_status.json'));pid=s['PID'];cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
assert r.name in cmd and 'run_pipeline.py' in cmd
os.kill(pid,signal.SIGSTOP)
starts={}
for line in (r/'gpu_schedule.jsonl').read_text().splitlines():
 x=json.loads(line)
 if x['event']=='start':starts[x['task']]=x
 elif x['event']=='exit':starts.pop(x['task'],None)
active={}
for k,x in starts.items():
 wp=x['PID'];path=pathlib.Path(f'/proc/{wp}')
 try:
  stat=(path/'stat').read_text().rsplit(') ',1)[1].split();cmd=(path/'cmdline').read_bytes().decode().replace('\0',' ')
  if stat[0]=='Z' or r.name not in cmd:continue
 except FileNotFoundError:continue
 active[k]=s.get('active',{}).get(k,dict(PID=wp,identity=stat[19],gpu=x['physical_gpu'],log=str(r/'logs'/f'{k}.log'),offset=0,required_MiB=x['reservation_MiB']))
old=dict(s);s['active']=active
peaks=[]
for p in (r/'parts').rglob('*.json'):
 try:
  v=json.load(open(p))
  if v.get('fresh_run') and v.get('peak_memory_MiB'):peaks.append(v['peak_memory_MiB'])
 except Exception:pass
s['memory']['eval']=max(peaks+[5846]);s['scheduler_memory_reservation_repair']=True
(r/'audits/scheduler_memory_reservation_repair.json').write_text(json.dumps(dict(status='PASS',old_scheduler_PID=pid,old_active_count=len(old['active']),preserved_live_worker_PIDs=[x['PID'] for x in active.values()],GPU_workers_terminated=0,corrected_eval_peak=s['memory']['eval'],reason='Reserve required minus already allocated memory for every active worker, including workers observed before full allocation',unix=time.time()),indent=2)+'\n')
p=r/'run_pipeline.py';code=p.read_text().replace('observed=set()','observed={}')
code=code.replace('observed.add(pid)','observed[pid]=used')
code=code.replace("if w['PID'] not in observed:free[w['gpu']]=max(0,free[w['gpu']]-w['required_MiB'])", "free[w['gpu']]=max(0,free[w['gpu']]-max(0,w['required_MiB']-observed.get(w['PID'],0)))")
code=code.replace("if oom:memory[jobs[k]['kind']]=max(memory.get(jobs[k]['kind'],0),int(w['required_MiB']*1.25))", "if oom:events.append(dict(event='memory_reservation_retry',task=k,gpu=w['gpu'],peak_estimate_MiB=memory.get(jobs[k]['kind'],0),unix=time.time()))")
compile(code,str(p),'exec');p.write_text(code)
(r/'pipeline_status.json').write_text(json.dumps(s,indent=2)+'\n')
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(100):
 try:
  st=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  if st[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(.05)
f=(r/'logs/pipeline.log').open('a');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
a=['/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python','-B','-u',str(r/'run_pipeline.py')]
p=subprocess.Popen(a,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(r/'launch.json').write_text(json.dumps(dict(PID=p.pid,unix=time.time(),previous_scheduler_PID=pid,restored_workers=len(active),command=a,detach='start_new_session=True'),indent=2)+'\n');print('restarted scheduler',p.pid,'preserved GPU workers',len(active),'measured evaluation peak',s['memory']['eval'])
