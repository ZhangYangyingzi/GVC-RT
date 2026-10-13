"""Single-instance resource-admitted fixed pipeline; no external AI calls."""
from audit31 import *
import argparse,fcntl,subprocess,threading,traceback
def state(**kw):
 try:s=load(ROOT/'status.json')
 except OSError:s={}
 s.update(kw,PID=os.getpid(),heartbeat=time.time());dump(ROOT/'status.json',s)
def resources():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(r.split(',')[0]):float(r.split(',')[1]) for r in rows.splitlines() if int(r.split(',')[0]) in (4,5,6,7)}
 mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
 return free,mem['MemAvailable']/1024
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 lf=(ROOT/'driver.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state(status='RUNNING',phase='canaries',calculation_status='RUNNING',report_status='PENDING',integrity_status='PENDING',publication_status='PENDING')
 tasks=tasklist();complete={uid(t) for t in tasks if valid(t)};failed={};active={};attempt={};profiles={};last_summary=0.;maxactive=16
 for t in tasks:
  if uid(t) in complete:
   r=load(receipt(t));st=t['stage']
   if not r.get('reused'):profiles[st]=dict(gpu=max(profiles.get(st,{}).get('gpu',0),r.get('peak_memory_MiB',0)),rss=max(profiles.get(st,{}).get('rss',0),r.get('peak_RSS_MiB',0)))
 firstcolor=[dict(stage='color',dataset='hevc_b',video=0,method=m,qp=0) for m in METHODS]
 firstprobe=next(t for t in tasks if t['stage']=='probe')
 firstreuse=[dict(stage='intervention',dataset='hevc_b',video=0,method=c,qp=0) for c in ('R00','R11')]
 def ready(t):
  st=t['stage'];u=uid(t)
  if st=='color':return u in [uid(x) for x in firstcolor] or all(uid(x) in complete for x in firstcolor)
  if st=='probe':return u==uid(firstprobe) or uid(firstprobe) in complete
  if st=='intervention':
   if t['method'] in ('R00','R11'):
    ct=dict(t,stage='color',method='original' if t['method']=='R00' else 'B1_initial');return uid(ct) in complete
   return all(uid(x) in complete for x in firstreuse)
 while len(complete)+len(failed)<len(tasks):
  for u,(p,t,g,log) in list(active.items()):
   ec=p.poll()
   if ec is None:continue
   log.close();del active[u]
   if ec==0 and valid(t):
    complete.add(u);r=load(receipt(t))
    if not r.get('reused'):
     old=profiles.get(t['stage'],{})
     profiles[t['stage']]=dict(gpu=max(old.get('gpu',0),r['peak_memory_MiB']),rss=max(old.get('rss',0),r['peak_RSS_MiB']))
    print('EVENT DONE',u,flush=True)
   else:
    text=(ROOT/'logs'/f'{u}_attempt{attempt[u]}.log').read_text()[-16000:]
    oom='out of memory' in text.lower();retry=oom and attempt[u]<2 and not (ROOT/'raw/probes'/(u+'.landed.json')).exists()
    if retry:maxactive=max(1,len(active));print('EVENT OOM_REQUEUED',u,flush=True)
    else:failed[u]=dict(task=t,exit_code=ec,reason=text);print('EVENT FAILED',u,flush=True)
    error(dict(UID=u,exit_code=ec,OOM=oom,retry=retry,traceback=text))
  gf,host=resources()
  pending=[t for t in tasks if uid(t) not in complete and uid(t) not in active and uid(t) not in failed]
  runnable=[t for t in pending if ready(t)]
  # Alternate stages; do not let hundreds of color tasks starve independent probes/cells.
  runnable.sort(key=lambda t: (sum(x[1]['stage']==t['stage'] for x in active.values()),tasks.index(t)))
  for t in runnable:
   if len(active)>=maxactive:break
   st=t['stage'];profile=profiles.get(st)
   if profile is None and any(x[1]['stage']==st for x in active.values()):continue
   # Before the first measurement, use one worker and conservative headroom.
   need=(profile or dict(gpu=20000,rss=12000));gp=1.25*need['gpu'];hp=1.25*need['rss']
   candidates=[]
   for g,free in gf.items():
    reserve=sum(.25*profiles.get(x[1]['stage'],dict(gpu=20000))['gpu'] for x in active.values() if x[2]==g)
    if free>=gp+2048+reserve:candidates.append((free-reserve,g))
   hostreserve=sum(.25*profiles.get(x[1]['stage'],dict(rss=12000))['rss'] for x in active.values())
   if not candidates or host<hp+hostreserve+8192:continue
   _,g=max(candidates);u=uid(t);attempt[u]=attempt.get(u,0)+1
   log=(ROOT/'logs'/f'{u}_attempt{attempt[u]}.log').open('a')
   worker={'color':'color_worker.py','intervention':'intervention_worker.py','probe':'probe_worker.py'}[st]
   env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
   argv=[PYTHON,'-B','-u',str(ROOT/worker),'--gpu',str(g),'--task',json.dumps(t)]
   p=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);active[u]=(p,t,g,log);gf[g]-=gp;host-=hp
   with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=argv,PID=p.pid,GPU=g,time=time.time()))+'\n')
   print('EVENT START',u,'GPU',g,'PID',p.pid,flush=True)
  counts={st:dict(planned=sum(t['stage']==st for t in tasks),completed=sum(t['stage']==st and uid(t) in complete for t in tasks),failed=sum(t['stage']==st and uid(t) in failed for t in tasks),remaining=sum(t['stage']==st and uid(t) not in complete and uid(t) not in failed for t in tasks)) for st in ('color','intervention','probe')}
  state(phase='fixed_queue',counts=counts,workers=[dict(UID=u,PID=x[0].pid,GPU=x[2]) for u,x in active.items()],resource_profiles=profiles,failed=failed)
  if time.time()-last_summary>=3600:
   summary=time.strftime('%Y-%m-%d %H:%M:%S')+' '+json.dumps(counts)+'\n';textfile(ROOT/'hourly_summary.txt',summary);print('HOURLY',summary.strip(),flush=True);last_summary=time.time()
  if not active and pending and not runnable:
   for t in pending:failed[uid(t)]=dict(task=t,reason='prerequisite canary/old decode failed')
   break
  if len(complete)+len(failed)>=len(tasks):break
  # Resource admission and status at 60s; child exits are handled immediately.
  deadline=time.time()+60
  while time.time()<deadline:
   if any(x[0].poll() is not None for x in active.values()):break
   time.sleep(1)
 state(phase='aggregate',calculation_status='COMPLETE' if not failed else 'PARTIAL',failed=failed,workers=[])
 def run(script):
  with (ROOT/'logs'/f'{script}.log').open('a') as f:subprocess.run([PYTHON,'-B','-u',str(ROOT/script)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
 run('report31.py');state(report_status='COMPLETE',integrity_status=load(ROOT/'final_integrity.json')['status'])
 state(phase='publication',publication_status='RUNNING');run('publish31.py');state(publication_status=load(ROOT/'publication_status.json')['status'])
 final=load(ROOT/'final_integrity.json');pub=load(ROOT/'publication_status.json')
 if final['status']=='PASS' and pub['status']=='PASS':
  dump(ROOT/'DONE.json',dict(status='PASS',counts=final['counts'],commit=pub['commit'],finished=time.time()));state(status='DONE',phase='finished',publication_status='PASS')
 else:
  dump(ROOT/'FAILED.json',dict(status='FAIL',integrity=final,publication=pub,failed=failed));state(status='FAILED',phase='finished')
 print('TERMINAL',load(ROOT/'status.json')['status'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  error(dict(driver=os.getpid(),traceback=traceback.format_exc()));dump(ROOT/'FAILED.json',dict(status='FAIL',error=traceback.format_exc()));state(status='FAILED',phase='exception');raise
