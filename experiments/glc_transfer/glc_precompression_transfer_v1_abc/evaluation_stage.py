"""Receipt-validated, memory-admitted real evaluations; no numerical changes."""
from preflight import *
from data import load,Frames
import math

METHODS=('A_original','B_rgb_P','C_latent_P')
MAX_WORKERS=8  # GPU4–7, at most two memory-admitted workers per physical GPU.
def admission_budget(torch_peak,device_peak=0):
 return max(13000,torch_peak*1.3+1024,device_peak*1.15+512)
def volatility_margin(previous,shortfall):
 return min(8192,previous+max(2048,shortfall+512))
def expected_signatures():
 common=dict(manifest=sha(R/'eval_manifest.json'),protocol=sha(R/'protocol.json'),runtime=sha(R/'runtime.py'),evaluation=sha(R/'evaluate.py'),models=load(R/'audits/checkpoint_audit.json')['models'])
 out={'A_original':common}
 for m in METHODS[1:]:out[m]=dict(common,P_sha256=sha(R/'final_P'/f'{m}_1000.pt'))
 return out
def task_name(t):return f"{t['dataset']}_{t['video']}_{t['branch']}_{t['q']}"
def receipt_valid(t,signatures):
 path=R/'results/points'/f'{task_name(t)}.json'
 if not path.exists():return False
 x=load(path)
 r=x['result'];v=load(R/'eval_manifest.json')[t['dataset']][t['video']]
 if (r['dataset'],r['video'],r['method'],r['q'])!=(t['dataset'],t['video'],t['branch'],t['q']):return False
 if r['name']!=v['name'] or r['rate_accounting_fps']!=v['rate_accounting_fps'] or r['rate_type']!='real_local_nonofficial_rans':return False
 if r['bpp']!=r['bytes']*8/(64*v['height']*v['width']):return False
 if not all(math.isfinite(r[k]) for k in ('bpp','LPIPS','DISTS','FloLPIPS','LPIPS_P_only','DISTS_P_only')):return False
 if any(Path(p).stat().st_size!=r['bytes'] for p in x['artifacts'] if p.endswith('.glc') and Path(p).exists()):return False
 return x.get('signature')==signatures[t['branch']] and x.get('status')=='PASS' and x.get('independent_decode') and x.get('frozen_unchanged') and r['frames']==64 and r['transitions']==63 and all(Path(p).exists() and sha(p)==h for p,h in x['artifacts'].items())
def memory_available(gpus,allocations,running):
 # Reserve the not-yet-visible native metric peak of every admitted worker.
 free={g:f-max(2048,t*.1) for g,t,f in gpus if g in (4,5,6,7)}
 for w in running:free[w['gpu']]-=max(0,w['need']-allocations.get(w['process'].pid,0))
 return free
def snapshot_memory(running):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
 gpus=[(int(a),float(b),float(c)) for a,b,c in (line.split(',') for line in raw.splitlines())]
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 allocations={int(a):float(b) for a,b in (line.split(',') for line in raw.splitlines()) if a.strip().isdigit() and b.strip().isdigit()}
 # Include CUDA context / non-PyTorch allocations, not just allocator reserves.
 for w in running:
  w['device_peak']=max(w.get('device_peak',0),allocations.get(w['process'].pid,0))
  w['need']=max(w['need'],w['device_peak']*1.15+512)
 return memory_available(gpus,allocations,running),{g:f-max(2048,t*.1) for g,t,f in gpus if g in (4,5,6,7)}
def run(command,state,PY):
 assert not load(R/'early_original_status.json').get('running'), 'Early A workers must drain before final evaluation'
 signatures=expected_signatures();manifest=load(R/'eval_manifest.json')
 for x in load(R/'audits/checkpoint_audit.json')['models']:assert sha(x['path'])==x['sha256']
 from concurrent.futures import ThreadPoolExecutor,as_completed
 verified=[];source_jobs=[(d,v) for d,vs in manifest.items() for v in vs]
 def verify_source(job):
   d,v=job
   assert len(v['frame_rgb_sha256'])==64
   f=Frames(v)
   for i in range(64):assert f.array(i).shape==(v['height'],v['width'],3)
   f.close();return dict(dataset=d,name=v['name'],RGB_hashes=v['frame_rgb_sha256'])
 with ThreadPoolExecutor(max_workers=4) as pool:
  for future in as_completed([pool.submit(verify_source,j) for j in source_jobs]):
   verified.append(future.result())
   save('audits/evaluation_source_progress.json',dict(status='RUNNING',verified_videos=len(verified),expected_videos=len(source_jobs),verified_frames=64*len(verified),strict_RGB_hashes=True,CPU_workers=4))
 save('audits/evaluation_source_revalidation.json',dict(status='PASS',videos=len(verified),frames=64*len(verified),manifest_sha256=sha(R/'eval_manifest.json'),sources=verified))
 save('audits/evaluation_source_progress.json',dict(status='PASS',verified_videos=len(verified),expected_videos=len(source_jobs),verified_frames=64*len(verified),strict_RGB_hashes=True,CPU_workers=4))
 tasks=[dict(dataset=d,video=v,branch=m,q=q) for d,vs in manifest.items() for v in range(len(vs)) for m in METHODS for q in load(R/'protocol.json')['q_grid']]
 expected=len(tasks);cached=[t for t in tasks if receipt_valid(t,signatures)];cached_keys={task_name(t) for t in cached};tasks=[t for t in tasks if task_name(t) not in cached_keys]
 peak={m:max([load(p)['result']['peak_MiB'] for p in (R/'results/points').glob('*.json') if load(p)['result']['method']==m] or [0]) for m in METHODS}
 defaults={'A_original':10356.8,'B_rgb_P':12000.,'C_latent_P':11000.}
 need={m:admission_budget(peak[m]) if peak[m] else max(14000,defaults[m]) for m in METHODS}
 device_peak={m:0 for m in METHODS}
 prior=load(R/'evaluation_scheduler.json') if (R/'evaluation_scheduler.json').exists() else {}
 prior=prior if prior.get('expected')==expected and prior.get('completed',0)<=len(cached) else {}
 need={m:max(need[m],prior.get('admission_MiB',{}).get(m,0)) for m in METHODS}
 device_peak={m:prior.get('device_peak_MiB',{}).get(m,0) for m in METHODS}
 running=[];done=len(cached);fresh=0;retry=prior.get('OOM_or_reference_retries',{});resource_preemptions=prior.get('resource_preemptions',{});cooldown={int(g):v for g,v in prior.get('GPU_cooldown_until',{}).items()};excluded_gpu=prior.get('excluded_gpu',{});max_workers=min(MAX_WORKERS,max(1,prior.get('max_workers',4)));deadline=time.time()+172800;resource_deadline=time.time()+3600
 gpu_margin={int(g):v for g,v in prior.get('GPU_volatility_margin_MiB',{}).items()}
 if not gpu_margin:
  for p in (R/'audits').glob('resource_guard_*.json'):
   event=load(p);g=event['GPU'];gpu_margin[g]=volatility_margin(gpu_margin.get(g,0),event['reserve_shortfall_MiB'])
 if not retry:max_workers=max(2,max_workers)
 last_scale_event=max(prior.get('last_scale_event',0),max(cooldown.values(),default=0)-300)
 def journal():
  save('evaluation_scheduler.json',dict(status='RUNNING',completed=done,expected=expected,reused=len(cached),new_completed=fresh,peak_MiB=peak,device_peak_MiB=device_peak,admission_MiB=need,max_workers=max_workers,per_GPU_cap=2,reserve='max(2048 MiB, total*0.10)',resource_preemptions=resource_preemptions,OOM_or_reference_retries=retry,excluded_gpu=excluded_gpu,GPU_cooldown_until=cooldown,GPU_volatility_margin_MiB=gpu_margin,last_scale_event=last_scale_event,running=[dict(task=w['task'],gpu=w['gpu'],PID=w['process'].pid,admission_MiB=w['need']) for w in running],time=time.time()))
  state('evaluation',completed=done,expected=expected,reused=len(cached),new_completed=fresh)
 journal()
 try:
  while tasks or running:
   if time.time()>deadline:raise RuntimeError('Evaluation 48-hour limit reached; completed receipts preserved')
   for w in list(running):
    p=w['process'];name=task_name(w['task'])
    if p.poll() is None:
     if time.time()-w['started']>1800:p.terminate();p.wait(timeout=30);raise RuntimeError(f'{name} exceeded 1800-second point limit')
     continue
    w['log'].close();running.remove(w)
    if p.returncode:
     tail=(R/'logs'/f'eval_{name}.log').read_text(errors='replace')[-5000:]
     reference_failure='AssertionError' in tail and '==hashes[i]' in tail
     if ('out of memory' in tail.lower() or reference_failure) and not retry.get(name):
      retry[name]=1;excluded_gpu[name]=w['gpu'];need[w['task']['branch']]=max(need[w['task']['branch']]*1.2,w['need']*1.2,14000 if reference_failure else 0);max_workers=max(1,max_workers-1);last_scale_event=time.time();tasks.insert(0,w['task'])
      save(f'audits/eval_failure_{name}.json',dict(task=w['task'],GPU=w['gpu'],error=tail,policy='one retry on different GPU; strict reference checks unchanged',time=time.time()));journal();continue
     raise RuntimeError(f'eval_{name} exit {p.returncode}; logs/eval_{name}.log')
    assert receipt_valid(w['task'],signatures),f'Invalid completed receipt {name}'
    result=load(R/'results/points'/f'{name}.json')['result'];m=w['task']['branch'];peak[m]=max(peak[m],result['peak_MiB']);device_peak[m]=max(device_peak[m],w.get('device_peak',0));need[m]=max(admission_budget(peak[m],device_peak[m]),need[m]);done+=1;fresh+=1;journal()
   free,headroom=snapshot_memory(running)
   for w in running:
    m=w['task']['branch'];device_peak[m]=max(device_peak[m],w.get('device_peak',0));need[m]=max(need[m],w['need'])
   for g,remaining in headroom.items():
    owned=[w for w in running if w['gpu']==g]
    if remaining>=0 or not owned:continue
    w=max(owned,key=lambda w:w['started']);p=w['process'];name=task_name(w['task'])
    if p.poll() is None:p.terminate();p.wait(timeout=30)
    w['log'].close();running.remove(w)
    valid=receipt_valid(w['task'],signatures)
    resource_preemptions[name]=resource_preemptions.get(name,0)+1;cooldown[g]=time.time()+300;last_scale_event=time.time();gpu_margin[g]=volatility_margin(gpu_margin.get(g,0),-remaining)
    save(f'audits/resource_guard_{name}_{time.time_ns()}.json',dict(task=w['task'],GPU=g,reserve_shortfall_MiB=-remaining,action='release newest owned GPU worker; other users preserved',valid_completed_receipt_retained=valid,resource_preemptions=resource_preemptions[name],GPU_cooldown_seconds=300,time=time.time()))
    if valid:done+=1;fresh+=1
    elif resource_preemptions[name]<=3:excluded_gpu[name]=g;max_workers=max(1 if retry else 2,max_workers-1);tasks.insert(0,w['task'])
    else:raise RuntimeError(f'{name}: GPU reserve lost after three resource requeues; partial results preserved')
    journal()
   if max_workers<MAX_WORKERS and time.time()-last_scale_event>=600:
    max_workers+=1;last_scale_event=time.time()
    save(f'audits/concurrency_recovery_{time.time_ns()}.json',dict(status='MEMORY_ADMITTED_RECOVERY',max_workers=max_workers,stable_interval_seconds=600,per_GPU_cap=2,measured_peak_and_reserve_admission_unchanged=True,time=time.time()))
    journal()
   free,_=snapshot_memory(running);free={g:f-gpu_margin.get(g,0) for g,f in free.items()};host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
   for g in sorted(free,key=free.get,reverse=True)*2:
    if not tasks or len(running)>=max_workers or sum(w['gpu']==g for w in running)>=2 or host<12000 or time.time()<cooldown.get(g,0):continue
    idx=next((i for i,t in enumerate(tasks) if need[t['branch']]<=free[g] and excluded_gpu.get(task_name(t))!=g),None)
    if idx is None:continue
    t=tasks.pop(idx);name=task_name(t);p,l=command(f'eval_{name}',[PY,'-B','-u',str(R/'evaluate.py'),'--task',json.dumps(t),'--gpu',str(g)],g)
    running.append(dict(process=p,log=l,task=t,gpu=g,need=need[t['branch']],started=time.time()));free[g]-=need[t['branch']];host-=5000;resource_deadline=time.time()+3600;journal()
   if tasks and not running and time.time()>resource_deadline:raise RuntimeError('GPU4–7 evaluation admission timeout after 3600 seconds')
   time.sleep(30)
  save('evaluation_scheduler.json',dict(status='COMPLETE',completed=done,expected=expected,reused=len(cached),new_completed=fresh,peak_MiB=peak,device_peak_MiB=device_peak,admission_MiB=need,OOM_or_reference_retries=retry,resource_preemptions=resource_preemptions,GPU_volatility_margin_MiB=gpu_margin,running=[]))
  return done,expected
 except Exception:
  for w in running:
   if w['process'].poll() is None:
    w['process'].terminate()
    try:w['process'].wait(timeout=30)
    except subprocess.TimeoutExpired:w['process'].kill();w['process'].wait(timeout=30)
   w['log'].close()
  save('evaluation_scheduler.json',dict(status='BLOCKED',completed=done,expected=expected,reused=len(cached),new_completed=fresh,error=traceback.format_exc(),resource_preemptions=resource_preemptions,running=[]))
  raise

def smoke():
 from types import SimpleNamespace
 assert MAX_WORKERS==8 and MAX_WORKERS==len((4,5,6,7))*2
 gpus=[(0,40960,40960),(4,40960,20000),(5,40960,11000),(6,40960,24000),(7,40960,16000)]
 workers=[dict(gpu=4,need=10356.8,process=SimpleNamespace(pid=1)),dict(gpu=6,need=10356.8,process=SimpleNamespace(pid=2))]
 out=memory_available(gpus,{1:2048,2:9376},workers)
 assert set(out)=={4,5,6,7} and abs(out[4]-(20000-4096-8308.8))<1e-6 and abs(out[6]-(24000-4096-980.8))<1e-6
 assert out[5]<10356.8 and out[7]>=10356.8
 assert admission_budget(9460)==13322 and admission_budget(0)==13000
 assert abs(admission_budget(9460,14000)-16612)<1e-6
 assert max(admission_budget(9460),14654.2)==14654.2
 assert volatility_margin(0,252)==2048 and volatility_margin(2048,2354)==4914 and volatility_margin(8000,1000)==8192
 original=subprocess.check_output
 responses=iter(['4,40960,20000\n','1,12000\n'])
 try:
  subprocess.check_output=lambda *a,**k:next(responses)
  native=[dict(gpu=4,need=10000,process=SimpleNamespace(pid=1))]
  actual,headroom=snapshot_memory(native)
  assert native[0]['device_peak']==12000 and abs(native[0]['need']-14312)<1e-6
  assert abs(actual[4]-(20000-4096-(14312-12000)))<1e-6 and headroom[4]==15904
 finally:subprocess.check_output=original
 task=dict(dataset='ulong',video=0,branch='A_original',q=0);s=load(R/'results/points/ulong_0_A_original_0.json')['signature']
 assert receipt_valid(task,{'A_original':s})
 assert not receipt_valid(task,{'A_original':dict(s,protocol='deliberate_mismatch')})
 save('audits/evaluation_scheduler_smoke.json',dict(status='PASS',CPU_only=True,numerical_training_or_codec_changes=False,memory_available=out,process_device_memory_overhead_accounted=True,expanded_admission_budget_checked=True,verified_original_canary=True,mismatched_protocol_rejected=True,formal_points_added=0,source_sha256=sha(__file__)))
 print('PASS: full measured peak reserved during worker startup; GPUs0–3 excluded')
if __name__=='__main__':smoke()
