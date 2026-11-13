"""One trainer, one bounded cache producer, allowed-GPU memory queue and atomic recovery."""
from util import *
import signal

def alive(pid):
 try:
  p=Path('/proc')/str(pid);return p.exists() and p.joinpath('stat').read_text().split()[2]!='Z' and p.joinpath('cwd').resolve()==ROOT
 except (FileNotFoundError,PermissionError):return False

def available_gpu():
 p=subprocess.run(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);rows=[tuple(int(v.strip()) for v in line.split(',')) for line in p.stdout.splitlines()];rows=[(g,m) for g,m in rows if m>=28000]
 return max(rows,key=lambda x:x[1])[0] if rows else None

def start(name,gpu):
 rec=ROOT/'receipts'/f'{name}_process.json'
 if rec.exists() and alive(load(rec)['pid']):return load(rec)
 if name=='train' and (ROOT/'receipts/train.error.json').exists():
  e=load(ROOT/'receipts/train.error.json');dump(ROOT/'audits/resource_recoveries'/f'previous_{e["pid"]}.json',e);(ROOT/'receipts/train.error.json').unlink()
 log=(ROOT/'logs'/f'{name}.log').open('a');p=subprocess.Popen([PYTHON,'-B',name+'.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);v=dict(pid=p.pid,GPU=gpu,entry=name+'.py',started=stamp());dump(rec,v);event(ROOT/'logs/runner_events.jsonl',event='CHILD_STARTED',task=name,**v);return v

def main():
 with lock('runner.lock'):
  protocol_sha();children={};failure=None;recoveries=0
  if not (ROOT/'receipts/cache.complete.json').exists():children['data']=start('data',0)
  while True:
   for name,rec in list(children.items()):
    if alive(rec['pid']):continue
    ok=(ROOT/'receipts/cache.complete.json').exists() if name=='data' else (ROOT/'audits/preflight.json').exists() if name=='preflight' else (ROOT/'completion.json').exists() if name=='train' else (ROOT/'final_integrity.json').exists() if name=='finalize' else (ROOT/'logs/publication_remote_final.json').exists()
    if not ok:
     errpath=ROOT/'receipts/train.error.json';e=load(errpath) if name=='train' and errpath.exists() else {}
     if name=='train' and ('OutOfMemoryError' in e.get('error','') or 'CUDA out of memory' in e.get('error','')) and recoveries<6:
      recoveries+=1;dump(ROOT/'audits/resource_recoveries'/f'oom_{rec["pid"]}.json',e);event(ROOT/'logs/runner_events.jsonl',event='TRANSIENT_OOM_REQUEUE',task=name,pid=rec['pid'],GPU=rec['GPU'],recovery=recoveries)
     else:failure=dict(task=name,pid=rec['pid'],log=str(ROOT/'logs'/f'{name}.log'))
    del children[name]
   if failure:
    for rec in children.values():
     if alive(rec['pid']):os.kill(rec['pid'],signal.SIGTERM)
    dump(ROOT/'status.json',dict(status='LOCAL_COMPLETE_PUBLICATION_FAILED' if failure['task']=='publish' else 'BLOCKED',complete=False,local_experiment_complete=(ROOT/'final_integrity.json').exists(),failure=failure,pid=os.getpid(),time=stamp()));return
   waiting=None
   if not (ROOT/'audits/preflight.json').exists() and 'preflight' not in children:
    gpu=available_gpu()
    if gpu is None:waiting='WAITING_FOR_GPU_MEMORY_28000MiB'
    else:children['preflight']=start('preflight',gpu)
   if (ROOT/'audits/preflight.json').exists() and not (ROOT/'completion.json').exists() and 'train' not in children:
    assert load(ROOT/'audits/preflight.json')['status']=='PASS';gpu=available_gpu()
    if gpu is None:waiting='WAITING_FOR_GPU_MEMORY_28000MiB'
    else:children['train']=start('train',gpu)
   if (ROOT/'completion.json').exists() and not (ROOT/'final_integrity.json').exists() and (ROOT/'finalize.py').exists() and 'finalize' not in children:
    gpu=available_gpu()
    if gpu is None:waiting='WAITING_FOR_FINALIZER_GPU_MEMORY'
    else:children['finalize']=start('finalize',gpu)
   if (ROOT/'final_integrity.json').exists() and not (ROOT/'logs/publication_remote_final.json').exists() and (ROOT/'publish.py').exists() and 'publish' not in children:children['publish']=start('publish',None)
   p=ROOT/'receipts/train.status.json';worker=load(p) if p.exists() else {};complete=(ROOT/'logs/publication_remote_final.json').exists() and not children;committed=load(ROOT/'completion.json')['committed_updates'] if (ROOT/'completion.json').exists() else worker.get('committed_updates',0)
   phase='COMPLETE' if complete else waiting or ('TRAINING_VALIDATION' if 'train' in children else 'FINALIZING' if 'finalize' in children else 'PUBLISHING' if 'publish' in children else 'PREFLIGHT')
   dump(ROOT/'status.json',dict(status='COMPLETE' if complete else 'RUNNING',complete=complete,phase=phase,pid=None if complete else os.getpid(),GPU=children.get('train',{}).get('GPU'),committed_updates=committed,cache_completed=len(list((ROOT/'cache/frozen').glob('*/completion.json'))),cache_planned=18,evaluation_points_completed=len(list((ROOT/'metrics/points').glob('*.json'))),children=[dict(task=n,**rec) for n,rec in children.items()],worker=worker,transient_OOM_recoveries=recoveries,waiting_reason=waiting,heartbeat=time.time(),local_time=stamp()))
   if complete:return
   time.sleep(10)
if __name__=='__main__':main()
