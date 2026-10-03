"""Detached resumable inference-only scheduler, restricted to idle GPUs 4-7."""
import argparse,fcntl,traceback
from stride_io import *
def idle_gpus():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 busy={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
 return [int(i) for line in out.splitlines() for i,u,free in [[x.strip() for x in line.split(',')]] if int(i) in (4,5,6,7) and u not in busy and int(free)>=22000]
def prepare():
 source=load(V65/'source_manifest.json');vs=[v for v in source['videos'] if v['name'] in VIDEOS]
 assert len(vs)==5 and all(v['dataset']=='uvg' and v['width']==1920 and v['height']==1080 for v in vs)
 oldcfg=load(V65/'config.json');cps={k:oldcfg['checkpoints'][k] for k in ('V62','V64')}
 expected={'V62':'fe63a6f00ca6bae59b23c8683146dbe64b6be67c1463a552af851bbfc31e9cba','V64':'1c7c9a3f89cbbe4abd4ba6f6f7c963d2fab22737e9928fe222f3aa4c8a8c75e1'}
 assert all(sha(cps[k]['path'])==cps[k]['sha256']==expected[k] for k in cps)
 assert oldcfg['force_zero_thres']==.12
 assert load(V65/'final_integrity.json')['status']=='PASS'
 if not (ROOT/'source_manifest.json').exists():dump(ROOT/'source_manifest.json',dict(videos=vs,origin=str(V65/'source_manifest.json'),origin_sha256=sha(V65/'source_manifest.json')))
 if not (ROOT/'config.json').exists():dump(ROOT/'config.json',dict(experiment='B',methods=METHODS,checkpoints=cps,force_zero_thres=.12,external_qps=[4],comparison_fps=30,compression_hash=oldcfg['compression_hash'],no_training=True,expected_pairs=120,expected_points=600,allowed_gpus=[4,5,6,7],anchors=ANCHORS,strides=STRIDES))
 deps={str(p):sha(p) for p in (V65/'v65_io.py',V65/'source_manifest.json',V65/'config.json',V65/'qp_semantics_audit.json',V52/'canonical_loader.py',V52/'ffmpeg_conversion_audit.json',AUDIT/'source_stats.py',ENGINE/'engine.py')}
 for k in cps:deps[cps[k]['path']]=expected[k]
 if (ROOT/'audits/dependency_hashes.json').exists():assert deps==load(ROOT/'audits/dependency_hashes.json')
 else:dump(ROOT/'audits/dependency_hashes.json',deps)
 dump(ROOT/'audits/preflight.json',dict(status='PASS',no_training=True,checkpoint_hashes_pass=True,old_v65_integrity_pass=True,compression_hash=oldcfg['compression_hash'],source_conversion_sha256=deps[str(V52/'ffmpeg_conversion_audit.json')]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
 os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 (ROOT/'logs').mkdir(parents=True,exist_ok=True)
 if a.launch:
  pidfile=ROOT/'pipeline.pid'
  if pidfile.exists():
   pid=int(pidfile.read_text())
   try:
    os.kill(pid,0)
    if str(ROOT/'run_pipeline.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes():
     print(json.dumps(dict(directory=str(ROOT),PID=pid,pipeline_status=load(ROOT/'pipeline_status.json'),log_path=str(ROOT/'logs/pipeline.log'))));return
   except (ProcessLookupError,FileNotFoundError):pass
  with (ROOT/'logs/pipeline.log').open('a') as log:proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
  pidfile.write_text(str(proc.pid)+'\n')
  print(json.dumps(dict(directory=str(ROOT),PID=proc.pid,pipeline_status='LAUNCHED',log_path=str(ROOT/'logs/pipeline.log'))));return
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 active={}
 def status(stage,**kw):dump(ROOT/'pipeline_status.json',dict(status=stage,PID=os.getpid(),updated_unix=time.time(),active=[dict(video=k,gpu=g,PID=p.pid) for k,(p,g,l) in active.items()],**kw))
 try:
  status('PREFLIGHT');prepare()
  if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':status('PASS');return
  pending=list(VIDEOS)
  while pending or active:
   for v,(proc,gpu,log) in list(active.items()):
    rc=proc.poll()
    if rc is not None:
     log.close();del active[v]
     if rc:raise RuntimeError(f'{v} worker exited {rc}; see logs/{v}.log')
     print('FINISHED',v,flush=True)
   pending=[v for v in pending if not (ROOT/'parts'/f'{v}_done.json').exists()]
   reserved={g for _,g,_ in active.values()}
   for gpu in idle_gpus():
    if gpu in reserved or not pending:continue
    v=pending.pop(0);log=(ROOT/'logs'/f'{v}.log').open('a')
    proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'video_worker.py'),'--video',v,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
    active[v]=(proc,gpu,log);print('STARTED',v,'GPU',gpu,'PID',proc.pid,flush=True)
   status('RUNNING' if active else 'WAITING_FOR_IDLE_GPU',pending=len(pending),pairs_completed=len(list((ROOT/'parts').glob('*/a*_s*.json'))))
   if pending or active:time.sleep(20)
  status('AGGREGATING');subprocess.run([PYTHON,'-B','-u',str(ROOT/'aggregate.py')],cwd=REPO,check=True)
  status('PASS')
 except Exception:
  status('FAIL',traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
  for proc,gpu,log in active.values():proc.wait();log.close()
  raise
if __name__=='__main__':main()
