"""Persistent v6.36 execution coordinator; waits for the sole training writer."""
import json,os,subprocess,sys,time,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent;PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python';REPO=R.parents[1]
def jread(p):return json.load(open(p)) if p.exists() else {}
def update(x):json.dump(x,open(R/'pipeline_status.json','w'),indent=2)
def main():
 lock=(R/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (R/'logs').mkdir(exist_ok=True);log=open(R/'logs/driver.log','a',buffering=1)
 log.write(f'DRIVER_START {time.time()} PID={os.getpid()}\n')
 while True:
  st=jread(R/'branches/B_basicLoss_noP_500/training_status.json')
  update({'status':'TRAINING' if st.get('status')!='PASS' else 'TRAINING_COMPLETE','training':st,'evaluation':jread(R/'evaluation_status.json'),'updated_unix':time.time()})
  if st.get('status')=='PASS' and st.get('committed_updates')==500:break
  if st.get('status') in ('FAILED','BLOCKED'):
   log.write('TRAINING_TERMINAL_FAILURE '+json.dumps(st)+'\n');return 2
  # Worker PIDs may live in a separate job namespace. Use the atomically
  # appended per-step training log as the heartbeat instead of host ps.
  log_path=R/'branches/B_basicLoss_noP_500/training_log.jsonl'
  if st.get('status')=='RUNNING' and log_path.exists() and time.time()-log_path.stat().st_mtime>1200:
   log.write('TRAINING_HEARTBEAT_STALE '+json.dumps(st)+'\n');update({'status':'BLOCKED_TRAINING_HEARTBEAT_STALE','training':st,'updated_unix':time.time()});return 3
  time.sleep(60)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def run(args):
  log.write('RUN '+json.dumps(args)+'\n');p=subprocess.run(args,cwd=R,env=env);log.write('EXIT '+str(p.returncode)+' '+json.dumps(args)+'\n');return p.returncode
 if run([PYTHON,'prepare_eval36.py']):return 4
 if run([PYTHON,'-u','run_eval_queue36.py']):return 5
 if jread(R/'evaluation_status.json').get('status')!='PASS':return 6
 if run([PYTHON,'aggregate36.py']):return 7
 # matplotlib is unavailable in the gvc-rt environment; use the installed plotting runtime only for plots.
 if run(['python3','plot36.py']):return 8
 if run([PYTHON,'finalize36.py']):return 9
 if run([PYTHON,'publish36.py']):return 10
 update({'status':'COMPLETE','training':jread(R/'branches/B_basicLoss_noP_500/training_status.json'),'evaluation':jread(R/'evaluation_status.json'),'integrity':jread(R/'final_integrity.json'),'publication':jread(R/'publication_status.json'),'updated_unix':time.time()})
 log.write('PIPELINE_COMPLETE\n');return 0
if __name__=='__main__':sys.exit(main())
