"""Exclusive, restartable stage runner. A stage never implies downstream PASS."""
from ab_common import *
import traceback
def main():
 with lock('runner.lock'):
  for stage,script in [('audit','audit.py'),('inventory','inventory.py'),('plans','prepare_plans.py'),('preflight','preflight.py'),('runtime_equivalence','equivalence.py'),('train_A','train_launch.py'),('train_B','train_launch.py'),('parent_GTFlow','evaluation_runner.py'),('development_and_test','evaluation_runner.py'),('integrity','integrity.py'),('publication','publish.py')]:
   receipt=ROOT/f'audits/stage_{stage}.json'
   if receipt.exists() and load(receipt)['status']=='PASS':continue
   status(stage);event('STAGE_STARTED',stage=stage)
   if stage=='inventory':
    # The same task's inventory may have been launched while engineering code
    # was being prepared. Acquire its existing lock before deciding to run it.
    with (ROOT/'inventory.lock').open('a') as f:
     while True:
      try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
      except BlockingIOError:
       event('TRACK_EXISTING_INVENTORY',record=load(ROOT/'inventory/status.json') if (ROOT/'inventory/status.json').exists() else {})
       time.sleep(30)
    if (ROOT/'inventory/status.json').exists() and load(ROOT/'inventory/status.json')['status']=='PASS':
     dump(receipt,dict(status='PASS',stage=stage,existing_task_tracked=True,time=time.time()));continue
   path=ROOT/script
   if not path.is_file():
    status(stage,'BLOCKED',error='REQUIRED_STAGE_IMPLEMENTATION_NOT_YET_READY',resume_command=f'{PYTHON} -u {ROOT}/runner.py')
    raise RuntimeError(f'Stage implementation required: {script}')
   with (ROOT/f'logs/{stage}.log').open('a') as log:
    result=subprocess.run([PYTHON,'-B','-u',str(path),'--stage',stage],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
   if result.returncode:
    if (ROOT/'audits/data_source_conflict.json').exists() and load(ROOT/'audits/data_source_conflict.json')['status']=='BLOCKED':
     status('source_group_split_isolation','BLOCKED',failed_stage=stage,log=f'logs/{stage}.log',formal_optimizer_updates=0,resume_command=f'{PYTHON} -u {ROOT}/runner.py')
     with (ROOT/'logs/blocked_finalize.log').open('a') as log:
      finalize=subprocess.run([PYTHON,'-B','-u',str(ROOT/'finalize_blocked.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
     if finalize.returncode:event('BLOCKED_PUBLICATION_FAILED',log='logs/blocked_finalize.log')
     return 2
    status(stage,'FAILED',exit_code=result.returncode,log=f'logs/{stage}.log',resume_command=f'{PYTHON} -u {ROOT}/runner.py');return result.returncode
   dump(receipt,dict(status='PASS',stage=stage,time=time.time(),script_sha256=sha(path)))
  status('complete','PASS');return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception:traceback.print_exc();sys.exit(1)
