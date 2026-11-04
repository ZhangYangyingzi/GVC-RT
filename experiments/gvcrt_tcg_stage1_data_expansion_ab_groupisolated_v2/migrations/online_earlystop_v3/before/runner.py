"""Exclusive persistent revision-v2 orchestrator; genuine failures never become PASS."""
from ab_common import *
import traceback
STAGES=[('reuse','reuse.py','audits/reuse.json'),('source_groups','source_graph.py','audits/source_groups.json'),('plans','prepare_plans.py','audits/plans_frozen.json'),('preflight','preflight.py','preflight.json'),('runtime_equivalence','equivalence.py','audits/runtime_equivalence.json'),('train_A','train_launch.py','A_clean_old_pool/training_summary.json'),('train_B','train_launch.py','B_expanded_ulong/training_summary.json'),('new_evaluation','evaluation_runner.py','summary.json'),('integrity','integrity.py','final_integrity.json'),('publication','publish.py','publication_status.json')]
def main():
 lease=(ROOT/'runner.lock').open('a')
 try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  event('TRACK_EXISTING_RUNNER',status_file=str(ROOT/'run_status.json'),action='No duplicate stage or optimizer update launched');lease.close();return 0
 with lease:
  assert load(ROOT/'task_identity.json')['task']==TASK_ID
  for stage,script,acceptance in STAGES:
   receipt=ROOT/f'audits/stage_{stage}.json';path=ROOT/script
   if receipt.exists():
    r=load(receipt)
    if r['status']=='PASS':
     assert r['script_sha256']==sha(path),('STAGE_CODE_CHANGED_AFTER_ACCEPTANCE',stage)
     assert (ROOT/acceptance).exists() and load(ROOT/acceptance)['status'] in ('PASS','PUBLISHED')
     continue
   status(stage);event('STAGE_STARTED',stage=stage)
   # Track an earlier launch of this same revision; never duplicate its workers.
   mutex={'source_groups':'source_graph.lock','plans':'prepare_plans.lock','runtime_equivalence':'equivalence.lock','new_evaluation':'evaluation.lock','integrity':'integrity.lock','publication':'publication.lock'}.get(stage)
   if mutex:
    with (ROOT/mutex).open('a') as f:
     while True:
      try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
      except BlockingIOError:event('TRACK_EXISTING_STAGE',stage=stage,profile_records=len(list((ROOT/'graph/profiles').glob('*.json'))) if stage=='source_groups' else None);time.sleep(30)
   done=ROOT/acceptance
   if done.exists() and load(done)['status'] in ('PASS','PUBLISHED') and stage in ('reuse','source_groups','plans','runtime_equivalence','new_evaluation','integrity'):
    dump(receipt,dict(status='PASS',stage=stage,script_sha256=sha(path),acceptance_sha256=sha(done),tracked_existing_v2_stage=True));continue
   env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   with (ROOT/f'logs/{stage}.log').open('a') as log:
    result=subprocess.run([PYTHON,'-B','-u',str(path),'--stage',stage],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    status(stage,'FAILED',exit_code=result.returncode,log=f'logs/{stage}.log',resume_command=f'{PYTHON} -B -u {ROOT}/runner.py')
    dump(ROOT/f'audits/failure_{int(time.time())}_{stage}.json',dict(status='FAILED',stage=stage,exit_code=result.returncode,log=f'logs/{stage}.log'))
    return result.returncode
   assert done.exists() and load(done)['status'] in ('PASS','PUBLISHED'),('MISSING_ACCEPTANCE',stage)
   dump(receipt,dict(status='PASS',stage=stage,time=time.time(),script_sha256=sha(path),acceptance_sha256=sha(done)))
  final=load(ROOT/'final_integrity.json');publication=load(ROOT/'publication_status.json');assert final['status']=='PASS' and publication['status']=='PUBLISHED' and publication['remote_verified']
  status('complete','PASS',protocol_revision=REVISION,verified_commit=publication['verified_commit'])
  event('ALL_STAGES_COMPLETE',verified_commit=publication['verified_commit']);return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception:
  status('runner','FAILED',error=traceback.format_exc());traceback.print_exc();sys.exit(1)
