"""Continue authorized publication after local finalization, without any training."""
from util import *
from runner import alive
with lock('auto_publish.lock'):
 try:
  while True:
   p=ROOT/'receipts/reconcile_finalize.status.json';rec=load(p) if p.exists() else {}
   if (ROOT/'final_integrity.json').exists() and not alive(rec.get('pid',-1)):
    assert load(ROOT/'final_integrity.json')['status']=='PASS';break
   if rec and not alive(rec['pid']) and rec.get('stage')!='LOCAL_COMPLETE_GITHUB_PENDING':raise RuntimeError(('FINALIZATION_STOPPED',rec))
   heartbeat('auto_publish','WAITING_FOR_LOCAL_FINALIZATION',waiting_for=str(ROOT/'final_integrity.json'));time.sleep(10)
  # No model evaluation or optimizer work remains at this point.
  for name in ['finalize','render0','render1','profile_parts','A_zero_base','B_flow_guided','fix_metrics0','fix_metrics1','fix_metrics2','fix_metrics3']:
   p=ROOT/'receipts'/f'{name}.status.json'
   if p.exists():assert not alive(load(p)['pid']),('OWN_WORKER_STILL_RUNNING',name)
  subprocess.run([PYTHON,'-B','manage.py','verify'],cwd=ROOT,check=True)
  heartbeat('auto_publish','PUBLISHING_AND_VERIFYING_REMOTE')
  subprocess.run([PYTHON,'-B','publish.py'],cwd=ROOT,check=True)
  heartbeat('auto_publish','COMPLETE')
 except Exception as e:
  import traceback
  error=dict(status='STOPPED_WITH_ERROR',error=repr(e),traceback=traceback.format_exc(),log=str(ROOT/'logs/auto_publish.log'),time=stamp());dump(ROOT/'receipts/auto_publish.error.json',error);heartbeat('auto_publish','STOPPED_WITH_ERROR',error=repr(e));s=load(ROOT/'status.json');s.update(status='STOPPED_WITH_ERROR',stage='PUBLICATION_OR_FINALIZATION_ERROR',error=error);dump(ROOT/'status.json',s);raise
