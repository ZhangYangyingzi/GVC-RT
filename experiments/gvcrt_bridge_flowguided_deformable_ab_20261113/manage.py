from util import *
import argparse,math

def verify():
 check_protocol();receipt=load(ROOT/'receipts/evaluation_queue.complete.json');assert receipt['points']==150 and receipt['per_frame_rows']==9600
 for g in ['A_zero_base','B_flow_guided']:
  done=load(ROOT/'checkpoints'/g/'completion.json');assert done['committed_updates']<=300
  cp=readtensor(ROOT/'checkpoints'/g/f"step_{done['committed_updates']:04d}.pt");assert cp['committed_updates']==done['committed_updates'] and cp['protocol_sha256']==check_protocol();assert cp['training_plan_sha256']==sha(ROOT/'training_plan.json')
 if (ROOT/'final_integrity.json').exists():assert load(ROOT/'final_integrity.json')['status']=='PASS'
 return dict(status='VERIFIED',groups={g:load(ROOT/'checkpoints'/g/'completion.json')['committed_updates'] for g in ['A_zero_base','B_flow_guided']},points=150,frames=9600,protocol_sha256=check_protocol())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['status','verify','resume']);a=p.parse_args()
 if a.action=='status':print(json.dumps(load(ROOT/'status.json'),indent=2,ensure_ascii=False))
 elif a.action=='verify':print(json.dumps(verify(),indent=2))
 elif (ROOT/'final_integrity.json').exists():print(json.dumps(verify()|{'resume':'NO_OP_LOCAL_TASK_ALREADY_COMPLETE'},indent=2))
 else:
  from runner import alive
  if (ROOT/'audits/native_validity_correction/plan.json').exists() and all((ROOT/'checkpoints'/g/'completion.json').exists() for g in ['A_zero_base','B_flow_guided']):
   active=ROOT/'receipts/reconcile_finalize.status.json'
   if active.exists() and alive(load(active)['pid']):print(json.dumps(dict(status='RUNNING',attached=load(active)),indent=2))
   else:subprocess.run([PYTHON,'-B','reconcile_finalize.py'],cwd=ROOT,check=True)
  else:subprocess.run([PYTHON,'-B','runner.py'],cwd=ROOT,check=True)
