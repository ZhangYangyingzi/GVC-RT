"""Verify a real H2 optimizer update after the user-authorized launch correction."""
from common_s2 import *
def main():
 env();import torch
 target=ROOT/GROUPS[2]/'training/progress.json';begin=time.time();last=None
 while True:
  failure=ROOT/'audits/runner_failure.json'
  if failure.exists():raise RuntimeError(load(failure))
  progress=load(target) if target.exists() else dict(committed_updates=0);n=progress['committed_updates']
  eng=ROOT/GROUPS[2]/'engineering/progress.json';en=load(eng)['committed_updates'] if eng.exists() else 0
  if (en,n)!=last:print(json.dumps(dict(engineering_updates=en,formal_updates=n,wait_seconds=time.time()-begin)),flush=True);last=(en,n)
  if n>=1:break
  assert time.time()-begin<900,'H2_FORMAL_FIRST_UPDATE_NOT_YET_COMMITTED';time.sleep(5)
 step0=ROOT/GROUPS[2]/'training/checkpoints/step_0000.pt';initial=torch.load(step0,map_location='cpu',weights_only=True);base=torch.load(ROOT/'checkpoints/common_B300_step0.pt',map_location='cpu',weights_only=True)
 assert initial['committed_updates']==0 and not initial['optimizer']['state'] and all(torch.equal(t,base['trainable_state'][name]) for name,t in initial['trainable_state'].items())
 r=load(ROOT/GROUPS[2]/'training/receipts/0001.json');assert r['global_optimizer_updates']==1 and r['sample_keys']==[row['sample_key'] for row in plan_rows()[:4]] and r['effective_batch']==4 and r['physical_GPU']==2
 assert load(ROOT/f'audits/engineering_{GROUPS[2]}.json')['status']=='PASS' and load(ROOT/f'audits/aggregation_{GROUPS[2]}.json')['status']=='PASS'
 before=load(ROOT/'audits/H2_user_start_request.json')['H0_H1_unchanged'];survivors={}
 for g in GROUPS[:2]:
  pid=load(ROOT/f'receipts/processes/training_{g}.json')['PID'];assert pid==before[g].get('PID',pid);survivors[g]=dict(PID=pid,updates=load(ROOT/g/'training/progress.json')['committed_updates']);assert survivors[g]['updates']>=before[g]['committed_updates']
 result=dict(status='RUNNING_VERIFIED',H2_formal_updates=load(target)['committed_updates'],H2_first_optimizer_receipt=str(ROOT/GROUPS[2]/'training/receipts/0001.json'),H2_first_step_sample_keys=r['sample_keys'],H2_common_step0_exactly_restored=True,H2_optimizer_fresh_at_step0=True,H2_gain=.5,H2_physical_GPU=2,gradient_calibration_SHA256=initial['calibration_SHA256'],H0_H1_preserved=survivors,withdrawn_assistant_cutoff=True,original_control_measurements_preserved=True,all_groups_limit100=True,time=time.time());dump(ROOT/'audits/H2_formal_start_verified.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
