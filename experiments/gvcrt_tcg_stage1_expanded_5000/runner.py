"""Sequential full-budget orchestration. Failure blocks downstream evaluation."""
import csv,os,subprocess,sys,traceback
from common_expanded import *
def run(command,log):
 with (ROOT/'logs'/log).open('a') as f:
  p=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID'))
 return p.returncode
def main():
 status=dict(status='RUNNING',stage='TRAINING',committed_global_updates=0,training_status='RUNNING',evaluation_status='NOT_STARTED')
 dump(ROOT/'run_status.json',status)
 command=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'train.py')]
 code=run(command,'torchrun.log')
 if code or not (ROOT/'audits/training_integrity.json').exists():
  cursor=0
  if (ROOT/'checkpoints/latest.pt').exists():
   import torch
   cursor=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True)['committed_global_updates']
  status.update(status='FAILED',failure_stage='TRAINING',training_status='FAILED',committed_global_updates=cursor,
                evaluation_status='BLOCKED_TRAINING_NOT_COMPLETE',error_log='logs/torchrun.log',exit_code=code)
  dump(ROOT/'run_status.json',status);dump(ROOT/'training_summary.json',status)
  dump(ROOT/'final_integrity.json',dict(status='FAILED',stage='TRAINING',committed_global_updates=cursor,evaluation_complete=False))
  (ROOT/'REPORT.md').write_text('# TCG Stage 1 expanded run\n\nTraining status: FAILED.\nCommitted global updates: '+str(cursor)+' / 5000.\n\nFormal evaluation: NOT RUN; training did not pass.\n\nProtocol: protocol.json. Data coverage: data_plan.json. Log: logs/torchrun.log.\n')
  for name in ('metrics.csv','deltas.csv','warping_error.csv'):
   (ROOT/name).write_text('status\nNOT_RUN\n')
  dump(ROOT/'evaluation/run_status.json',dict(status='BLOCKED',reason='Training did not complete'))
  event('RUN_FAILED',stage='TRAINING',committed_global_updates=cursor)
  return 1
 status.update(stage='EVALUATION',training_status='PASS',committed_global_updates=5000,evaluation_status='RUNNING')
 dump(ROOT/'run_status.json',status)
 code=run([sys.executable,'-B','-u',str(ROOT/'evaluate.py')],'evaluation.log')
 status.update(stage='COMPLETE' if code==0 else 'EVALUATION',status='PASS' if code==0 else 'FAILED',
               evaluation_status='PASS' if code==0 else 'FAILED',exit_code=code)
 dump(ROOT/'run_status.json',status)
 return code
if __name__=='__main__':sys.exit(main())
