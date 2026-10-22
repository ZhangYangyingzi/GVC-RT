"""Bounded GPU 5 tail shard; worker locks prevent duplicate evaluation."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
TASKS=json.loads((HERE/'evaluation/task_manifest.json').read_text())[-15:]
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
def main():
 for x in TASKS:
  t=x['task'];key=f"{t['dataset']}_{t['video']}_q{t['qp']}"
  p=HERE/'tasks'/f'{key}.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='PASS':continue
  cmd=[PYTHON,str(HERE/'intervene41.py'),'--task',json.dumps(t),'--gpu','5']
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='5',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=HERE/'logs'/f'{key}.gpu5.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a') as out:
   rc=subprocess.run(cmd,cwd=HERE.parents[1],env=env,stdout=out,stderr=subprocess.STDOUT).returncode
  print('EXTRA',key,rc,flush=True)
  if rc:break
if __name__=='__main__':main()
