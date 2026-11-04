"""Serial four-GPU A then B; resumable latest/receipts, never two independent seeds."""
from ab_common import *
def wait_lease(path,arm):
 path.parent.mkdir(parents=True,exist_ok=True);lease=path.open('a')
 while True:
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);return lease
  except BlockingIOError:event('TRACK_EXISTING_TRAINING',arm=arm,lease=path.name);time.sleep(30)
def already_complete(arm):
 p=ROOT/arm/'training_summary.json'
 if not p.exists() or load(p)['status']!='PASS':return False
 import torch
 protocol=load(ROOT/'protocol.json');S=protocol['additional_global_updates'];s=load(p)
 checkpoint=ROOT/arm/'checkpoints/latest.pt';manifest=load(ROOT/arm/'checkpoints_manifest.json')
 expected=next(r for r in manifest['checkpoints'] if r['filename']=='latest.pt');assert sha(checkpoint)==expected['sha256']
 st=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert s['committed_additional_updates']==st['additional_step']==st['plan_cursor']==S
 assert s['absolute_optimizer_step']==st['absolute_optimizer_step']==5000+S and st['rank_sample_cursor']==4*S
 assert st['arm']==arm and st['parent_checkpoint_sha256']==PARENT_SHA and st['protocol_sha256']==sha(ROOT/'protocol.json')
 assert st['plan_index_sha256']==protocol['plan_index_sha256'][arm]
 assert {int(x['step']) for x in st['optimizer']['state'].values()}=={5000+S}
 return True
def main():
 stage=sys.argv[sys.argv.index('--stage')+1];arm={'train_A':'A_clean_old_pool','train_B':'B_expanded_ulong'}[stage]
 if arm=='B_expanded_ulong':assert load(ROOT/'A_clean_old_pool/training_summary.json')['status']=='PASS'
 with wait_lease(ROOT/arm/'training.lock',arm) as lease:
  # Rank zero keeps a separate lifetime lock if a launcher is interrupted.
  # Release that probe before starting ranks: rank zero will acquire it itself.
  with wait_lease(ROOT/arm/'rank0_active.lock',arm):pass
  if already_complete(arm):event('TRACK_COMPLETED_TRAINING',arm=arm)
  else:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID',TCG_ARM=arm,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'train.py')]
   dump(ROOT/f'{arm}/launcher.json',dict(command=command,physical_GPUs=[4,5,6,7],parent_checkpoint_sha256=PARENT_SHA,started=time.time()))
   result=subprocess.run(command,cwd=REPO,env=env,pass_fds=(lease.fileno(),));assert result.returncode==0
   assert already_complete(arm)
 if arm=='B_expanded_ulong':
  arms={a:load(ROOT/f'{a}/training_summary.json') for a in ('A_clean_old_pool','B_expanded_ulong')};assert all(s['status']=='PASS' for s in arms.values())
  dump(ROOT/'training_summary.json',dict(status='PASS',protocol_revision=REVISION,S=load(ROOT/'protocol.json')['additional_global_updates'],arms=arms,paired_parent_initialization=True))
  dump(ROOT/'checkpoints_manifest.json',dict(arms={a:load(ROOT/f'{a}/checkpoints_manifest.json')['checkpoints'] for a in arms},parent_checkpoint=dict(filename='step_5000.pt',sha256=PARENT_SHA),large_files_local_only=True))
if __name__=='__main__':main()
