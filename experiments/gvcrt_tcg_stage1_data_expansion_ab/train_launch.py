"""Serial four-GPU A then B; resumable latest/receipts, never two independent seeds."""
from ab_common import *
def main():
 stage=sys.argv[sys.argv.index('--stage')+1];arm={'train_A':'A_old_pool','train_B':'B_expanded_ulong'}[stage]
 if arm=='B_expanded_ulong':assert load(ROOT/'A_old_pool/training_summary.json')['status']=='PASS'
 assert not (ROOT/'runner.lock').is_dir()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID',TCG_ARM=arm,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'train.py')]
 dump(ROOT/f'{arm}/launcher.json',dict(command=command,physical_GPUs=[4,5,6,7],parent_checkpoint_sha256=PARENT_SHA,started=time.time()))
 result=subprocess.run(command,cwd=REPO,env=env);assert result.returncode==0
 st=load(ROOT/f'{arm}/training_summary.json');assert st['status']=='PASS' and st['committed_additional_updates']==load(ROOT/'protocol.json')['additional_global_updates']
if __name__=='__main__':main()
