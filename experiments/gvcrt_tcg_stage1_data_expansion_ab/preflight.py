from ab_common import *
def main():
 assert load(ROOT/'audits/plans_frozen.json')['status']=='PASS'
 assert load(ROOT/'split_audit.json')['status']=='PASS'
 os.environ.update(CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID')
 import torch
 assert torch.cuda.is_available() and torch.cuda.device_count()==4
 cards=[]
 for rank in range(4):
  torch.cuda.set_device(rank);x=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize(rank);free,total=torch.cuda.mem_get_info(rank)
  cards.append(dict(rank=rank,physical_GPU=4+rank,name=torch.cuda.get_device_name(rank),free_bytes=free,total_bytes=total,CUDA_allocation='PASS'));del x
 from ab_model_context import RAFT,PILOT
 assert sha(RAFT)==load(P/'protocol.json')['raft_sha256'] and sha(PILOT)==load(P/'protocol.json')['source_checkpoint']['sha256']
 for n in ('GVC-RT_I.pt','GVC-RT_P.pt'):
  f=REPO/'checkpoints'/n
  if not f.exists():f.symlink_to(ORIGINAL/'checkpoints'/n)
  assert sha(f)==next(c['sha256'] for c in load(P/'source_manifest.json')['model_file_sources'] if c['filename']==n)
 dump(ROOT/'preflight.json',dict(status='PASS',cards=cards,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,raft_sha256=sha(RAFT),pilot_full_state_sha256=sha(PILOT),parent_TCG_sha256=sha(P/'checkpoints/step_5000.pt'),minimum_free_GPU_threshold=None,other_processes_untouched=True,**disk()))
if __name__=='__main__':main()
