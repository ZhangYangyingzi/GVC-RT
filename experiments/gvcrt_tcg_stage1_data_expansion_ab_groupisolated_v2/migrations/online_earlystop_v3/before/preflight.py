from ab_common import *
def main():
 deadline=time.monotonic()+86400
 while True:
  output=subprocess.run([PYTHON,'-B','-u',str(ROOT/'preflight.py'),'--probe'],cwd=REPO,capture_output=True,text=True)
  if output.returncode==0:break
  dump(ROOT/'preflight.json',dict(status='WAITING_RESOURCES',physical_GPUs=[4,5,6,7],error=output.stderr[-4000:],minimum_free_memory_bytes=None,check='CUDA driver/device initialization only; no arbitrary free-memory threshold'))
  status('preflight','WAITING_RESOURCES',physical_GPUs=[4,5,6,7]);event('WAITING_FOR_AUTHORIZED_CUDA',error=output.stderr[-1000:])
  if time.monotonic()>=deadline:raise RuntimeError('CUDA_RESOURCE_WAIT_EXCEEDED_24_HOURS')
  time.sleep(30)
 assert load(ROOT/'audits/plans_frozen.json')['status']=='PASS'
 assert load(ROOT/'split_audit.json')['status']=='PASS'
 os.environ.update(CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID')
 import torch
 assert torch.cuda.is_available() and torch.cuda.device_count()==4
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
 physical=[]
 for line in raw.strip().splitlines():
  index,bus,uuid,name,total,free=[x.strip() for x in line.split(',')]
  physical.append(dict(index=int(index),pci_bus_id=bus,UUID=uuid,name=name,total_MiB=int(total),free_MiB=int(free)))
 assert [g['index'] for g in sorted(physical,key=lambda g:g['pci_bus_id'])]==list(range(len(physical))),'GPU_INDEX_PCI_ORDER_MISMATCH'
 cards=[]
 for rank in range(4):
  torch.cuda.set_device(rank);x=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize(rank);free,total=torch.cuda.mem_get_info(rank)
  cards.append(dict(rank=rank,physical_GPU=4+rank,name=torch.cuda.get_device_name(rank),free_bytes=free,total_bytes=total,CUDA_allocation='PASS',physical_identity=next(g for g in physical if g['index']==4+rank)));del x
 from ab_model_context import RAFT,PILOT
 assert sha(RAFT)==load(P/'protocol.json')['raft_sha256'] and sha(PILOT)==load(P/'protocol.json')['source_checkpoint']['sha256']
 for group in ('training_sources','evaluation_sources'):
  for name,h in load(P/'source_manifest.json').get(group,{}).items():
   if name.startswith('src/') or name.startswith('experiments/gvcrt_neural_wrapper_v6_40_'):assert sha(REPO/name)==h,('PINNED_RUNTIME_SOURCE_MISMATCH',name)
 for n in ('GVC-RT_I.pt','GVC-RT_P.pt'):
  f=REPO/'checkpoints'/n
  if not f.exists():f.symlink_to(ORIGINAL/'checkpoints'/n)
  assert sha(f)==next(c['sha256'] for c in load(P/'source_manifest.json')['model_file_sources'] if c['filename']==n)
 dump(ROOT/'preflight.json',dict(status='PASS',cards=cards,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,raft_sha256=sha(RAFT),pilot_full_state_sha256=sha(PILOT),parent_TCG_sha256=sha(P/'checkpoints/step_5000.pt'),minimum_free_GPU_threshold=None,other_processes_untouched=True,**disk()))
if __name__=='__main__':
 if '--probe' in sys.argv:
  os.environ.update(CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID')
  import torch
  assert torch.cuda.is_available() and torch.cuda.device_count()==4
  for rank in range(4):
   torch.cuda.set_device(rank);x=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize(rank);del x
 else:main()
