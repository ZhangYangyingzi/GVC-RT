import os,sys,subprocess
from common_expanded import *
def main():
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7')
 import torch
 rows=[r for p in sorted((ROOT/'plans').glob('base_rows_*.json')) for r in load(p)]
 assert len(rows)==5000
 manifests={d:load(ROOT/f'plans/train_manifest_{d}.local.json')['videos'] for d in ('uvg','ulong')}
 look={d:{v['filename' if d=='ulong' else 'name']:v for v in vs} for d,vs in manifests.items()}
 sources={}
 for r in rows:
  m=look[r['domain']][r['video']]
  assert r['source_path']==m['path' if r['domain']=='ulong' else 'source_path']
  assert r['source_sha256']==m['sha256' if r['domain']=='ulong' else 'source_sha256']
  assert Path(r['source_path']).is_file(),r['source_path']
  sources[r['source_path']]=r['source_sha256']
 assert len(sources)>=940
 assert sha(RAFT)=='ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322'
 assert sha(SOURCE)=='06cf8f53f432d1d1f0fed7e8a72e911af70f5ee1d2a0b869ceaabef94b1f8744'
 assert torch.cuda.is_available() and torch.cuda.device_count()==4
 devices=[]
 for rank,gpu in enumerate((4,5,6,7)):
  with torch.cuda.device(rank):
   x=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
   devices.append(dict(rank=rank,physical_GPU=gpu,name=torch.cuda.get_device_name(rank),free_bytes=free,total_bytes=total,CUDA=True));del x
 dump(ROOT/'preflight.json',dict(status='PASS',devices=devices,data_source_files=len(sources),
      data_source_presence=True,source_hashes_match_audited_manifests=True,
      per_frame_RGB_and_crop_hash_validation='on every sample read using inherited PlanFrames',
      raft_sha256=sha(RAFT),source_checkpoint_sha256=sha(SOURCE),minimum_memory_threshold=None))
 event('PREFLIGHT_PASS',physical_gpus=[4,5,6,7])
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  dump(ROOT/'preflight.json',dict(status='FAILED',error=str(e),traceback=traceback.format_exc()))
  raise
