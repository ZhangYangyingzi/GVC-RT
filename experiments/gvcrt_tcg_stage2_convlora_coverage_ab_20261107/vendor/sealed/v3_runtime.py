"""Audited parent raw/RAFT pipeline, with explicit per-arm physical device mapping."""
from v3_common import *
from ab_runtime import raw,Frames,optimized_loss
import concurrent.futures,multiprocessing,traceback
import numpy as np
def flow_worker(pipe,rank,cards):
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,cards)) and set(cards)<=set((4,5,6,7))
  import torch
  from ab_model_context import RAFT,module
  from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
  from src.utils.common import set_torch_env
  set_torch_env();torch.set_num_threads(1);torch.cuda.set_device(rank)
  assert sha(RAFT)==load(P/'protocol.json')['raft_sha256']
  raft=raft_large(weights=None,progress=False);raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
  raft=raft.cuda(rank).eval().requires_grad_(False);tf=Raft_Large_Weights.C_T_SKHT_V2.transforms()
  audit=module('v3_helper_frozen',P/'runtime_acceleration.py');initial=audit.frozen_hash(raft.state_dict().items())
  pipe.send(dict(status='READY',frozen_hash=initial,physical_GPU=cards[rank],GPU_mapping=gpu_mapping(rank,cards)))
  while True:
   row=pipe.recv()
   if row is None:break
   a=raw(row);flows=[]
   with torch.inference_mode():
    prev=a[48:49].to(rank).float()/255
    for pos in range(49,64):
     cur=a[pos:pos+1].to(rank).float()/255;ga,gb=tf(prev,cur);flows.append(raft(ga,gb)[-1].cpu().numpy().copy());prev=cur
   array=np.stack(flows);assert array.shape==(15,1,2,256,256) and np.isfinite(array).all() and audit.frozen_hash(raft.state_dict().items())==initial
   pipe.send(dict(status='PASS',sample=digest(row),flows=array,sha256=hashlib.sha256(array.tobytes()).hexdigest(),frozen_hash=initial))
  audit.full_frozen_hash_audit();pipe.close()
 except Exception:
  try:pipe.send(dict(status='FAILED',traceback=traceback.format_exc()))
  finally:traceback.print_exc();sys.exit(1)
class Pipeline:
 def __init__(self,sequence,rank,cards):
  self.sequence=sequence;self.rank=rank;self.cards=cards;self.cursor=0;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe();self.pipe=parent;self.process=ctx.Process(target=flow_worker,args=(child,rank,cards),daemon=True);self.process.start();child.close()
  self.ready=self.pipe.recv();assert self.ready['status']=='READY' and self.ready['physical_GPU']==cards[rank],self.ready;self.submit()
 def submit(self):
  self.future=None
  if self.cursor<len(self.sequence):
   row=self.sequence[self.cursor];self.pipe.send(row);self.future=self.pool.submit(raw,row)
 def next(self,row):
  import torch
  assert digest(row)==digest(self.sequence[self.cursor]);data=self.future.result();r=self.pipe.recv();assert r['status']=='PASS' and r['sample']==digest(row),r
  assert hashlib.sha256(r['flows'].tobytes()).hexdigest()==r['sha256'] and r['frozen_hash']==self.ready['frozen_hash']
  self.last_target_audit=dict(target_flow_sha256=r['sha256'],frozen_RAFT_state_hash=r['frozen_hash'],auxiliary_PID=self.process.pid,physical_GPU=self.cards[self.rank],GPU_mapping=self.ready['GPU_mapping'],sample_sha256=r['sample'])
  self.cursor+=1;self.submit();return Frames(data,torch.from_numpy(r['flows']),torch.device('cuda',self.rank))
 def close(self):
  if self.future is not None:
   self.future.result();pending=self.pipe.recv();assert pending['status']=='PASS',pending
  self.pipe.send(None);self.process.join(timeout=60);assert self.process.exitcode==0;self.pool.shutdown(wait=True)
