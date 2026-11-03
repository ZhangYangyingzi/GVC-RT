"""Parent-audited prefetch/GT-RAFT worker semantics with a native ZIP-capable reader."""
from ab_common import *
from ab_data_reader import read_arrays
import concurrent.futures,multiprocessing,traceback
import numpy as np
def raw(row):
 import torch
 # Identical uint8 RGB -> NCHW channels-last, H2D then float()/255 ordering.
 return torch.from_numpy(np.stack(read_arrays(row))).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
def flow_worker(pipe,rank):
 try:
  import torch
  from ab_model_context import RAFT
  from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
  from src.utils.common import set_torch_env
  set_torch_env();torch.set_num_threads(1);torch.cuda.set_device(rank)
  assert sha(RAFT)==load(P/'protocol.json')['raft_sha256']
  raft=raft_large(weights=None,progress=False);raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
  raft=raft.cuda(rank).eval().requires_grad_(False);tf=Raft_Large_Weights.C_T_SKHT_V2.transforms()
  audit=__import__('ab_model_context').module('helper_frozen_audit',P/'runtime_acceleration.py');initial=audit.frozen_hash(raft.state_dict().items())
  pipe.send(dict(status='READY',frozen_hash=initial,physical_GPU=rank+4))
  while True:
   row=pipe.recv()
   if row is None:break
   a=raw(row);flows=[]
   with torch.inference_mode():
    prev=a[48:49].to(rank).float()/255
    for pos in range(49,64):
     cur=a[pos:pos+1].to(rank).float()/255;ga,gb=tf(prev,cur);flows.append(raft(ga,gb)[-1].cpu().numpy().copy());prev=cur
   array=np.stack(flows);assert np.isfinite(array).all() and audit.frozen_hash(raft.state_dict().items())==initial
   pipe.send(dict(status='PASS',sample=digest(row),flows=array,sha256=hashlib.sha256(array.tobytes()).hexdigest(),frozen_hash=initial))
  audit.full_frozen_hash_audit();pipe.close()
 except Exception:
  try:pipe.send(dict(status='FAILED',traceback=traceback.format_exc()))
  finally:traceback.print_exc();sys.exit(1)
class Frames:
 def __init__(self,data,flows,device):self.frames=data.pin_memory();self.target_flows=flows.to(device);self.device=device
 def __len__(self):return 64
 def __getitem__(self,pos):
  if pos>=64:raise IndexError(pos)
  frame=self.frames[pos:pos+1];a=frame.as_strided(frame.shape,(3,1,768,3));return a.to(self.device,non_blocking=True).float()/255
 def close(self):self.frames=None;self.target_flows=None
class Pipeline:
 def __init__(self,rows,rank,start):
  self.rows=rows;self.rank=rank;self.cursor=start;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe();self.pipe=parent;self.process=ctx.Process(target=flow_worker,args=(child,rank),daemon=True);self.process.start();child.close()
  self.ready=self.pipe.recv();assert self.ready['status']=='READY',self.ready;self.submit()
 def submit(self):
  self.future=None
  if self.cursor<len(self.rows)//4:
   r=self.rows[self.cursor*4+self.rank];self.pipe.send(r);self.future=self.pool.submit(raw,r)
 def next(self,row):
  import torch
  assert digest(row)==digest(self.rows[self.cursor*4+self.rank])
  data=self.future.result();r=self.pipe.recv();assert r['status']=='PASS' and r['sample']==digest(row),r
  assert hashlib.sha256(r['flows'].tobytes()).hexdigest()==r['sha256']
  self.cursor+=1;self.submit();return Frames(data,torch.from_numpy(r['flows']),torch.device('cuda',self.rank))
 def close(self):
  self.pipe.send(None);self.process.join(timeout=60);assert self.process.exitcode==0;self.pool.shutdown(wait=True)
def optimized_loss(original):
 import inspect
 code=inspect.getsource(original);needle='with torch.no_grad(): target_flow=raft(ga,gb)[-1]';assert code.count(needle)==1
 code=code.replace(needle,'with torch.no_grad(): target_flow=frames.target_flows[pos-49]')
 ns=dict(original.__globals__);exec(compile(code,str(ROOT/'ab_runtime.py'),'exec'),ns);return ns['basic_crop']
