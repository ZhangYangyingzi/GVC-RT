"""One frozen target-flow GPU process per training rank, fixed-plan lookahead."""
import hashlib,multiprocessing,os,sys,traceback
import numpy as np
import torch
from common_expanded import ROOT,REPO,RAFT,SEED,load,sha
import runtime_acceleration as acceleration

def worker(connection,rank):
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
  torch.cuda.set_device(rank);device=torch.device('cuda',rank)
  sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'))
  import frames32
  from src.utils.common import set_torch_env
  set_torch_env();torch.set_num_threads(1)
  for domain in ('uvg','ulong'):
   frames32._SOURCES[domain]={v['filename' if domain=='ulong' else 'name']:v for v in load(ROOT/f'plans/train_manifest_{domain}.local.json')['videos']}
  raw_class=acceleration.build_raw_reader(frames32)
  from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
  raft=raft_large(weights=None,progress=False)
  assert sha(RAFT)==load(ROOT/'protocol.json')['raft_sha256']
  raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
  raft=raft.to(device).eval().requires_grad_(False);transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
  frozen=acceleration.frozen_hash(raft.state_dict().items())
  connection.send(dict(status='READY',physical_GPU=rank+4,model_hash=frozen))
  while True:
   row=connection.recv()
   if row is None:break
   frames=raw_class(row,torch.device('cpu'));flows=[]
   try:
    with torch.inference_mode():
     previous=frames[48].to(device).float()/255
     for pos in range(49,64):
      current=frames[pos].to(device).float()/255
      first,second=transform(previous,current)
      flow=raft(first,second)[-1].cpu().numpy().copy()
      flows.append(flow);previous=current
   finally:frames.close()
   assert acceleration.frozen_hash(raft.state_dict().items())==frozen
   array=np.stack(flows)
   assert array.shape==(15,1,2,256,256) and np.isfinite(array).all()
   connection.send(dict(status='PASS',identity=acceleration.identity(row),flows=array,
                        flow_sha256=hashlib.sha256(array.tobytes()).hexdigest(),frozen_model_hash=frozen))
  acceleration.full_frozen_hash_audit();connection.close()
 except Exception:
  try:connection.send(dict(status='FAILED',traceback=traceback.format_exc()))
  finally:traceback.print_exc();sys.exit(1)

class FlowService:
 def __init__(self,rank):
  ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe()
  self.connection=parent;self.process=ctx.Process(target=worker,args=(child,rank),daemon=False)
  self.process.start();child.close()
  self.ready=self.connection.recv();assert self.ready['status']=='READY',self.ready
 def submit(self,row):self.connection.send(row)
 def receive(self,row):
  result=self.connection.recv();assert result['status']=='PASS',result
  assert result['identity']==acceleration.identity(row)
  assert hashlib.sha256(result['flows'].tobytes()).hexdigest()==result['flow_sha256']
  return torch.from_numpy(result['flows'])
 def close(self):
  self.connection.send(None);self.process.join(timeout=60)
  assert self.process.exitcode==0,'Frozen target-flow process failed to close'
  self.connection.close()

class ParallelPrefetcher(acceleration.Prefetcher):
 def __init__(self,schedule,rows,rank,start,raw_class):
  self.flow_service=FlowService(rank);self.current_flows=None
  super().__init__(schedule,rows,rank,start,raw_class)
 def submit(self):
  if self.cursor<len(self.schedule):self.flow_service.submit(self.row())
  return super().submit()
 def next(self,row):
  assert acceleration.identity(row)==acceleration.identity(self.row())
  data=self.future.result();self.current_flows=self.flow_service.receive(row)
  self.cursor+=1;self.future=self.submit()
  return data

class ParallelFrames(acceleration.PrefetchedFrames):
 def __init__(self,row,device,full=False):
  super().__init__(row,device,full)
  self.target_flows=acceleration._prefetch.current_flows.to(device)
 def close(self):
  super().close();self.target_flows=None

def configure(frames_module,schedule,rows,rank,start):
 raw=acceleration.build_raw_reader(frames_module)
 acceleration._prefetch=ParallelPrefetcher(schedule,rows,rank,start,raw)
 frames_module.PlanFrames=ParallelFrames

def loss_function(original):
 import inspect
 source=inspect.getsource(original)
 old='with torch.no_grad(): target_flow=raft(ga,gb)[-1]'
 assert source.count(old)==1
 source=source.replace(old,'with torch.no_grad(): target_flow=frames.target_flows[pos-49]')
 namespace=dict(original.__globals__)
 exec(compile(source,str(ROOT/'parallel_flow_runtime.py'),'exec'),namespace)
 return namespace['basic_crop']

def shutdown():
 if acceleration._prefetch is not None:acceleration._prefetch.flow_service.close()
 acceleration.shutdown()
