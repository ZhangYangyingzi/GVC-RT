"""Audited uint8/crop reader and frozen auxiliary GT-RAFT prefetch."""
from common_s2 import *
from data_reader import read_arrays
import concurrent.futures,multiprocessing,traceback
import numpy as np

def raw(row):
    import torch
    return torch.from_numpy(np.stack(read_arrays(row))).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)

class Frames:
    def __init__(self,data,device,target_flows=None):
        self.frames=data.pin_memory();self.device=device
        self.target_flows=None if target_flows is None else target_flows.to(device)
    def __len__(self):return 64
    def __getitem__(self,pos):
        if pos>=64:raise IndexError(pos)
        a=self.frames[pos:pos+1]
        a=a.as_strided(a.shape,(3,1,768,3))
        return a.to(self.device,non_blocking=True).float()/255
    def close(self):self.frames=None;self.target_flows=None

def flow_worker(pipe,rank):
    try:
        env()
        assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
        import torch
        sys.path.insert(0,str(ROOT/'runtime_tree'))
        from src.utils.common import set_torch_env
        set_torch_env();torch.set_num_threads(1);torch.cuda.set_device(rank)
        from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
        assert sha(RAFT)==RAFT_SHA
        raft=raft_large(weights=None,progress=False)
        raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
        raft=raft.cuda(rank).eval().requires_grad_(False)
        transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
        guard=FrozenGuard(raft.state_dict().items())
        pipe.send(dict(status='READY',frozen_hash=guard.digest,physical_GPU=rank+4))
        while True:
            row=pipe.recv()
            if row is None:break
            a=raw(row);flows=[]
            with torch.inference_mode():
                previous=a[48:49].to(rank).float()/255
                for pos in range(49,64):
                    current=a[pos:pos+1].to(rank).float()/255
                    x,y=transform(previous,current)
                    flows.append(raft(x,y)[-1].cpu().numpy().copy());previous=current
            result=np.stack(flows)
            assert result.shape==(15,1,2,256,256) and np.isfinite(result).all()
            guard.check()
            pipe.send(dict(status='PASS',sample=digest(row),flows=result,sha256=hashlib.sha256(result.tobytes()).hexdigest(),frozen_hash=guard.digest))
        guard.full();pipe.close()
    except Exception:
        try:pipe.send(dict(status='FAILED',error=traceback.format_exc()))
        finally:traceback.print_exc();raise

class Pipeline:
    def __init__(self,sequence,rank):
        self.sequence=sequence;self.rank=rank;self.cursor=0
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        parent,child=multiprocessing.get_context('spawn').Pipe()
        self.pipe=parent
        self.process=multiprocessing.get_context('spawn').Process(target=flow_worker,args=(child,rank),daemon=True)
        self.process.start();child.close()
        self.ready=self.pipe.recv();assert self.ready['status']=='READY',self.ready
        self.submit()
    def submit(self):
        self.future=None
        if self.cursor<len(self.sequence):
            row=self.sequence[self.cursor];self.pipe.send(row);self.future=self.pool.submit(raw,row)
    def next(self,row):
        import torch
        assert digest(row)==digest(self.sequence[self.cursor])
        data=self.future.result();result=self.pipe.recv();assert result['status']=='PASS',result
        assert result['sample']==digest(row) and hashlib.sha256(result['flows'].tobytes()).hexdigest()==result['sha256']
        assert result['frozen_hash']==self.ready['frozen_hash']
        self.last_target_audit=dict(target_flow_sha256=result['sha256'],RAFT_frozen_hash=result['frozen_hash'],auxiliary_PID=self.process.pid,physical_GPU=4+self.rank)
        self.cursor+=1;self.submit()
        return Frames(data,torch.device('cuda',self.rank),torch.from_numpy(result['flows']))
    def close(self):
        if self.future is not None:
            self.future.result();r=self.pipe.recv();assert r['status']=='PASS',r
        self.pipe.send(None);self.process.join(timeout=60);assert self.process.exitcode==0
        self.pool.shutdown(wait=True)

def optimized_loss(original):
    import inspect
    source=inspect.getsource(original)
    needle='with torch.no_grad(): target_flow=raft(ga,gb)[-1]'
    assert source.count(needle)==1
    source=source.replace(needle,'with torch.no_grad(): target_flow=frames.target_flows[pos-49]')
    namespace=dict(original.__globals__)
    exec(compile(source,str(ROOT/'pipeline.py'),'exec'),namespace)
    return namespace['basic_crop']
