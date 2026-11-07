"""Numerically neutral I/O prefetch and exact frozen-device-byte checks."""
import concurrent.futures,hashlib,inspect
import torch

_cache=None
_prefetch=None
_raw=None

def byte_hash(items):
 h=hashlib.sha256()
 for n,t in items:
  h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def frozen_hash(items):
 global _cache
 items=list(items)
 if _cache is None:
  views=[t.detach().contiguous().reshape(-1).view(torch.uint8) for _,t in items]
  # All views refer to frozen persistent state. Equality is byte-for-byte,
  # including floating signed zero and every applicable buffer.
  baseline=torch.cat(views).clone()
  _cache=(views,baseline,byte_hash(items),[(n,t) for n,t in items])
 else:
  views,baseline,digest,original=_cache
  assert [n for n,_ in items]==[n for n,_ in original]
  assert all(t.data_ptr()==old.data_ptr() for (_,t),(_,old) in zip(items,original))
  current=[t.detach().contiguous().reshape(-1).view(torch.uint8) for _,t in items]
  assert torch.equal(torch.cat(current),baseline),'Frozen state device bytes changed'
 return _cache[2]

def full_frozen_hash_audit():
 assert _cache is not None
 assert byte_hash(_cache[3])==_cache[2],'Frozen SHA256 changed'
 return _cache[2]

def build_raw_reader(frames_module):
 source=inspect.getsource(frames_module.PlanFrames)
 assert source.count('.to(self.device).float()/255')==1
 source=source.replace('class PlanFrames:','class RawFrames:').replace('.to(self.device).float()/255','')
 ns=dict(frames_module.__dict__)
 exec(compile(source,'<audited-uint8-prefetch-reader>','exec'),ns)
 return ns['RawFrames']

def load_uint8(row,raw_class):
 frames=raw_class(row,torch.device('cpu'))
 try:
  result=torch.cat([frames[i] for i in range(64)],dim=0).contiguous(memory_format=torch.channels_last)
  assert result.dtype==torch.uint8 and result.shape==(64,3,256,256)
  return result
 finally:frames.close()

def identity(row):
 return (row['video'],tuple(row['canonical_indices']),row['crop_x'],row['crop_y'],row['external_qp'])

class Prefetcher:
 def __init__(self,schedule,rows,rank,start,raw_class):
  self.schedule=schedule;self.rows=rows;self.rank=rank;self.cursor=start
  self.raw_class=raw_class;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  self.future=self.submit()
 def row(self):
  item=self.schedule[self.cursor]['ranks'][self.rank]
  return dict(self.rows[item['base_row']],external_qp=item['external_qp'])
 def submit(self):
  return None if self.cursor>=len(self.schedule) else self.pool.submit(load_uint8,self.row(),self.raw_class)
 def next(self,row):
  assert identity(row)==identity(self.row()),'Prefetch cursor differs from fixed plan'
  data=self.future.result();self.cursor+=1;self.future=self.submit()
  return data

class PrefetchedFrames:
 def __init__(self,row,device,full=False):
  assert not full and _prefetch is not None
  cpu=_prefetch.next(row).pin_memory()
  # Preserve the original order: uint8 H2D, then float32 conversion and division
  # on the GPU. CPU float division is deliberately never substituted.
  self.frames=cpu
  self.device=device
 def __len__(self):return 64
 def __getitem__(self,pos):
  if pos>=64:raise IndexError(pos)
  frame=self.frames[pos:pos+1]
  raw=frame.as_strided(frame.shape,(3,1,768,3))
  return raw.to(self.device,non_blocking=True).float()/255
 def close(self):self.frames=None

def configure(frames_module,schedule,rows,rank,start):
 global _prefetch,_raw
 _raw=build_raw_reader(frames_module)
 _prefetch=Prefetcher(schedule,rows,rank,start,_raw)
 frames_module.PlanFrames=PrefetchedFrames

def shutdown():
 if _prefetch is not None:_prefetch.pool.shutdown(wait=True)
