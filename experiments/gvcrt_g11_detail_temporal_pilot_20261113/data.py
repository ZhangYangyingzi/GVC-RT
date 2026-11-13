"""Frozen parent streams/Original references; task-local bounded prefix and GT-flow cache."""
from util import *
import torch
import torch.nn.functional as F
from parent_data import OriginalData,load_raft
from model import load_cref,Receiver
from alignment import valid_flow

class Source(OriginalData):
 def __init__(self,row):
  super().__init__(row);self._done=load(self.dest/'completion.json') if self.data is None else None
  if self.data is None:assert sha(self.dest/'info.json')==self._done['info_sha256']
  ready=load(PARENT/'cache/metadata'/f"{row['sequence']}_ready.json");path=PARENT/'cache/metadata'/f"{row['sequence']}.pt";assert sha(path)==ready['metadata_sha256'];self.meta=torch.load(path,map_location='cpu',weights_only=True,mmap=True);self._verified=set()
 def verify(self,p):
  p=Path(p)
  if str(p) not in self._verified:
   expected=self._done['files'][p.name];assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'];self._verified.add(str(p))
 def feature(self,i):
  if self.data is None:self.verify(self.dest/f'f_{i:02d}.pt')
  return super().feature(i)
 def aux(self,i):
  if self.data is None:self.verify(self.dest/f'aux_{i:02d}.pt')
  return super().aux(i)

def cache_dir(seq):return ROOT/'cache/frozen'/seq
class Data:
 def __init__(self,c):self.c=c;self.receiver=Receiver(c,'stage3');self.sources={};self.frames={}
 def source(self,seq):
  if seq not in self.sources:self.sources[seq]=Source(next(r for r in videos() if r['sequence']==seq))
  return self.sources[seq]
 def tensors(self,seq,i):
  p=cache_dir(seq)/f'frame_{i:02d}.pt';wait_file(p.with_suffix('.pt.json'),'WAITING_FOR_CACHE');key=(seq,i)
  if key not in self.frames:
   if len(self.frames)>18:self.frames.clear()
   self.frames[key]=read(p)
  d=self.frames[key];m={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['meta'].items()};return {k:v.cuda() if torch.is_tensor(v) else v for k,v in d.items() if k!='meta'}|dict(meta=m)
 def gt(self,seq,i):
  a=self.source(seq).aux(i)['GT'];return a.permute(2,0,1)[None].cuda().float()/255
 def supervision(self,seq,i):
  p=cache_dir(seq)/f'GT_flow_{i:02d}.pt';wait_file(p.with_suffix('.pt.json'),'WAITING_FOR_GT_FLOW');d=read(p);return d['backward'].cuda(),d['valid'].cuda()

@torch.no_grad()
def produce(row,c,raft,transform):
 seq=row['sequence'];dest=cache_dir(seq);dest.mkdir(exist_ok=True,parents=True)
 if (dest/'completion.json').exists():return
 src=Source(row);g=c['models']['generator'];receiver=Receiver(c,'stage3');g.reset_temporal_buffer();g.set_temporal_cascade_grad(False);previous_c=None;begin=time.perf_counter();identity=[]
 for i in range(1,64):
  heartbeat('PRODUCING_FROZEN_PREFIX',video=seq,frame=i)
  f=src.feature(i).cuda().float();q=c['pm'].q_scale_recon[src.info['actual_qps'][i]:src.info['actual_qps'][i]+1].float();meta=src.meta.get(i)
  if meta is not None and meta['scene_cut']:g.reset_temporal_buffer()
  h,x,q=receiver.prefix(f,q)
  if i>=46:
   enc=c['S'].E(f)
   if i>=47:
    m={k:v.cuda() if torch.is_tensor(v) else v for k,v in meta.items()};hs,_=c['S'](f,f,h,m,condition='S');p=dest/f'frame_{i:02d}.pt'
    if not p.with_suffix('.pt.json').exists():save(p,dict(h_s=hs.cpu(),current_c=enc.cpu(),previous_c=previous_c.cpu(),h_raw=h.cpu(),x=x.cpu(),q=q.cpu(),meta=meta,codec_feature_sha256=thash([('f',f.half())]),actual_QP=src.info['actual_qps'][i],frame=i))
    if i in [47,49,63]:identity.append(dict(frame=i,feature_sha256=thash([('f',f.half())]),S_output_sha256=thash([('S',hs)])))
   previous_c=enc
 g.reset_temporal_buffer();prefix_seconds=time.perf_counter()-begin;t=time.perf_counter();native_h,native_w=src.info['native_shape']
 for i in range(50,64):
  p=dest/f'GT_flow_{i:02d}.pt'
  if p.with_suffix('.pt.json').exists():continue
  heartbeat('PRODUCING_GT_SUPERVISION',video=seq,frame=i)
  a=src.aux(i)['GT'].permute(2,0,1)[None].cuda().float()/255;b=src.aux(i-1)['GT'].permute(2,0,1)[None].cuda().float()/255;a,b=transform(a,b);back=raft(a,b,num_flow_updates=12)[-1];fw=raft(b,a,num_flow_updates=12)[-1];valid=valid_flow(back,fw);assert back.shape[-2:]==(native_h,native_w);assert torch.isfinite(back).all();save(p,dict(backward=back.cpu(),valid=valid.cpu(),GT_supervision_only=True,RAFT_iterations=12,source_sha256=row['source_sha256']))
 dump(dest/'completion.json',dict(status='PASS',sequence=seq,source_sha256=row['source_sha256'],stream_sha256=src.info['stream_SHA256'],stream_path=src.info['stream_path'],real_bytes=src.info['real_bytes'],bpp=src.info['bpp'],codec_trace=src.info['codec_trace'],parent_metadata_sha256=sha(PARENT/'cache/metadata'/f'{seq}.pt'),prefix_seconds=prefix_seconds,GT_flow_seconds=time.perf_counter()-t,actual_QPs=src.info['actual_qps'],identities=identity,C_ref_branch_sha256=load(ROOT/'weight_manifest.json')['D12_S_900']['sha256'],files={p.name:load(p.with_suffix(p.suffix+'.json')) for p in dest.glob('*.pt')}))
 event(ROOT/'logs/cache.jsonl',stage='CACHE_COMMITTED',video=seq,prefix_seconds=prefix_seconds,GT_flow_seconds=time.perf_counter()-t)

if __name__=='__main__':
 with lock('cache_producer.lock'):
  setup_torch();c=load_cref();raft,transform=load_raft();cal=load(ROOT/'sample_schedule.json')['calibration'];ids=list(dict.fromkeys([x['video'] for x in cal]+[x['sequence'] for x in videos('validation')]+[x['video'] for x in load(ROOT/'sample_schedule.json')['updates']]));assert len(ids)==18
  for seq in ids:produce(next(r for r in videos() if r['sequence']==seq),c,raft,transform)
  dump(ROOT/'receipts/cache.complete.json',dict(status='COMPLETE',videos=18,finite_bounded_task=True,pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES']))
