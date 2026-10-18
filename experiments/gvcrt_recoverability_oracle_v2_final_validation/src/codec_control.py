"""Signalled activation control only; all native codec kernels remain unchanged."""
from base import *
import struct,types,gc,inspect,textwrap
MAGIC=b'GROV1\x00\x00\x00'
LEVELS=(1.,.5,2.)
def pack(plan,native):
 out=MAGIC+struct.pack('>H',len(plan))
 for p in plan:out+=struct.pack('>5HB',p['frame'],p['y'],p['x'],p['h'],p['w'],LEVELS.index(p['factor']))
 return out+native
def unpack(data):
 assert data[:8]==MAGIC
 count=struct.unpack('>H',data[8:10])[0];end=10+11*count;assert end<len(data)
 plan=[]
 for i in range(count):
  f,y,x,h,w,level=struct.unpack('>5HB',data[10+11*i:21+11*i]);assert level<len(LEVELS)
  plan.append(dict(frame=f,y=y,x=x,h=h,w=w,factor=LEVELS[level]))
 return plan,data[end:],end

def runtime(job,artifact_dir):
 import torch,numpy as np
 e=module('recoverability_engine',ENGINE)
 sys.path.insert(0,str(V32));from training32 import official_model
 cfg=dict(experiment='B',force_zero_thres=.12,methods={'original':[None,None]},checkpoints={})
 expected=load(V32/'evaluation_config.json')
 def observe(pm,rt,role):
  state=dict(groups=[]);orig_prior=pm.res_prior_param_decoder;orig_process=pm.process_with_mask;orig_quant=pm.compress_prior_2x;orig_get=pm.gaussian_encoder.get_y
  def selected():return int(pm.curr_poc) in (2,8)
  def params(_self,z,ctx):
   p=orig_prior(z,ctx);f=int(pm.curr_poc)
   if selected():state['original_q']=p[:,:128].clamp_min(.5).detach().cpu().numpy().copy()
   edits=[a for a in rt.active_plan if a['frame']==f and a['factor']!=1.]
   if edits:
    p=p.clone();q=p[:,:128]
    for a in edits:
     y,x,h,w=a['y'],a['x'],a['h'],a['w'];assert 0<=y<y+h<=q.shape[2] and 0<=x<x+w<=q.shape[3]
     raw=q[:,:,y:y+h,x:x+w];base=raw.clamp_min(.5);changed=(base*a['factor']).clamp_min(.5)
     q[:,:,y:y+h,x:x+w]=torch.where(changed==base,raw,changed)
    p[:,:128]=q
   if selected():state['effective_q']=p[:,:128].clamp_min(.5).detach().cpu().numpy().copy()
   return p
  def process(_self,y,s,m,mask):
   out=orig_process(y,s,m,mask)
   if selected():state['groups'].append(out[1].detach().cpu().numpy().astype(np.int8))
   return out
  def quant(_self,y,p,spatial):
   raw=y.detach().cpu().numpy().copy() if selected() else None
   state['groups']=[];out=orig_quant(y,p,spatial)
   if selected():
    f=int(pm.curr_poc);assert len(state['groups'])==2
    rec=dict(y=raw,original_q=state['original_q'],effective_q=state['effective_q'],symbols=(state['groups'][0].astype(np.int16)+state['groups'][1]).astype(np.int8),packed_0=out[0].detach().cpu().numpy().astype(np.int8),packed_1=out[1].detach().cpu().numpy().astype(np.int8),y_hat=out[-1].detach().cpu().numpy().copy())
    rt.captures[f]=rec
   return out
  def get_y(*args,**kwargs):
   out=orig_get(*args,**kwargs)
   if selected():
    f=int(pm.curr_poc);group=rt.decoded_groups.setdefault((role,f),[])
    actual=out.detach().cpu().numpy().astype(np.int8);wanted=rt.captures[f]['packed_'+str(len(group))]
    assert np.array_equal(actual,wanted),(role,f,len(group),'entropy decoded symbols differ')
    group.append(dict(shape=list(actual.shape),sha256=hashlib.sha256(actual.tobytes()).hexdigest(),exact=True))
   return out
  pm.res_prior_param_decoder=types.MethodType(params,pm)
  if role==0:
   pm.process_with_mask=types.MethodType(process,pm);pm.compress_prior_2x=types.MethodType(quant,pm)
  else:pm.gaussian_encoder.get_y=get_y
 class Runtime(e.Runtime):
  def __init__(self):
   super().__init__(cfg,'original',torch.device('cuda:0'))
   self.metric_models=(*self.metric_models[:2],official_model())
   self.active_plan=job['plan'];self.captures={};self.decoded_groups={};self.calls=0;self.timings=[];self.weight_checks=[]
  def models(self,h,w):
   im,pm=super().models(h,w);role=self.calls;self.calls+=1;assert role in (0,1,2)
   assert e.core.compression_hash(im,pm)==expected['compression_hash'] and e.core.module_hash(im)==expected['I_hash']
   assert self.loaded_receiver_hashes==expected['original_receiver_hashes']
   before=[e.core.module_hash(m) for m in (im,pm)];observe(pm,self,role)
   for model,kind in ((im,'I'),(pm,'P')):
    name='compress' if role==0 else 'decompress';original=getattr(model,name)
    def timed(*a,_fn=original,_role=role,_kind=kind,_name=name,**kw):
     torch.cuda.synchronize();start=time.perf_counter();out=_fn(*a,**kw);torch.cuda.synchronize()
     self.timings.append(dict(role=_role,kind=_kind,operation=_name,ms=(time.perf_counter()-start)*1000))
     return out
    setattr(model,name,timed)
   # Before model deletion, full state is checked on the last P call.
   original=pm.compress if role==0 else pm.decompress
   def last(*a,**kw):
    out=original(*a,**kw)
    if pm.curr_poc==16:
     after=[e.core.module_hash(m) for m in (im,pm)];assert after==before
     self.weight_checks.append(dict(role=role,full_module_hashes=after,unchanged=True))
    return out
   setattr(pm,'compress' if role==0 else 'decompress',last)
   return im,pm
  def parse_persisted(self,data):
   plan,native,n=unpack(data);self.active_plan=plan;self.decoded_header_bytes=n
   assert plan==job['plan'],'Independent decoder control plan must come from persisted stream'
   return native
 rt=Runtime()
 source=textwrap.dedent(inspect.getsource(e.Runtime.run))
 changes={
  'data=buffer.getvalue();stream_path.parent.mkdir':'native_data=buffer.getvalue();data=pack(self.active_plan,native_data);stream_path.parent.mkdir',
  'stream=io.BytesIO(stream_path.read_bytes());helper=eco.SPSHelper()':'stream=io.BytesIO(self.parse_persisted(stream_path.read_bytes()));helper=eco.SPSHelper()',
  'assert stream.tell()==len(data) and len(hashes)==n':'assert stream.tell()==len(native_data) and len(hashes)==n',
  'assert sum(r[\'real_bits\'] for r in rows)==len(data)*8':"rows[0]['real_bits']+=self.decoded_header_bytes*8\n    assert sum(r['real_bits'] for r in rows)==len(data)*8",
  'bytes_consumed=stream.tell()':'bytes_consumed=stream.tell()+self.decoded_header_bytes',
  'del ie,pe,idc,pdc':'del ie,pe,idc,pdc\n    gc.collect()',
  'del ia,pa':'del ia,pa\n    gc.collect()',
 }
 for a,b in changes.items():assert source.count(a)==1,a;source=source.replace(a,b)
 ns=dict(e.__dict__);ns.update(pack=pack,gc=gc);exec(compile(source,str(ROOT/'src/codec_control.py'),'exec'),ns)
 rt.run=types.MethodType(ns['run'],rt)
 return e,rt
