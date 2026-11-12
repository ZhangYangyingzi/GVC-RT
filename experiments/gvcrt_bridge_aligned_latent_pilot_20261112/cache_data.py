"""Independent Original codec cache. RGB is cropped before codec for training."""
from util import *
import argparse,io,types,gc

def load_raft():
 import torch
 from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
 from common_s2 import RAFT,RAFT_SHA
 assert sha(RAFT)==RAFT_SHA
 m=raft_large(weights=None,progress=False);m.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
 return m.cuda().eval().requires_grad_(False),Raft_Large_Weights.C_T_SKHT_V2.transforms()
def fp32_original(pm):
 g=pm.recon_generation_net;g.float()
 for m in g.modules():
  if m.__class__.__name__ in ['DepthConvBlock','StageBlock']:
   m.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),m)
   if hasattr(m,'proxy'):m.proxy=None
 # Original generator's entire path uses PyTorch as B300 does.
 dec=g.decoder
 dec.forward=types.MethodType(lambda self,*a,**k:self.forward_torch(*a,**k),dec) if hasattr(dec,'forward_torch') else dec.forward
 original=g.forward
 g.forward=types.MethodType(lambda self,f,q:original(f.float(),q.float()),g)

def make(task,rows):
 torch=setup();import numpy as np,cv2
 from original_context import runtime
 from original_runtime import load_models
 from temporal import flow_valid
 import torch.nn.functional as F
 rt=runtime(0);im,pm=load_models(rt);fp32_original(pm);raft,transform=load_raft();sh=rt['stream']
 before=thash(im.state_dict().items())+thash(pm.state_dict().items());verified=set()
 for row in rows:
  qp=row['external_qp'];dest=cachepath(row['id'],qp)
  if dest.with_suffix('.pt.json').exists():readtensor(dest);continue
  check_protocol();heartbeat(task,'CACHE_SOURCE',sample=row['id'],qp=qp)
  t0=time.time()
  if row['source_path'] not in verified:
   assert sha(row['source_path'])==row['source_sha256'];verified.add(row['source_path'])
  arr=[]
  if row.get('pixel_format')=='yuv420p':
   assert row['start']==0
   command=['ffmpeg','-v','error','-threads','2','-f','rawvideo','-pixel_format','yuv420p','-video_size',f"{row['width']}x{row['height']}",'-framerate',row['source_fps'],'-i',row['source_path'],'-frames:v',str(row['frames']),'-vf','scale=in_range=tv:out_range=pc:in_color_matrix=bt709','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
   raw=subprocess.run(command,capture_output=True,check=True).stdout;arr=list(np.frombuffer(raw,dtype=np.uint8).reshape(row['frames'],row['height'],row['width'],3).copy())
  else:
   cap=cv2.VideoCapture(row['source_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,row['start'])
   for i in range(row['frames']):
    ok,a=cap.read();assert ok,('SOURCE_SHORT',row['id'],i);a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    if row['crop'] is not None:
     x,y,w,h=row['crop'];a=a[y:y+h,x:x+w]
    arr.append(a.copy())
   cap.release()
  h,w=arr[0].shape[:2];ph,pw=(h+63)//64*64,(w+63)//64*64
  gt=torch.from_numpy(np.stack(arr));payloads=[];fs=[None];qps=[];trace=[];original=[];pm.clear_dpb();pm.set_curr_poc(0)
  with torch.no_grad():
   for i in range(row['frames']):
    x=gt[i].permute(2,0,1)[None].cuda().float()/255;x=rt['replicate_pad'](x.half(),ph-h,pw-w)*2-1
    q=qp if i==0 else pm.shift_qp(qp,(0,1,0,2,0,2,0,2)[i%8]);qps.append(q)
    enc=im.compress(x,q) if i==0 else pm.compress(x,q);payloads.append(enc['bit_stream'])
    if i==0:pm.add_ref_frame(None,enc['x_hat'])
    trace.append(None if i==0 else thash([('f',pm.dpb[0].feature)]))
   encode_sec=time.time()-t0;pm.clear_dpb();pm.set_curr_poc(0);tdecode=time.time();stream=io.BytesIO();sps=dict(sps_id=0,height=ph,width=pw,ec_part=1,use_ada_i=0);sh.write_sps(stream,sps);bits=[]
   for i,payload in enumerate(payloads):
    a=stream.tell();sh.write_ip(stream,i==0,0,qps[i],payload);bits.append((stream.tell()-a)*8)
    dec=im.decompress(payload,sps,qps[i]) if i==0 else pm.decompress(payload,sps,qps[i])
    if i==0:pm.add_ref_frame(None,dec['x_hat'])
    else:
     f=pm.dpb[0].feature;assert thash([('f',f)])==trace[i],'CODEC_ENCODE_DECODE_FEATURE_MISMATCH';fs.append(f.cpu().clone())
    if row['split']=='dev' or i==0:original.append(((dec['x_hat'].float().clamp(-1,1)+1)/2)[...,:h,:w].cpu())
   decode_sec=time.time()-tdecode
   raw=stream.getvalue();sp=dest.with_suffix('.bin');tmp=sp.with_suffix('.bin.tmp');tmp.write_bytes(raw);tmp.replace(sp)
   # All supervision comes only from GT current/previous. Padded teacher region is excluded.
   tflow=time.time();flows=[torch.zeros(1,2,h,w)];valid=[torch.zeros(1,1,h,w,dtype=torch.bool)]
   for i in range(1,row['frames']):
    heartbeat(task,'CACHE_TEACHER',sample=row['id'],frame=i,qp=qp)
    cur=gt[i].permute(2,0,1)[None].cuda().float()/255;old=gt[i-1].permute(2,0,1)[None].cuda().float()/255
    cur=F.pad(cur,(0,pw-w,0,ph-h),mode='replicate');old=F.pad(old,(0,pw-w,0,ph-h),mode='replicate')
    a,b=transform(cur,old);back=raft(a,b,num_flow_updates=12)[-1];fw=raft(b,a,num_flow_updates=12)[-1];mask=flow_valid(back,fw)
    from model import bounds,grid
    mask=mask & bounds(grid(ph,pw,'cuda')+back,h,w)
    flows.append(back[...,:h,:w].cpu());valid.append(mask[...,:h,:w].cpu())
   assert before==thash(im.state_dict().items())+thash(pm.state_dict().items()),'FROZEN_CODEC_CHANGED'
  rec=dict(id=row['id'],row=row,GT=gt,features=fs,flow=flows,valid=valid,actual_qps=qps,original=original,native_shape=[h,w],padded_shape=[ph,pw],codec_feature_hashes=trace,stream_sha256=sha(sp),stream_path=str(sp),bits=bits,bpp=len(raw)*8/(row['frames']*h*w),encode_seconds=encode_sec,decode_seconds=decode_sec,teacher_seconds=time.time()-tflow,Original_FP32=row['split']=='dev',frozen_sha256=before,teacher='GT_RAFT_pseudo_label_supervision_only')
  save(dest,rec);event('CACHE_COMMITTED',id=row['id'],qp=qp,seconds=time.time()-t0,bytes=dest.stat().st_size,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  del rec,fs,flows,valid,gt,original;gc.collect();torch.cuda.empty_cache()
 heartbeat(task,'COMPLETE')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);a.add_argument('--dev',action='store_true');a.add_argument('--quick',action='store_true');a.add_argument('--extra',action='store_true');p=a.parse_args();m=manifest();m['dev']=m['dev']+(load(ROOT/'additional_development_manifest.json')['dev'] if (ROOT/'additional_development_manifest.json').exists() and not p.quick else [])
 if p.extra:m['dev']=load(ROOT/'additional_development_manifest.json')['dev']
 rows=[dict(r,external_qp=q) for r in m['dev'] for q in ([5] if p.quick else [5,0,9])] if p.dev else m['train']
 task=('extracache' if p.extra else 'devcache' if p.dev else 'traincache')+str(p.shard)
 with lock(task+'.lock'):make(task,rows if p.dev else rows[p.shard::3])
