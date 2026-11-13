"""Native64-frame data/bitstream streaming, shared reference sidecars, GT-only metric supervision."""
from util import *
import numpy as np,torch,io,gc
import torch.nn.functional as F
from reference import decoded_flow
from model import resize_flow,grid,bounds
from temporal import flow_valid

def raw_frames(row):
 import cv2
 path=Path(row['source_path'])
 if not path.exists() and (ROOT/'source_path_migrations.json').exists():
  matches=[m for m in load(ROOT/'source_path_migrations.json')['migrations'] if m['old_path']==str(path) and m['sha256']==row['source_sha256']];assert len(matches)==1,'SOURCE_MIGRATION_NOT_VERIFIED';path=Path(matches[0]['path'])
 key=(str(path),path.stat().st_size,path.stat().st_mtime_ns,row['source_sha256'])
 if key not in _verified_sources:assert sha(path)==row['source_sha256'];_verified_sources.add(key)
 if row.get('pixel_format')=='yuv420p':
  cmd=['ffmpeg','-v','error','-threads','2','-f','rawvideo','-pixel_format','yuv420p','-video_size',f"{row['width']}x{row['height']}",'-framerate',row['source_fps'],'-i',str(path),'-frames:v','64','-vf','scale=in_range=tv:out_range=pc:in_color_matrix=bt709','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];raw=subprocess.run(cmd,capture_output=True,check=True).stdout;return np.frombuffer(raw,np.uint8).reshape(64,row['height'],row['width'],3).copy()
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,row['start']);out=[]
 for i in range(64):ok,a=cap.read();assert ok,('SOURCE_SHORT',row['id'],i);out.append(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
 cap.release();return np.stack(out)
_verified_sources=set()

def stream_file(rt,im,pm,row,qp,arrays):
 p=ROOT/'cache/streams'/f"{row['id']}_q{qp}.bin";meta=p.with_suffix('.bin.json')
 with lock(f'stream_{row["id"]}_q{qp}.lock',True):
  if meta.exists():verify_file(p);return p,load(meta)['meta']
  # Reuse only exact64-frame bitstreams with proven frame/source/precision identity.
  old=PREV/'cache/dev'/f"{row['id']}_video64_q{qp}.pt"
  if old.with_suffix('.pt.json').exists():
   d=readtensor(old);assert d['row']['source_sha256']==row['source_sha256'] and d['row']['source_frame_indices']==row['source_frame_indices'] and len(d['features'])==64;sp=Path(d['stream_path']);assert sha(sp)==d['stream_sha256'];return sp,dict(actual_qps=d['actual_qps'],features=d['codec_feature_hashes'],bpp=d['bpp'],source='previous exact64 independent Original',previous_cache=str(old))
  sh=rt['stream'];h,w=arrays.shape[1:3];ph,pw=(h+63)//64*64,(w+63)//64*64;sps=dict(sps_id=0,height=ph,width=pw,ec_part=1,use_ada_i=0);out=io.BytesIO();sh.write_sps(out,sps);pm.clear_dpb();pm.set_curr_poc(0);qps=[];features=[];t=time.perf_counter()
  with torch.no_grad():
   for i in range(64):
    x=torch.from_numpy(arrays[i]).permute(2,0,1)[None].cuda().float()/255;x=rt['replicate_pad'](x.half(),ph-h,pw-w)*2-1;q=qp if i==0 else pm.shift_qp(qp,(0,1,0,2,0,2,0,2)[i%8]);qps.append(q);enc=im.compress(x,q) if i==0 else pm.compress(x,q)
    if i==0:pm.add_ref_frame(None,enc['x_hat'])
    features.append(None if i==0 else thash([('f',pm.dpb[0].feature)]));sh.write_ip(out,i==0,0,q,enc['bit_stream'])
  p.parent.mkdir(parents=True,exist_ok=True);disk();tmp=p.with_suffix('.bin.part');tmp.write_bytes(out.getvalue());tmp.replace(p);m=dict(actual_qps=qps,features=features,bpp=p.stat().st_size*8/(64*h*w),source='new independent Original FP16 codec, FP32 P reference',frames=64,source_row=row,encoding_seconds=time.perf_counter()-t);dump(meta,dict(sha256=sha(p),bytes=p.stat().st_size,meta=m));return p,m

def decode_frames(rt,im,pm,path,meta):
 sh=rt['stream'];reader=io.BytesIO(Path(path).read_bytes());helper=sh.SPSHelper();pm.clear_dpb();pm.set_curr_poc(0)
 with torch.no_grad():
  for i in range(64):
   header=sh.read_header(reader)
   while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
   sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader);assert q==meta['actual_qps'][i];t=time.perf_counter();dec=im.decompress(payload,sps,q) if i==0 else pm.decompress(payload,sps,q)
   if i==0:pm.add_ref_frame(None,dec['x_hat']);f=None
   else:f=pm.dpb[0].feature;assert thash([('f',f)])==meta['features'][i],'STREAM_FEATURE_MISMATCH'
   rgb=(dec['x_hat'].float().clamp(-1,1)+1)/2;yield i,q,f,rgb,sps,time.perf_counter()-t
 assert reader.tell()==Path(path).stat().st_size

def ref_sidecar(row,qp,i,ref,prev,raft,transform):
 ph,pw=ref.shape[-2:];target=(ph//16,pw//16);p=ROOT/'cache/eval_decoded_refs'/f"{row['id']}_q{qp}"/f'{i:03d}.npz'
 if p.with_suffix('.npz.json').exists():
  r=npzload(p);return torch.from_numpy(r['decoded_ref_backward']).cuda(),torch.from_numpy(r['decoded_ref_evidence']).cuda()
 with lock(f'ref_{row["id"]}_q{qp}_{i}.lock',True):
  if p.with_suffix('.npz.json').exists():r=npzload(p);return torch.from_numpy(r['decoded_ref_backward']).cuda(),torch.from_numpy(r['decoded_ref_evidence']).cuda()
  back,forward,e=decoded_flow(raft,transform,ref,prev);b16=resize_flow(back,target);f16=resize_flow(forward,target);e16=F.interpolate(e,size=target,mode='bilinear',align_corners=False);npzsave(p,dict(decoded_ref_backward=b16.cpu().numpy(),decoded_ref_forward=f16.cpu().numpy(),decoded_ref_evidence=e16.cpu().numpy()),dict(source_sha256=row['source_sha256'],source_frames=row['source_frame_indices'][i-1:i+1],QP=qp,frame=i,precision='Original P FP32; codec FP16',grid='latent16, dx/dy vector units scaled by actual width/height',RAFT_iterations=12,implementation=sha(ROOT/'reference.py'),Original_RGB_hash=[thash([('RGB',prev)]),thash([('RGB',ref)])]));return b16,e16

_old_gt={}
def native_supervision(flow,valid):
 h,w=flow.shape[-2:]
 return flow,valid.bool() & bounds(grid(h,w,flow.device)+flow,h,w)
def gt_supervision(row,i,arrays,raft,transform):
 if row['id'] not in _old_gt:
  path=PREV/'cache/dev'/f"{row['id']}_video64_q5.pt";_old_gt[row['id']]=readtensor(path) if path.with_suffix('.pt.json').exists() else None
 d=_old_gt[row['id']]
 if d is not None:
  assert d['row']['source_sha256']==row['source_sha256'] and d['row']['source_frame_indices']==row['source_frame_indices'] and len(d['flow'])==64;return native_supervision(d['flow'][i].cuda().float(),d['valid'][i].cuda().bool())
 p=ROOT/'cache/evaluation_supervision'/row['id']/f'{i:03d}.npz'
 if p.with_suffix('.npz.json').exists():d=npzload(p);return native_supervision(torch.from_numpy(d['GT_backward']).cuda(),torch.from_numpy(d['GT_valid']).cuda())
 with lock(f'gt_{row["id"]}_{i}.lock',True):
  if p.with_suffix('.npz.json').exists():d=npzload(p);return native_supervision(torch.from_numpy(d['GT_backward']).cuda(),torch.from_numpy(d['GT_valid']).cuda())
  h,w=arrays.shape[1:3];ph,pw=(h+63)//64*64,(w+63)//64*64;a=torch.from_numpy(arrays[i]).permute(2,0,1)[None].cuda().float()/255;b=torch.from_numpy(arrays[i-1]).permute(2,0,1)[None].cuda().float()/255;a=F.pad(a,(0,pw-w,0,ph-h),mode='replicate');b=F.pad(b,(0,pw-w,0,ph-h),mode='replicate');aa,bb=transform(a,b);back=raft(aa,bb,num_flow_updates=12)[-1];fw=raft(bb,aa,num_flow_updates=12)[-1];valid=flow_valid(back,fw);back=back[...,:h,:w];valid=valid[...,:h,:w];back,valid=native_supervision(back,valid);npzsave(p,dict(GT_backward=back.cpu().numpy(),GT_valid=valid.cpu().numpy()),dict(source_sha256=row['source_sha256'],frames=row['source_frame_indices'][i-1:i+1],supervision_only=True,inference_field=False));return native_supervision(back,valid)
