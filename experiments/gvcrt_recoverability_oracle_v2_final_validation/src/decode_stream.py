"""Decode a saved stream without source video, encoder captures or an oracle plan."""
from base import *
import argparse
import io
import types
import traceback

def decode(path,gpu,output):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from codec_control import unpack,MAGIC
 e=module('recoverability_standalone_engine',ENGINE)
 data=Path(path).read_bytes()
 plan,native,overhead=unpack(data) if data.startswith(MAGIC) else ([],data,0)
 im,pm=e.core.load_models(torch.device('cuda:0'),force_zero_thres=.12)
 original=pm.res_prior_param_decoder
 def prior(_self,z,context):
  params=original(z,context)
  active=[a for a in plan if a['frame']==int(pm.curr_poc) and a['factor']!=1.]
  if active:
   params=params.clone()
   for a in active:
    region=params[:,:128,a['y']:a['y']+a['h'],a['x']:a['x']+a['w']]
    base=region.clamp_min(.5);changed=(base*a['factor']).clamp_min(.5)
    region.copy_(torch.where(changed==base,region,changed))
  return params
 pm.res_prior_param_decoder=types.MethodType(prior,pm)
 pm.clear_dpb();pm.set_curr_poc(0)
 stream=io.BytesIO(native);helper=e.eco.SPSHelper();hashes=[];qps=[]
 start=time.perf_counter()
 with torch.inference_mode():
  while stream.tell()<len(native):
   header=e.eco.read_header(stream)
   while header['nal_type']==e.eco.NalType.NAL_SPS:
    helper.add_sps_by_id(e.eco.read_sps_remaining(stream,header['sps_id']));header=e.eco.read_header(stream)
   sps=helper.get_sps_by_id(header['sps_id']);q,bits=e.eco.read_ip_remaining(stream)
   if header['nal_type']==e.eco.NalType.NAL_I:
    dec=im.decompress(bits,sps,q);pm.clear_dpb();pm.add_ref_frame(None,dec['x_hat'])
   else:dec=pm.decompress(bits,sps,q)
   hashes.append(e.eco.tensor_sha(dec['x_hat']));qps.append(q)
 torch.cuda.synchronize()
 assert stream.tell()+overhead==len(data)
 report=dict(status='PASS',stream_path=str(path),stream_sha256=sha(path),actual_bits=len(data)*8,bytes_consumed=stream.tell()+overhead,frames=len(hashes),actual_qps=qps,plan_read_only_from_stream=plan,source_video_used=False,encoder_used=False,encoder_captures_used=False,external_oracle_plan_used=False,frame_reconstruction_hashes=hashes,reconstruction_sha256=e.eco.sha256_bytes(''.join(hashes).encode()),decode_wall_seconds=time.perf_counter()-start,gpu=gpu,pid=os.getpid())
 dump(output,report)
 return report

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stream',required=True);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 try:print(json.dumps(decode(a.stream,a.gpu,a.output)))
 except Exception:
  dump(a.output,dict(status='FAILED',traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid()));raise
