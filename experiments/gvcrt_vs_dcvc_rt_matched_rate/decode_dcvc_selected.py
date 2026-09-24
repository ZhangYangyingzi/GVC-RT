#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,os,re,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parent;DCVC=Path('/Huang_group/zyyz/Projects/DCVC_RT');sys.path.insert(0,str(DCVC))
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import get_state_dict
from src.utils.stream_helper import SPSHelper,NalType,read_header,read_sps_remaining,read_ip_remaining
from src.utils.transforms import ycbcr2rgb
def mh(*models):
 h=hashlib.sha256()
 for m in models:
  for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def recdir(tag,rc):
 a,b=rc.replace('I','').split('_P');return ROOT/'reconstructions'/tag/(f'DCVC_q{a}' if a==b else f'DCVC_qi{a}_qp{b}')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--tags',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0')
 im=DMCI().to(dev).eval();im.load_state_dict(get_state_dict(str(DCVC/'checkpoints/cvpr2025_image.pth.tar')));im.update(.12);im.half()
 pm=DMC().to(dev).eval();pm.load_state_dict(get_state_dict(str(DCVC/'checkpoints/cvpr2025_video.pth.tar')));pm.update(.12);pm.half();im.set_use_two_entropy_coders(True);pm.set_use_two_entropy_coders(True)
 before=mh(im,pm);pairs=list(csv.DictReader(open(ROOT/'matched_rate_pairs.csv')));rows=[]
 for x in pairs:
  tag=f"{x['dataset']}_{int(x['video_id']):02d}"
  if tag not in a.tags.split(','):continue
  path=Path(x['dcvc_bitstream_path']);buf=io.BytesIO(path.read_bytes());shelper=SPSHelper();pm.set_curr_poc(0);count=0;maxdiff=0;finite=True;shape=True
  with torch.inference_mode():
   while buf.tell()<path.stat().st_size:
    h=read_header(buf)
    while h['nal_type']==NalType.NAL_SPS:
     sp=read_sps_remaining(buf,h['sps_id']);shelper.add_sps_by_id(sp);h=read_header(buf)
    sp=shelper.get_sps_by_id(h['sps_id']);qp,payload=read_ip_remaining(buf)
    if h['nal_type']==NalType.NAL_I:
     d=im.decompress(payload,sp,qp);pm.clear_dpb();pm.add_ref_frame(None,d['x_hat'])
    else:
     if sp['use_ada_i']:pm.reset_ref_feature()
     d=pm.decompress(payload,sp,qp)
    out=d['x_hat'][:,:,:1080,:1920];finite &= bool(torch.isfinite(out).all());shape &= tuple(out.shape)==(1,3,1080,1920)
    rgb=torch.clamp(ycbcr2rgb(out)*255,0,255).round().byte()[0].cpu().numpy()
    ref=np.asarray(Image.open(recdir(tag,x['dcvc_rate_control'])/f'im{count+1:05d}.png').convert('RGB'),dtype=np.uint8).transpose(2,0,1)
    maxdiff=max(maxdiff,int(np.abs(rgb.astype(np.int16)-ref.astype(np.int16)).max()));count+=1
  after=mh(im,pm);ok=count==64 and buf.tell()==path.stat().st_size and finite and shape and maxdiff==0 and before==after
  rows.append({'codec':'DCVC-RT','dataset':x['dataset'],'video':x['video'],'video_id':x['video_id'],'rate_control':x['dcvc_rate_control'],'bitstream_path':str(path),'bytes_consumed':buf.tell(),'bitstream_bytes':path.stat().st_size,'frame_count':count,'frame_count_correct':count==64,'shape_correct':shape,'finite':finite,'reconstruction_max_abs_diff_uint8':maxdiff,'model_hash_before':before,'model_hash_after':after,'decode_status':'PASS' if ok else 'FAIL'})
  print(tag,x['dcvc_rate_control'],'PASS' if ok else 'FAIL',flush=True)
 (ROOT/'parts'/f'dcvc_decode_audit_gpu{a.gpu}.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
