#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src'
for p in (REPO,V9):sys.path.insert(0,str(p))
from gvc_hooks import load_models
from run_oracle import unit,save_png,model_hash
from src.utils.stream_helper import read_header,read_sps_remaining,read_ip_remaining,NalType,SPSHelper
def read(p):return list(csv.DictReader(open(p,newline='')))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--videos',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0');rows=[]
 for tag in a.videos.split(','):
  s=read(ROOT/'parts'/f'{tag}_summaries.csv');sel=read(ROOT/'parts'/f'{tag}_selected.csv');base=next(x for x in s if x['kind']=='baseline');paths={base['bitstream_path']:('baseline','')}
  for x in sel:
   if x['target_reached']=='True':paths[x['bitstream_path']]=('selected',x['target_ratio'])
  for path,(kind,target) in paths.items():
   im,dec=load_models(dev);before=model_hash(im,dec);dec.clear_dpb();dec.set_curr_poc(0);p=Path(path);buf=io.BytesIO(p.read_bytes());helper=SPSHelper();count=0;finite=True;shape=True;maxdiff=0
   while buf.tell()<p.stat().st_size:
    h=read_header(buf)
    while h['nal_type']==NalType.NAL_SPS:
     sp=read_sps_remaining(buf,h['sps_id']);helper.add_sps_by_id(sp);h=read_header(buf)
    sp=helper.get_sps_by_id(h['sps_id']);qp,payload=read_ip_remaining(buf)
    with torch.inference_mode():
     if h['nal_type']==NalType.NAL_I:
      d=im.decompress(payload,sp,qp);dec.clear_dpb();dec.add_ref_frame(None,d['x_hat'])
     else:d=dec.decompress(payload,sp,qp)
    out=unit(d['x_hat']);finite&=bool(torch.isfinite(out).all());shape&=tuple(out.shape)==(1,3,1080,1920)
    od=ROOT/'baseline_frames'/tag if kind=='baseline' else ROOT/'reconstruction_frames'/tag/f'target_{target}'
    op=od/f'frame_{count:06d}.png'
    if op.exists():
     from PIL import Image
     import numpy as np
     old=np.asarray(Image.open(op).convert('RGB'),dtype=np.int16);new=torch.clamp(out*255,0,255).round().byte()[0].permute(1,2,0).cpu().numpy().astype(np.int16);maxdiff=max(maxdiff,int(np.abs(old-new).max()))
    else:save_png(out,op)
    count+=1
   after=model_hash(im,dec);ok=count==16 and buf.tell()==p.stat().st_size and finite and shape and maxdiff==0 and before==after
   rows.append({'video_tag':tag,'stream_type':kind,'target_ratio':target,'bitstream_path':str(p),'bitstream_bytes':p.stat().st_size,'bytes_consumed':buf.tell(),'frame_count':count,'shape_correct':shape,'finite':finite,'reconstruction_max_abs_diff_uint8':maxdiff,'model_hash_before':before,'model_hash_after':after,'decode_status':'PASS' if ok else 'FAIL'})
   print(tag,kind,target,'PASS' if ok else 'FAIL',flush=True)
 (ROOT/'parts'/f'decode_gpu{a.gpu}.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
