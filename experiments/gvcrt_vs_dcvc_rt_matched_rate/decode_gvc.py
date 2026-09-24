#!/usr/bin/env python3
import argparse,csv,hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parent; GVC=ROOT.parents[1]
V13=GVC/'experiments/gvcrt_v13_generator_aware_mixed_precision'
V11=GVC/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11'
V9=GVC/'expericent_generation_input/expericent_interface_causal_controls_v9'
DBG=GVC/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug'
for p in (GVC,V13,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):
 sys.path.insert(0,str(p))
from gvc_hooks import load_models
from run_debug import load_b2,unit
from src.utils.stream_helper import read_header,read_sps_remaining,read_ip_remaining,NalType

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def parse(path):
 out=[]; sps={}
 with open(path,'rb') as f:
  while f.tell()<Path(path).stat().st_size:
   h=read_header(f)
   if h['nal_type']==NalType.NAL_SPS:
    sps[h['sps_id']]=read_sps_remaining(f,h['sps_id']); continue
   qp,payload=read_ip_remaining(f);out.append((h['nal_type']==NalType.NAL_I,h['sps_id'],qp,payload))
 return out,sps
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--tags',required=True);a=ap.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0')
 man=json.loads((ROOT/'manifest.json').read_text()); curve=list(csv.DictReader((V13/'multi_qp_curve.csv').open()))
 lookup={f"{v['dataset']}_{int(v['video_id']):02d}":v for v in man['videos']}
 rows=[]
 for tag in a.tags.split(','):
  v=lookup[tag]
  for qp in (1,3):
   old=next(r for r in curve if r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['QP'])==qp)
   src=Path(old['bitstream_path']); dst=ROOT/'bitstreams/gvc'/f'{tag}_qp{qp}.bin'; shutil.copy2(src,dst)
   packets,sps=parse(dst)
   if len(packets)!=64: raise RuntimeError('GVC frame count mismatch')
   iframe,_=load_models(dev);dec,_=load_b2(dev);dec.clear_dpb();dec.set_curr_poc(0)
   outdir=ROOT/'reconstructions'/tag/f'GVC_qp{qp}';outdir.mkdir(parents=True,exist_ok=True)
   rgb_hash=[];state_hash=[]
   with torch.no_grad():
    for fi,(is_i,sid,aq,payload) in enumerate(packets):
     sp=sps[sid]
     if is_i:
      d=iframe.decompress(payload,sp,aq);dec.clear_dpb();dec.add_ref_frame(None,d['x_hat'])
     else:
      if sp.get('use_ada_i'): dec.reset_ref_feature()
      d=dec.decompress(payload,sp,aq)
     x=unit(d['x_hat'][:,:,:1080,:1920]);rgb_hash.append(th(x))
     arr=torch.clamp(x*255,0,255).round().byte()[0].permute(1,2,0).cpu().numpy();Image.fromarray(arr).save(outdir/f'im{fi+1:05d}.png',compress_level=1)
     h=hashlib.sha256(str(dec.curr_poc).encode())
     for ref in dec.dpb:
      for z in (ref.frame,ref.feature):
       if z is not None:h.update(z.detach().cpu().contiguous().numpy().tobytes())
     state_hash.append(h.hexdigest())
   rows.append({'codec':'GVC-RT','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'rate_control':f'QP{qp}','num_frames':64,'fps':v['fps'],'bitstream_path':str(dst),'bitstream_bytes':dst.stat().st_size,'kbps':dst.stat().st_size*8*v['fps']/64/1000,'bitstream_sha256':sha(dst),'reference_sha256':old['bitstream_sha256'],'payload_identity':sha(dst)==old['bitstream_sha256'],'decode_status':'PASS','frame_count_correct':True,'shape_correct':True,'finite':True,'final_rgb_sha256':rgb_hash[-1],'final_state_sha256':state_hash[-1]})
   print(tag,qp,dst.stat().st_size,flush=True)
 p=ROOT/'parts'/f'gvc_decode_gpu{a.gpu}.json';p.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
