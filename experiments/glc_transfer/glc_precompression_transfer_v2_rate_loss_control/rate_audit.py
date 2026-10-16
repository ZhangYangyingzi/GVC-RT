"""Training-pool paired official estimate / unchanged rANS audits; zero updates."""
from preflight import *
from data import load,Frames
import argparse,math,types
V1=GLC/'experiments/glc_precompression_transfer_v1_abc'
def refhash(d):return state_hash({k:d[k] for k in ('ref_frame','ref_latent','ref_y') if d.get(k) is not None})
def run(index,method,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from runtime import models,Sender,step,quality,losses
 from src.entropy_coder import GaussianRansCodec,make_gop_stream,parse_gop_stream
 from src.entropy_coder.bitstream import parse_frame_packet,_FRAME_HEADER,_GOP_HEADER,_FRAME_SIZE
 from train import norm
 row=load(R/'plans/rate_audit.json')['rows'][index];outpath=f'audits/rate_pairs/{index:02d}_{method}.json'
 signature=dict(plan=sha(R/'plans/rate_audit.json'),worker=sha(__file__),runtime=sha(R/'runtime.py'),source=sha(V1/'final_P/C_latent_P_1000.pt'))
 if (R/outpath).exists() and load(R/outpath).get('signature')==signature and load(R/outpath)['status']=='PASS':return
 im,pm=models('cuda:0');p=None
 if method!='A_original':
  p=Sender(method).cuda().eval();p.load_state_dict(torch.load(V1/'final_P'/f'{method}_1000.pt',map_location='cpu',weights_only=False)['P'],strict=True)
 fixed=[state_hash(m.state_dict()) for m in (im,pm)];p_before=state_hash(p.state_dict()) if p else None
 f=Frames(row,True);coder=GaussianRansCodec();packets=[];hashes=[];records=[];dpb=None;tic=time.time();quant=[]
 for model in (im,pm):
  original=model.quant
  def capture(x,original=original):
   y=original(x);quant.append(state_hash({'quantized':y}));return y
  model.quant=capture
 with torch.inference_mode():
  for pos in range(64):
   x=f.tensor(pos,'cuda:0');quant.clear();estimated,re,rd,pad=step(im,pm,x,dpb,row['q'],pos,p);estimated_symbols=list(quant)
   quant.clear();encoded,rec,dpb,_=step(im,pm,x,dpb,row['q'],pos,p,mode='compress',coder=coder)
   assert estimated_symbols==quant,('quantization symbols mismatch',index,method,pos)
   err=float((re-rec).abs().max());assert torch.equal(re,rec),('official/compress reconstruction mismatch',index,method,pos,err)
   assert refhash(rd)==refhash(dpb),('reference mismatch',index,method,pos)
   packet=encoded['bit_stream'];parsed=parse_frame_packet(packet);packets.append(packet);hashes.append(refhash(dpb));hyper_bytes=(parsed['z_height']*parsed['z_width']*14+7)//8
   actual=len(packet);payload=len(parsed['y_payload']);assert actual==_FRAME_HEADER.size+hyper_bytes+payload
   recrow=dict(frame=pos,official_bits=float(estimated['bit']),official_y_bits=float(estimated['bit_y']),official_hyper_bits=float(estimated['bit_z']),payload_bits=payload*8,hyper_bits=hyper_bytes*8,frame_header_bits=_FRAME_HEADER.size*8,frame_bytes=actual,alignment_and_rans_flush='combined in y payload; cannot isolate from opaque coder',padding=list(pad),symbol_hashes=estimated_symbols,reconstruction_sha256=state_hash({'rec':rec}),reference_sha256=hashes[-1],max_reconstruction_difference=err)
   records.append(recrow)
 data=make_gop_stream(256,256,packets);hh,ww,packets=parse_gop_stream(data);assert (hh,ww)==(256,256)
 frozen_ok=fixed==[state_hash(m.state_dict()) for m in (im,pm)];del im,pm;torch.cuda.empty_cache();im,pm=models('cuda:0');dpb=None
 with torch.inference_mode():
  for pos,packet in enumerate(packets):
   e=im.decompress(packet,coder) if pos==0 else pm.decompress(packet,dpb,coder)
   dpb=dict(ref_frame=e['x_hat'],ref_latent=e['ref_latent'],ref_y=None) if pos==0 else e['dpb'];assert refhash(dpb)==hashes[pos],('independent decoder reference mismatch',index,method,pos)
 assert frozen_ok and fixed==[state_hash(m.state_dict()) for m in (im,pm)]
 assert p is None or p_before==state_hash(p.state_dict())
 overhead=(_GOP_HEADER.size+64*_FRAME_SIZE.size)*8;assert len(data)*8==sum(r['frame_bytes'] for r in records)*8+overhead
 save(outpath,dict(status='PASS',signature=signature,index=index,method=method,domain=row['domain'],q=row['q'],sample=row,frames=records,official_bits=sum(r['official_bits'] for r in records),payload_bits=sum(r['payload_bits'] for r in records),hyper_bits=sum(r['hyper_bits'] for r in records),frame_header_bits=sum(r['frame_header_bits'] for r in records),GOP_header_and_lengths_bits=overhead,complete_bytes=len(data),actual_bpp=len(data)*8/(64*256*256),effective_area=256*256,padded_area=256*256,symbols_and_reference_history_identical=True,independent_decode=True,stream_sha256=__import__('hashlib').sha256(data).hexdigest(),coder_statistics=coder.statistics,CDF_sha256=__import__('hashlib').sha256(coder.cdfs.tobytes()).hexdigest(),CDF_scale_min=coder.scale_min,CDF_scale_max=coder.scale_max,CDF_scale_levels=coder.scale_levels,optimizer_updates=0,frozen_unchanged=True,seconds=time.time()-tic,peak_MiB=torch.cuda.max_memory_reserved()/2**20))
 f.close();print('PAIR PASS',index,method,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--index',type=int,required=True);a.add_argument('--method',choices=['A_original','B_rgb_P','C_latent_P'],required=True);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);x=a.parse_args();run(x.index,x.method,x.gpu)
