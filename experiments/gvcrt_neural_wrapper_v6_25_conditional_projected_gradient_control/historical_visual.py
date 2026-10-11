"""Independent decode of existing complete streams; fixed real-rate image panels."""
import argparse,io,math
from io21 import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));a=ap.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
 import torch,numpy as np
 from PIL import Image,ImageDraw
 from adapter import engine,frames_for,validate
 torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();eng=engine();v=next(v for v in sources('hevc_b') if v['name']=='BQTerrace');methods=('B1_initial','historical_A_continue_1000','historical_B_projected_1000');raw={m:[load(point('hevc_b',m,v['video_index'],q)) for q in range(10)] for m in methods}
 lo=max(min(r['bpp'] for r in rr) for rr in raw.values());hi=min(max(r['bpp'] for r in rr) for rr in raw.values());assert lo<hi
 frames=frames_for(v);selected={};bands=[];cfg=evalcfg();frame=48;box=(704,284,1216,796);out=ROOT/'historical/visuals';out.mkdir(parents=True,exist_ok=True)
 for j,band in enumerate(('low','middle','high')):
  target=math.exp(math.log(lo)+(j+.5)/3*(math.log(hi)-math.log(lo)));ref=min(raw['B1_initial'],key=lambda r:abs(math.log(r['bpp']/target)));rs=[ref,*[min(raw[m],key=lambda r:abs(math.log(r['bpp']/ref['bpp']))) for m in methods[1:]]];panels=[];spec=[]
  gt=(frames[frame][0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8);panels.append((Image.fromarray(gt).crop(box),'GT frame48; fixed ROI'))
  for r in rs:
   m,q=r['method'],r['external_qp'];validate(r,v,q,m);key=f'{m}_q{q}'
   if key not in selected:
    cp=cfg['checkpoints'][m];rt=eng.Runtime(dict(experiment='B',IP=-1,methods={m:[m,m]},checkpoints={m:cp},force_zero_thres=.12),m,torch.device('cuda:0'));im,pm=rt.models(1088,1920);pm.clear_dpb();pm.set_curr_poc(0);p=Path(r['bitstream_path']);assert sha(p)==r['bitstream_sha256'];stream=io.BytesIO(p.read_bytes());helper=eng.eco.SPSHelper();hashes=[]
    with torch.inference_mode():
     for i in range(64):
      header=eng.eco.read_header(stream)
      while header['nal_type']==eng.eco.NalType.NAL_SPS:helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
      sps=helper.get_sps_by_id(header['sps_id']);qa,bits=eng.eco.read_ip_remaining(stream);assert qa==r['actual_qps'][i]
      if i==0:
       assert header['nal_type']==eng.eco.NalType.NAL_I;z=im.decompress(bits,sps,qa);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,z['x_hat'])
      else:z=pm.decompress(bits,sps,qa)
      hashes.append(eng.eco.tensor_sha(z['x_hat']))
      if i==frame:
       x=eng.core.unit(z['x_hat'],1080,1920)[0].float().cpu().permute(1,2,0).numpy();selected[key]=Image.fromarray((x*255).round().clip(0,255).astype(np.uint8)).crop(box)
    assert stream.tell()==p.stat().st_size and eng.eco.sha256_bytes(''.join(hashes).encode())==r['reconstruction_sha256'];rt.assert_frozen(im,pm);del rt,im,pm;torch.cuda.empty_cache()
   delta=r['bpp']-ref['bpp'];panels.append((selected[key],f'{m} QP{q}; bpp={r["bpp"]:.7f}; delta={delta:+.7f}'));spec.append(dict(method=m,QP=q,bpp=r['bpp'],delta_bpp_vs_B1=delta,bitstream=str(r['bitstream_path']),bitstream_sha256=r['bitstream_sha256'],reconstruction_sha256=r['reconstruction_sha256']))
  image=Image.new('RGB',(512*len(panels),560),'white');draw=ImageDraw.Draw(image)
  for k,(pic,label) in enumerate(panels):image.paste(pic,(512*k,48));draw.text((512*k+6,8),label,fill='black')
  p=out/f'BQTerrace_frame48_{band}.png';image.save(p);bands.append(dict(band=band,fixed_frame_index=frame,source_frame_index=v['source_frame_indices'][frame],fixed_ROI=box,common_bpp=[lo,hi],target_bpp=target,selected_points=spec,image=str(p),sha256=sha(p),selection='nearest measured ln(bpp), paired to measured B1 point; no image interpolation',display_conversion='standard RGB [0,1] to uint8, no enhancement'))
 dump(ROOT/'historical/visual_integrity.json',dict(status='PASS',independent_decode=True,full64=True,stream_reconstruction_hash_verified=True,panels=bands,physical_gpu=a.gpu,logical_gpu=0,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
if __name__=='__main__':main()
