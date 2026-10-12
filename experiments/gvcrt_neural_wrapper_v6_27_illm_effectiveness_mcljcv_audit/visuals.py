"""Fresh independent decode of nearest measured rate points, fixed video/frame/ROI."""
from io21 import *
import io,math

def main():
 while True:
  gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(x.split(',')[0]):float(x.split(',')[1]) for x in gpu_rows if int(x.split(',')[0]) in (4,5,6,7)};gpu=max(free,key=free.get)
  if free[gpu]>=16000:break
  time.sleep(15)
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from PIL import Image,ImageDraw
 from adapter import engine,frames_for
 torch.set_num_threads(2);videos=sources('mcljcv');chosen=[videos[0],videos[len(videos)//2],videos[-1]];cfg=evalcfg();eng=engine();audit=[]
 for v in chosen:
  data={m:[load(point('mcljcv',m,v['video_index'],q)) for q in range(10)] for m in METHODS};lo=max(min(r['bpp'] for r in rs) for rs in data.values());hi=min(max(r['bpp'] for r in rs) for rs in data.values());assert hi>lo;target=math.sqrt(lo*hi);ref=min(data['B1_initial'],key=lambda r:abs(math.log(r['bpp']/target)));records={m:min(rs,key=lambda r:abs(math.log(r['bpp']/ref['bpp']))) for m,rs in data.items()};captured={};frames=frames_for(v)
  for m,r in records.items():
   runtime=eng.Runtime(dict(experiment='B',IP=-1,methods={m:[None,None] if m=='original' else [m,m]},checkpoints={} if m=='original' else {m:cfg['checkpoints'][m]},force_zero_thres=.12),m,torch.device('cuda:0'));ia,pa=runtime.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0);eco=eng.eco;assert sha(r['bitstream_path'])==r['bitstream_sha256'];stream=io.BytesIO(Path(r['bitstream_path']).read_bytes());helper=eco.SPSHelper();hashes=[];imgs={}
   with torch.inference_mode():
    for i in range(64):
     header=eco.read_header(stream)
     while header['nal_type']==eco.NalType.NAL_SPS:helper.add_sps_by_id(eco.read_sps_remaining(stream,header['sps_id']));header=eco.read_header(stream)
     sps=helper.get_sps_by_id(header['sps_id']);q,bits=eco.read_ip_remaining(stream)
     if i==0:decoded=ia.decompress(bits,sps,q);pa.clear_dpb();pa.add_ref_frame(None,decoded['x_hat'])
     else:decoded=pa.decompress(bits,sps,q)
     hashes.append(eco.tensor_sha(decoded['x_hat']))
     if i in (16,48,63):
      z=eng.core.unit(decoded['x_hat'],1080,1920);imgs[i]=(z[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
   assert eco.sha256_bytes(''.join(hashes).encode())==r['reconstruction_sha256'] and stream.tell()==r['real_bytes'];runtime.assert_frozen(ia,pa);del ia,pa,runtime;torch.cuda.empty_cache();captured[m]=imgs
  for i in (16,48,63):
   # Center512x512 ROI is fixed before model comparison; only layout is resized, pixels are unchanged.
   box=(704,284,1216,796);gt=(frames[i][0].permute(1,2,0).numpy()*255).round().astype(np.uint8);panels=[('GT',Image.fromarray(gt).crop(box))]+[(m,Image.fromarray(captured[m][i]).crop(box)) for m in METHODS];sheet=Image.new('RGB',(512*3,560*2),'white');draw=ImageDraw.Draw(sheet)
   for j,(m,img) in enumerate(panels):
    x=(j%3)*512;y=(j//3)*560;sheet.paste(img,(x,y+48));label=m if m=='GT' else f"{m}\nQP{records[m]['QP']} bpp={records[m]['bpp']:.7f} delta={records[m]['bpp']-ref['bpp']:+.7f}";draw.text((x+4,y+4),label,fill='black')
   p=ROOT/'plots/visuals'/f'{v["name"]}_frame{i}.png';p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p);audit.append(dict(sequence=v['name'],frame=i,ROI=box,nearest_actual_rate_points={m:dict(QP=r['QP'],bpp=r['bpp'],delta_bpp_vs_B1=r['bpp']-ref['bpp'],bitstream_sha256=r['bitstream_sha256']) for m,r in records.items()},image_path=str(p),SHA256=sha(p),no_image_interpolation=True,no_postprocessing=True,independent_decode_verified=True))
 write(ROOT/'results/mcljcv/visual_rate_points.csv',audit);dump(ROOT/'audits/visual_integrity.json',dict(status='PASS',videos=[v['name'] for v in chosen],frames=[16,48,63],images=len(audit),ROI=[704,284,1216,796],source_index_selection='first, middle floor(N/2), last sorted identity',independent_decode=True))
if __name__=='__main__':main()
