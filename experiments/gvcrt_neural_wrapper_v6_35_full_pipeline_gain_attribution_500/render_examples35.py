import os,json
from pathlib import Path
R=Path(__file__).resolve().parent
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ.setdefault('CUDA_VISIBLE_DEVICES','6')
import torch,numpy as np
from PIL import Image,ImageDraw
from io35 import load,sha
from adapter35 import frames_for,runtime,decode
def load_csv(p):
 import csv
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
methods=['original','N_oursLoss_noP_500','P_oursLoss_withP_500']
def fixed_source_frames(v):
 if v['dataset']!='ulong':return frames_for(v)
 import cv2,torch,hashlib
 from PIL import Image
 cv2.setNumThreads(1);row=v['source_row'];cap=cv2.VideoCapture(v['source_path']);out=[]
 assert cap.isOpened() and len(v['source_frame_indices'])==64
 for pos,frame_index in enumerate(v['source_frame_indices']):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index));ok,bgr=cap.read();assert ok,(v['name'],frame_index)
  a=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(a.tobytes()).hexdigest()==v['frame_rgb_sha256'][pos]
  out.append(torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).float()/255)
 cap.release();return out
for ds,vid in [('ulong',0),('uvg_holdout',2),('hevc_b',0)]:
 manifest=load(R/'eval_manifest.json');v=next(x for x in manifest[ds] if x['video_index']==vid);src=fixed_source_frames(v)
 captured={}
 for m in methods:
  task={'stage':'baseline' if m=='original' else 'adapt','dataset':ds,'video':vid,'method':m,'step':500,'branch':m,'qp':5}
  eng,rt=runtime(task)
  point=next(x for x in load_csv(R/'results/all_rd_points.csv') if x['dataset']==ds and int(x['video'])==vid and x['method']==m and int(x['QP'])==5)
  point['actual_qps']=json.loads(point['actual_qps']) if isinstance(point['actual_qps'],str) else point['actual_qps']
  bitstream=Path(point['bitstream_path']);assert bitstream.is_file() and sha(bitstream)==point['bitstream_sha256']
  captures={}
  for i,rgb,info in decode(eng,rt,bitstream,point):
   if i not in (16,32,63):continue
   a=(rgb[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())
   gt=(src[i][0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())
   y0=(a.shape[0]-512)//2;x0=(a.shape[1]-512)//2
   captures[i]=(gt[y0:y0+512,x0:x0+512],a[y0:y0+512,x0:x0+512])
  captured[m]=captures
 out=R/'plots/fixed_examples';out.mkdir(parents=True,exist_ok=True)
 canvas=Image.new('RGB',(4*512,3*540),'white');d=ImageDraw.Draw(canvas)
 for row,i in enumerate((16,32,63)):
  images=[captured[m][i][0 if m=='original' else 1] if m!='original' else captured[m][i][0] for m in methods]
  # Ground truth is identical for all methods; always show the source frame from the original decode entry.
  images=[captured['original'][i][0],captured['original'][i][1],captured['N_oursLoss_noP_500'][i][1],captured['P_oursLoss_withP_500'][i][1]]
  bpp=[]
  for m in methods:bpp.append(float(next(x['actual_bpp'] for x in load_csv(R/'results/all_rd_points.csv') if x['dataset']==ds and int(x['video'])==vid and x['method']==m and int(x['QP'])==5)))
  labels=['GT','Original',f'N no-P (500) bpp={bpp[1]:.6g}',f'P (500) bpp={bpp[2]:.6g}']
  for col,(label,a) in enumerate(zip(labels,images)):
   canvas.paste(Image.fromarray(a),(col*512,row*540+24));d.text((col*512+6,row*540+5),f'{label} | {v["name"]} | QP5 | frame {i}',fill='black')
 p=out/f'{ds}_{v["name"]}_qp5_frames16_32_63_center512.png';canvas.save(p)
 meta=dict(dataset=ds,video_index=vid,sequence=v['name'],QP=5,frames=[16,32,63],crop_xywh=[(v['width']-512)//2,(v['height']-512)//2,512,512],methods=methods,bitstreams={m:{'path':next(x['bitstream_path'] for x in load_csv(R/'results/all_rd_points.csv') if x['dataset']==ds and int(x['video'])==vid and x['method']==m and int(x['QP'])==5),'sha256':next(x['bitstream_sha256'] for x in load_csv(R/'results/all_rd_points.csv') if x['dataset']==ds and int(x['video'])==vid and x['method']==m and int(x['QP'])==5)} for m in methods},output=str(p),output_sha256=sha(p))
 (p.with_suffix('.json')).write_text(json.dumps(meta,indent=2)+'\n')
 print(p,flush=True)
