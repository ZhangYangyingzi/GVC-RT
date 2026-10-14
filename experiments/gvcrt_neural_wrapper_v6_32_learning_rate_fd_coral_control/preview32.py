"""Fixed first HEVC/MCL QP5 frames16/32/63; no encoding or result selection."""
from io32 import *
import argparse
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 import torch.nn.functional as F
 from adapter32 import runtime,decode,frames_for,backend
 from evaluate32 import valid,receipt
 backend();records=[]
 for d in ('hevc_b','mcljcv'):
  v=min(load(ROOT/'eval_manifest.json')[d],key=lambda v:v['video_index']);frames=frames_for(v)
  models=[('original',None),('B1_initial',None)]+[(f'{b}_1000',b) for b in BRANCHES]
  for m,b in models:
   t=dict(stage='baseline' if b is None else 'new',dataset=d,video=v['video_index'],method=m,qp=5)
   if b:t.update(branch=b,step=1000)
   if not valid(t):records.append(dict(dataset=d,method=m,status='unavailable',reason='missing fixed model evaluation'));continue
   r=load(receipt(t))['result'];eng,rt=runtime(t)
   for i,x,info in decode(eng,rt,r['bitstream_path'],r):
    if i not in (16,32,63):continue
    outs={m:x}
    if m=='original':outs['GT']=frames[i].cuda()
    for name,im in outs.items():
     p=ROOT/'previews'/d/name/f'frame_{i:02d}.png';eng.eco.save_png(F.interpolate(im,size=(270,480),mode='area'),p);records.append(dict(dataset=d,video=v['name'],method=name,frame=i,bpp=r['bpp'],path=str(p),sha256=sha(p),display_only=True,status='available'))
   del rt,eng;torch.cuda.empty_cache()
 dump(ROOT/'audits/fixed_previews.json',dict(status='PARTIAL' if any(r['status']=='unavailable' for r in records) else 'PASS',records=records,extra_encodes=0,selection='first by video_index, QP5, frames16/32/63'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));main(ap.parse_args().gpu)
