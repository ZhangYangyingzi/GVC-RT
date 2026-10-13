"""Only the prescribed first videos/QP5/frame16 missing full-float previews."""
from audit31 import *
import argparse,subprocess
def worker(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 import torch.nn.functional as F
 from adapter31 import backend,runtime,decode
 from color31 import transform
 backend();art={}
 for d in ('hevc_b','mcljcv'):
  for method in METHODS:
   t=dict(stage='color',dataset=d,video=0,method=method,qp=5);assert valid(t)
   fits=read(ROOT/'raw/color'/(uid(t)+'.fit.csv'))
   params={r['type']:{key:json.loads(r[key]) for key in ('a','b')} for r in fits}
   r=load(oldpoint(d,method,0,5));eng,rt=runtime(method)
   for i,x,_ in decode(eng,rt,r['bitstream_path'],r):
    if i==16:
     for typ in ('BIAS','AFFINE'):
      out=transform(x,typ,params);p=ROOT/'previews'/d/method/f'frame16_{typ}.png';eng.eco.save_png(F.interpolate(out,size=(270,480),mode='area'),p);art[str(p)]=sha(p)
     break
   del rt,eng;torch.cuda.empty_cache()
 dump(ROOT/'audits/fixed_previews.json',dict(status='PASS',artifacts=art,frame=16,fit_P_frames=list(range(1,32)),uses_frozen_fit=True,display_only=True,extra_formal_RD_points=0,extra_encodings=0,optimizer_updates=0,annotation='frame16 is in fit range, shown only; no evaluation metric uses it',physical_GPU=gpu))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--gpu',type=int,choices=(4,5,6,7));a=ap.parse_args()
 if a.launch:
  if (ROOT/'audits/fixed_previews.json').exists():raise SystemExit(0)
  from run_all import resources
  while True:
   free,host=resources();g=max(free,key=free.get)
   if free[g]>16000 and host>16000:break
   time.sleep(60)
  subprocess.run([PYTHON,'-B','-u',str(ROOT/'preview31.py'),'--gpu',str(g)],check=True,cwd=REPO)
 else:worker(a.gpu)
