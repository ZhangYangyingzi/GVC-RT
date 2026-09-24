#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
GVC=ROOT.parents[1]
V14=GVC/'experiments/gvcrt_v14_replacement_oracle'
DCVC=Path('/Huang_group/zyyz/Projects/DCVC_RT')
sys.path.insert(0,str(GVC/'expericent_generation_input/expericent_interface_causal_controls_v9/src'))
sys.path.insert(0,str(GVC))
from src.utils.video_reader import YUV420Reader
from src.utils.transforms import ycbcr420_to_444_np

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def tag(v): return f"{v['dataset']}_{int(v['video_id']):02d}"

def main():
 man=json.loads((V14/'manifest.json').read_text())
 (ROOT/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
 for d in ('bitstreams/gvc','bitstreams/dcvc','reconstructions','comparison_frames','visualizations','logs','parts','source_frames'):
  (ROOT/d).mkdir(parents=True,exist_ok=True)
 audits=[]
 for v in man['videos']:
  out=ROOT/'source_frames'/tag(v); out.mkdir(parents=True,exist_ok=True)
  frames=[]
  if v['dataset']=='fresh_ulong':
   p=subprocess.Popen(['ffmpeg','-v','error','-i',v['source_path'],'-map','0:v:0','-frames:v','64','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
   for i in range(64):
    raw=p.stdout.read(1920*1080*3)
    if len(raw)!=1920*1080*3: raise RuntimeError(f'short source {tag(v)} frame {i}')
    frames.append(np.frombuffer(raw,np.uint8).reshape(1080,1920,3).copy())
   p.stdout.close(); rc=p.wait()
   if rc: raise RuntimeError(f'ffmpeg source decode failed {tag(v)}')
  else:
   r=YUV420Reader(v['source_path'],1920,1080)
   try:
    for i in range(64):
     y,uv=r.read_one_frame()
     if y is None: raise RuntimeError(f'short source {tag(v)} frame {i}')
     a=ycbcr420_to_444_np(y,uv,order=0)
     frames.append(np.asarray(a).transpose(1,2,0).astype(np.uint8))
   finally: r.close()
  hh=hashlib.sha256()
  for i,a in enumerate(frames,1):
   hh.update(a.tobytes()); Image.fromarray(a).save(out/f'im{i}.png',compress_level=1)
  audits.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'frames':64,'width':1920,'height':1080,'fps':v['fps'],'source_path':v['source_path'],'source_sha256':v['source_sha256'],'rgb_tensor_sha256':hh.hexdigest(),'png_directory':str(out)})
 (ROOT/'parts/source_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
 print(json.dumps({'source_sequences':len(audits),'manifest_sha256':sha(ROOT/'manifest.json')},indent=2))
if __name__=='__main__': main()
