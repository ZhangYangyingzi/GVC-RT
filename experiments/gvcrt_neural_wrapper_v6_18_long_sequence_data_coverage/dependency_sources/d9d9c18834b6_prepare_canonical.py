import json, hashlib, subprocess
from pathlib import Path
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parent
V51=ROOT.parent/'gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization'
ulong=json.loads((V51/'test_manifest.json').read_text())['videos']
uvg=json.loads((V51/'uvg_available_manifest.json').read_text())['videos']
out=[]
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def save_frames(dataset,name,source,indices,cmd):
 d=ROOT/'canonical_sources'/dataset/name; d.mkdir(parents=True,exist_ok=True)
 size=1920*1080*3; p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 hashes=[]
 try:
  for i in range(len(indices)):
   raw=p.stdout.read(size); assert len(raw)==size,(name,i,len(raw))
   a=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
   q=d/f'frame_{i:06d}.png'; Image.fromarray(a,'RGB').save(q,format='PNG',compress_level=0)
   hashes.append(digest(q))
 finally:
  err=p.stderr.read(); code=p.wait()
  assert code==0,err.decode()
 whole=hashlib.sha256(''.join(hashes).encode()).hexdigest()
 out.append(dict(dataset=dataset,name=name,original_path=source,original_sha256=digest(source),original_width=1920,original_height=1080,original_fps=30.0 if dataset=='ulong' else 120.0,original_format='MP4' if dataset=='ulong' else 'raw YUV420p',canonical_width=1920,canonical_height=1080,canonical_fps=30.0,canonical_num_frames=64,canonical_format='RGB24 lossless PNG',selected_original_frame_indices=indices,canonical_frame_sha256=hashes,whole_sequence_rgb_hash=whole,loader='load_canonical_frames'))
for v in ulong:
 tag=v['name']; cmd=['ffmpeg','-v','error','-i',v['source_path'],'-vf','select=between(n\\,0\\,63)','-vsync','0','-frames:v','64','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 save_frames('ulong',tag,v['source_path'],list(range(64)),cmd)
for v in uvg:
 idx=list(range(0,256,4)); cmd=['ffmpeg','-v','error','-f','rawvideo','-pixel_format','yuv420p','-video_size','1920x1080','-framerate','120','-i',v['path'],'-vf','select=not(mod(n\\,4)),scale=in_color_matrix=bt709:in_range=limited:out_range=full,format=rgb24','-vsync','0','-frames:v','64','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 save_frames('uvg',v['sequence_name'],v['path'],idx,cmd)
(ROOT/'canonical_source_audit.json').write_text(json.dumps({'status':'PASS','records':out,'canonical_fps':30.0,'canonical_num_frames':64,'uvg_stride':4},indent=2)+'\n')
print('canonical PASS',len(out))
