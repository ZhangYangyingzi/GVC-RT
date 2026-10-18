"""Streaming, hash-checked fixed-plan RGB; no altered time axis or resize."""
from io32 import *
def metadata(row):
 vs=load(ROOT/f'train_manifest_{row["domain"]}.json')['videos']
 return next(v for v in vs if v['filename' if row['domain']=='ulong' else 'name']==row['video'])
class PlanFrames:
 def __init__(self,row,device,full=False):
  import cv2
  cv2.setNumThreads(1)
  self.row=row;self.device=device;self.full=full;self.v=metadata(row);self.cap=None;self.next=None;self.geometry=None
  assert row['source_path']==self.v['path' if row['domain']=='ulong' else 'source_path']
  assert row['source_sha256']==self.v['sha256' if row['domain']=='ulong' else 'source_sha256']
  assert row['canonical_indices']==list(range(row['canonical_indices'][0],row['canonical_indices'][0]+64))
 def __len__(self):return 64
 def __getitem__(self,pos):
  if pos>=64:raise IndexError(pos)
  import cv2,numpy as np,torch
  from PIL import Image
  r=self.row;i=r['canonical_indices'][pos]
  if r['domain']=='ulong':
   if self.cap is None:self.cap=cv2.VideoCapture(self.v['path'])
   if self.next!=i:self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
   ok,a=self.cap.read();assert ok,(r['video'],i);self.next=i+1;a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
  else:
   p=Path(self.v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==self.v['frame_file_sha256'][i]
   with Image.open(p) as im:a=np.asarray(im,dtype=np.uint8).copy()
  assert hashlib.sha256(a.tobytes()).hexdigest()==r['full_RGB_sha256'][pos],('full hash mismatch',r['video'],i)
  crop=a[r['crop_y']:r['crop_y']+256,r['crop_x']:r['crop_x']+256].copy()
  assert crop.shape==(256,256,3) and hashlib.sha256(crop.tobytes()).hexdigest()==r['cropped_RGB_sha256'][pos]
  if self.geometry is None:self.geometry=a.shape
  assert a.shape==self.geometry and a.shape==(self.v.get('height',1080),self.v.get('width',1920),3),('NATIVE_GEOMETRY_MISMATCH',r['video'],a.shape,self.v.get('height'))
  return torch.from_numpy(a.copy() if self.full else crop).permute(2,0,1).unsqueeze(0).to(self.device).float()/255
 def close(self):
  if self.cap is not None:self.cap.release();self.cap=None
 def __del__(self):self.close()
def full_eligible(row):
 v=metadata(row)
 return v.get('width',1920)>0 and v.get('height',1080)>0
