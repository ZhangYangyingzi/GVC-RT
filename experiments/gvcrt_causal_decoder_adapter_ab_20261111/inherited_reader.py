"""Native RGB/window reader, SHA checks and bounded task-private ZIP leases."""
from common_s2 import *
P=PARENT
import numpy as np,zipfile
_VERIFIED_SOURCES={}
_GEOMETRY_ERRATA=None
def expected_native_shape(row):
 global _GEOMETRY_ERRATA
 declared=(row.get('height',1080),row.get('width',1920),3)
 if _GEOMETRY_ERRATA is None:
  path=ROOT/'vendor/sealed/repairs/native_geometry_v1/geometry_errata.json'
  assert sha(path)=='08ad3f8416354b1141583fdb54e652392a1af5562efb249d2d255e6df83fa18f','NATIVE_GEOMETRY_ERRATA_CHANGED'
  record=load(path)
  assert sha(HIST/'inventory/qualified.json')==record['inventory_sha256']
  _GEOMETRY_ERRATA=record['entries']
 correction=_GEOMETRY_ERRATA.get(row['source_sha256'])
 if correction is None:return declared
 actual=tuple(correction['audited_native_shape'])
 assert correction['domain']==row['domain'] and correction['video']==row['video']
 if declared!=actual:
  assert row.get('window_verification')=='trusted_parent_reuse' and declared==tuple(correction['declared_shape']),'UNREGISTERED_GEOMETRY_METADATA_MISMATCH'
 return actual
@contextlib.contextmanager
def source_file(row):
 s=row['source_path']
 if not s.startswith('zip://'):
  p=Path(s)
  if not p.is_file():raise FileNotFoundError(s)
  stat=p.stat();stamp=(stat.st_size,stat.st_mtime_ns,row['source_sha256'])
  if _VERIFIED_SOURCES.get(s)!=stamp:
   assert sha(p)==row['source_sha256'],('SOURCE_FILE_CHANGED',s)
   _VERIFIED_SOURCES[s]=stamp
  yield p;return
 archive,entry=s[6:].split('::',1);cache=ROOT/'cache/media';cache.mkdir(exist_ok=True)
 key=row['source_sha256'];p=cache/(key+'.mp4')
 with (cache/(key+'.lock')).open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if not p.exists():
   with lock('cache_quota.lock',blocking=True):
    disk();used=sum(x.stat().st_size for x in cache.glob('*.mp4'))
    size=int(row['source_size_bytes'])
    if size>20*2**30:raise RuntimeError('MEDIA_EXCEEDS_PRIVATE_CACHE_WORKER_CAP')
    for old in sorted(cache.glob('*.mp4'),key=lambda x:x.stat().st_mtime):
     if used+size<=30*2**30:break
     with old.with_suffix('.lock').open('a') as lease:
      try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:continue
      used-=old.stat().st_size;old.unlink()
    if used+size>30*2**30:raise RuntimeError('PRIVATE_CACHE_CAPACITY_BLOCKED')
    tmp=p.with_suffix('.tmp')
    with zipfile.ZipFile(ARCHIVES/archive) as z,z.open(entry) as src,tmp.open('wb') as out:
     shutil.copyfileobj(src,out,8*2**20)
    assert tmp.stat().st_size==size and sha(tmp)==key
    tmp.replace(p)
  os.utime(p,None)
  fcntl.flock(f,fcntl.LOCK_SH)
  yield p
def read_arrays(row,full=False,verify=True):
 import cv2
 from PIL import Image
 cv2.setNumThreads(1);cap=None;results=[];expected_shape=expected_native_shape(row)
 with source_file(row) if row['domain']=='ulong' else contextlib.nullcontext(None) as path:
  try:
   if path is not None:
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,row['canonical_indices'][0])
   for pos,index in enumerate(row['canonical_indices']):
    if cap is not None:
     ok,a=cap.read();assert ok,(row['video'],index);a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    else:
     m=load(P/'plans/train_manifest_uvg.local.json')['videos'];v=next(x for x in m if x['name']==row['video'])
     pathpng=Path(v['input_dir'])/f'frame_{index:06d}.png';assert sha(pathpng)==v['frame_file_sha256'][index]
     with Image.open(pathpng) as image:a=np.asarray(image,dtype=np.uint8).copy()
    assert a.shape==expected_shape,('AUDITED_NATIVE_GEOMETRY_MISMATCH',row['video'],a.shape,expected_shape)
    crop=a[row['crop_y']:row['crop_y']+256,row['crop_x']:row['crop_x']+256].copy();assert crop.shape==(256,256,3)
    if verify:
     assert hashlib.sha256(a.tobytes()).hexdigest()==row['full_RGB_sha256'][pos]
     assert hashlib.sha256(crop.tobytes()).hexdigest()==row['cropped_RGB_sha256'][pos]
    results.append(a.copy() if full else crop)
  finally:
   if cap is not None:cap.release()
 return results
class PlanFrames:
 def __init__(self,row,device,full=False):
  import torch
  self.device=device;self.arrays=read_arrays(row,full);self.full=full
 def __len__(self):return 64
 def __getitem__(self,i):
  import torch
  if i>=64:raise IndexError(i)
  return torch.from_numpy(self.arrays[i]).permute(2,0,1).unsqueeze(0).to(self.device).float()/255
 def close(self):self.arrays=None
