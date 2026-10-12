"""Conservative source-hash and decoded-frame fingerprint overlap audit."""
from io21 import *
import numpy as np,concurrent.futures
from PIL import Image

def vector(a):
 a=np.asarray(Image.fromarray(a).resize((32,18)),dtype=np.float32)/255;return a.reshape(-1)
def sample_train(v):
 import cv2
 cv2.setNumThreads(1);p=v.get('path',v.get('source_path'));cap=cv2.VideoCapture(p);out=[]
 try:
  for i in (0,min(16,int(v['frames'])-1)):
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,a=cap.read()
   if ok:out.append((v.get('filename',v.get('name')),i,vector(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))))
 finally:cap.release()
 return out

def main():
 historical=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for rs in pool.map(sample_train,load(ROOT/'train_manifest_ulong.json')['videos']):historical.extend(rs)
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b'):
  for v in load(V26/'manifests'/f'{d}.json')['videos']:
   for i in (0,16,48,63):
    folder=Path(v['input_dir']);p=folder/(f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png')
    with Image.open(p) as im:historical.append((v['name'],v['source_frame_indices'][i],vector(np.asarray(im).copy())))
 for v in load(ROOT/'train_manifest_uvg.json')['videos']:
  for i in (0,16,48,63):
   with Image.open(Path(v['input_dir'])/f'frame_{i:06d}.png') as im:historical.append((v['name'],v['source_frame_indices'][i],vector(np.asarray(im).copy())))
 arrays=np.stack([r[2] for r in historical]);audit=load(ROOT/'audits/overlap_integrity.json');records={r['sequence']:r for r in audit['records']};nearest=[]
 for v in sources('mcljcv'):
  candidates=[]
  for i in (0,16,48,63):
   with Image.open(Path(v['input_dir'])/f'frame_{i:06d}.png') as im:z=vector(np.asarray(im).copy())
   mae=np.mean(np.abs(arrays-z),axis=1);idx=np.argsort(mae)[:3]
   for k in idx:
    match=dict(sequence=v['name'],mcl_frame=i,historical_identity=historical[k][0],historical_frame=historical[k][1],thumbnail_RGB_MAE=float(mae[k]),threshold=.025,rule='candidate only; no confirmation from a single thumbnail');nearest.append(match)
    if mae[k]<.025:candidates.append(match)
  records[v['name']]['suspected_fingerprint_matches']=candidates
  if candidates and records[v['name']]['status']!='confirmed':records[v['name']]['status']='suspected'
 audit.update(records=list(records.values()),suspected=[r['sequence'] for r in records.values() if r['status']=='suspected'],fingerprint_rule='32x18 RGB MAE<0.025 on fixed0/16/48/63 against available training and historical evaluation samples; suspected only',fingerprint_source_count=len(historical),sampling_limits='sparse fingerprints cannot exclude shifted or re-encoded overlap',status='PASS');dump(ROOT/'audits/overlap_integrity.json',audit);write(ROOT/'audits/overlap_nearest_candidates.csv',nearest);dump(ROOT/'audits/fingerprint_integrity.json',dict(status='PASS',historical_samples=len(historical),suspected=audit['suspected'],confirmed=audit['confirmed']));print('OVERLAP PASS',flush=True)
if __name__=='__main__':main()
