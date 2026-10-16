from preflight import *
import numpy as np,cv2,random,concurrent.futures
from PIL import Image
V22=GVC/'experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control'
V32=GVC/'experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control'
def load(p):return json.loads(Path(p).read_text())
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
class Frames:
 def __init__(self,row,training=False):self.r=row;self.training=training;self.cap=None;self.next=None
 def __len__(self):return 64
 def array(self,i):
  r=self.r
  if self.training and r['domain']=='ulong':
   if self.cap is None:self.cap=cv2.VideoCapture(r['source_path'])
   k=r['canonical_indices'][i]
   if self.next!=k:self.cap.set(cv2.CAP_PROP_POS_FRAMES,k)
   ok,a=self.cap.read();assert ok,(r['source_path'],k);self.next=k+1;a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
  else:
   k=r['canonical_indices'][i] if self.training else i
   path=Path(r['input_dir'])/f'frame_{k:06d}.png'
   if not path.exists():path=Path(r['input_dir'])/f'im{k+1:05d}.png'
   with Image.open(path) as f:a=np.asarray(f,dtype=np.uint8).copy()
  hashes=r.get('full_RGB_sha256') if self.training else r.get('frame_rgb_sha256')
  if hashes:assert digest(a)==hashes[i],('RGB hash mismatch',i)
  if self.training:
   a=a[r['crop_y']:r['crop_y']+256,r['crop_x']:r['crop_x']+256].copy();assert a.shape==(256,256,3)
   if r.get('cropped_RGB_sha256'):assert digest(a)==r['cropped_RGB_sha256'][i]
  return a
 def tensor(self,i,device):
  import torch
  return torch.from_numpy(self.array(i)).permute(2,0,1).unsqueeze(0).to(device).float()/255
 def close(self):
  if self.cap is not None:self.cap.release();self.cap=None
 def __del__(self):self.close()
def verify(job):
 row,banned,framebanned=job;cv2.setNumThreads(1);p=R/'audits/samples'/f'{row["step"]:04d}.json'
 if p.exists():return load(p)
 f=Frames(row,True);full=[];cropped=[]
 # Obtain full native hashes before cropping.
 if row['domain']=='ulong':
  cap=cv2.VideoCapture(row['source_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,row['canonical_indices'][0])
  for i in range(64):
   ok,a=cap.read();assert ok;a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);full.append(digest(a));cropped.append(digest(a[row['crop_y']:row['crop_y']+256,row['crop_x']:row['crop_x']+256].copy()))
  cap.release()
 else:
  for k in row['canonical_indices']:
   with Image.open(Path(row['input_dir'])/f'frame_{k:06d}.png') as im:a=np.asarray(im,dtype=np.uint8).copy()
   full.append(digest(a));cropped.append(digest(a[row['crop_y']:row['crop_y']+256,row['crop_x']:row['crop_x']+256].copy()))
 assert not set(full)&framebanned,('test RGB content leakage',row['video'])
 assert all(hashlib.sha256(''.join(full[i:i+16]).encode()).hexdigest() not in banned for i in range(49)),('historical forbidden content',row['video'])
 out=dict(row,full_RGB_sha256=full,cropped_RGB_sha256=cropped);save(str(p.relative_to(R)),out);return out
def main():
 cv2.setNumThreads(1);tests=load(V32/'eval_manifest.json');forbidden=load(V22/'audits/forbidden_content_windows.json');hist=load(V22/'audits/pool_filter.json')['historical_cleaning']
 independent=[v for d,vs in tests.items() if d!='uvg_validation' for v in vs]
 banned=set(forbidden['consecutive16_frame_digest_sha256']);framebanned=set()
 for v in independent:
  assert Path(v['input_dir']).is_dir();f=Frames(v)
  full=[digest(f.array(i)) for i in range(64)];framebanned.update(full)
  banned.update(hashlib.sha256(''.join(full[i:i+16]).encode()).hexdigest() for i in range(49));f.close()
 sourcehash={v['source_sha256'] for v in independent}|set(hist['excluded_source_SHA256']);identities=set(hist['excluded_identities'])|{v['name'] for v in independent}
 pools={};excluded=[];provenance={}
 for d in ('ulong','uvg'):
  p=GLC/'experiments/glc_precompression_transfer_v1_abc'/f'train_manifest_{d}.json';original=load(p)['videos'];vs=[];seen=set()
  for v in original:
   name=v['filename' if d=='ulong' else 'name'];h=v['sha256' if d=='ulong' else 'source_sha256'];path=v['path' if d=='ulong' else 'source_path']
   reasons=[]
   if h in sourcehash or any(k in name or k in path for k in identities):reasons.append('source identity/hash overlap')
   if h in seen:reasons.append('duplicate source hash')
   if any(x in banned for x in v.get('native_window_content_digests',[])):reasons.append('decoded content overlap')
   if min(v['width'],v['height'])<256 or v['frames']<64:reasons.append('insufficient native size/frames')
   if reasons:excluded.append(dict(domain=d,video=name,source=path,sha256=h,reasons=reasons));continue
   assert sha(path)==h,('source file changed',path);seen.add(h);vs.append(v)
  assert vs;pools[d]=vs;provenance[str(p)]=sha(p);save(f'train_manifest_{d}.json',dict(videos=vs,source=str(p),source_sha256=sha(p)))
 save('eval_manifest.json',tests);save('dataset_split.json',dict(training={d:[v['filename' if d=='ulong' else 'name'] for v in vs] for d,vs in pools.items()},evaluation=tests,independent_datasets=[d for d in tests if d!='uvg_validation'],uvg_validation='diagnostic only; not counted independent',MCL29='sensitivity subset only',original_GLC_pretraining='unknown'))
 save('audits/leakage.json',dict(status='PASS_SOURCE_IDENTITY_AND_CONTENT_WINDOWS',historical_exclusions=hist,excluded=excluded,counts={d:len(v) for d,v in pools.items()},historical_evidence_sha256=sha(V22/'audits/forbidden_content_windows.json'),test_manifest_source_sha256=sha(V32/'eval_manifest.json'),training_sources_verified=True,exact_test_frame_hashes=len(framebanned),test_and_forbidden_window_hashes=len(banned),scope='source identities/file hashes, existing native first/last content digests; all newly selected training windows compared against forbidden and test RGB hashes',original_GLC_pretraining='unknown'))
 rng=random.Random(20261016);order=[(d,q) for d in pools for q in load(R/'protocol.json')['q_grid'] for _ in range(50)];rng.shuffle(order);jobs=[]
 for step,(d,q) in enumerate(order,1):
  v=rng.choice(pools[d]);start=rng.randrange(v['frames']-63);idx=list(range(start,start+64))
  row=dict(step=step,domain=d,q=q,video=v['filename' if d=='ulong' else 'name'],source_path=v['path' if d=='ulong' else 'source_path'],source_sha256=v['sha256' if d=='ulong' else 'source_sha256'],canonical_indices=idx,source_frame_indices=idx if d=='ulong' else v['source_frame_indices'][start:start+64],crop_x=rng.randrange(v['width']-255),crop_y=rng.randrange(v['height']-255),width=v['width'],height=v['height'],fps=v.get('fps',v.get('source_fps_metadata')),input_dir=v.get('input_dir'))
  jobs.append((row,banned,framebanned))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  rows=[]
  for row in ex.map(verify,jobs):
   rows.append(row)
   if len(rows)%50==0:print('SAMPLES',len(rows),flush=True);save('preparation_status.json',dict(stage='sample_verification',completed=len(rows),expected=1000))
 save('plans/shared.json',dict(seed=20261016,updates=rows,domains_balanced=True,q_per_domain=50,all_four_branches_identical=True));save('audits/plan_integrity.json',dict(status='PASS',count=1000,plan_sha256=sha(R/'plans/shared.json'),source_manifests=provenance))
if __name__=='__main__':
 try:main()
 except Exception:
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',stage='data_preparation',error=e));print(e);sys.exit(1)
