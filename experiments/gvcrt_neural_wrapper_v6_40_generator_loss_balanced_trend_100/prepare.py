"""Generate and freeze the balanced fixed-window plan; audit immutable source bytes."""
from io32 import *
import copy,random,collections
OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_38_bridge_generator_factorial_trend'
INIT=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
EXPECTED='18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074'
def main():
 import numpy as np,torch,cv2
 from PIL import Image
 cv2.setNumThreads(1)
 assert sha(INIT)==EXPECTED
 source=torch.load(INIT,map_location='cpu',weights_only=True)
 assert all(tensor_hash(source[k])==source['initial_module_hashes'][k] for k in ['wrapper','bridge','generator'])
 prior=torch.load(OLD/'branches/G/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)
 assert all(tensor_hash(prior[k])==tensor_hash(source[k]) for k in ['wrapper','bridge','generator'])
 assert set(prior['buffers'])<=set(source['buffers'])
 assert all(torch.equal(v,source['buffers'][m][k]) for m,bs in prior['buffers'].items() for k,v in bs.items())
 assert torch.equal(source['crop_rng']['torch'],prior['crop_rng']['torch']) and torch.equal(source['crop_rng']['cuda'],prior['crop_rng']['cuda'])
 dump(ROOT/'audits/initialization_source.json',dict(status='PASS',path=str(INIT),sha256=EXPECTED,bytes=INIT.stat().st_size,model_tensors=source['initial_module_hashes'],v638_step0_tensor_buffer_RNG_equal=True,source_step=0,no_step100_resume=True))
 del prior,source
 historical=load(OLD/'plans/first100.json');first={}
 for item in historical['updates']:
  row=item['source_row'];first.setdefault(row['domain'],item)
 assert set(first)=={'uvg','ulong'} and first['uvg']['source_row']['video']=='Beauty'
 assert first['ulong']['source_row']['video']=='clips_long_1920/6bea102c-a661-4195-b089-d12022eac35d.mp4'
 rng=random.Random(20261034);crops={};window_audit=[];arrays={}
 for domain,item in first.items():
  row=item['source_row'];manifest=load(ROOT/f'train_manifest_{domain}.json')['videos'];meta=next(v for v in manifest if v['filename' if domain=='ulong' else 'name']==row['video'])
  assert sha(row['source_path'])==row['source_sha256'];frames=[];files=[]
  cap=cv2.VideoCapture(row['source_path']) if domain=='ulong' else None
  if cap is not None:cap.set(cv2.CAP_PROP_POS_FRAMES,row['canonical_indices'][0])
  for pos,i in enumerate(row['canonical_indices']):
   if cap is not None:
    ok,a=cap.read();assert ok;a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
   else:
    p=Path(meta['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==meta['frame_file_sha256'][i];a=np.asarray(Image.open(p).convert('RGB')).copy();files.append(dict(path=str(p),sha256=sha(p),source_frame_index=row['source_frame_indices'][pos]))
   assert hashlib.sha256(a.tobytes()).hexdigest()==row['full_RGB_sha256'][pos];frames.append(a)
  if cap is not None:cap.release()
  arrays[domain]=frames;h,w=frames[0].shape[:2];used=set();crops[domain]=[]
  while len(used)<50:
   xy=(rng.randrange(w-256+1),rng.randrange(h-256+1))
   if xy not in used:used.add(xy);crops[domain].append(xy)
  window_audit.append(dict(domain=domain,video=row['video'],source_path=row['source_path'],source_sha256=row['source_sha256'],source_bytes=Path(row['source_path']).stat().st_size,native_geometry=[w,h],canonical_indices=row['canonical_indices'],source_frame_indices=row['source_frame_indices'],full_RGB_sha256=row['full_RGB_sha256'],canonical_files=files,distinct_time_windows=1,distinct_spatial_crops=50,source_v638_step=item['step']))
 # Explicit 50 paired units, each QP exactly five times. RNG is local, never training RNG.
 units=[q for q in range(10) for _ in range(5)];rng.shuffle(units);updates=[];counts=collections.Counter();vqp=collections.Counter();crop_index=collections.Counter()
 for unit,q in enumerate(units,1):
  order=['uvg','ulong'];rng.shuffle(order)
  for domain in order:
   row=copy.deepcopy(first[domain]['source_row']);ix=crop_index[domain];crop_index[domain]+=1;x,y=crops[domain][ix];step=len(updates)+1
   row.update(adaptation_step=step,crop_x=x,crop_y=y,external_qp=q,actual_qps=[q+[0,2,1][[0,1,0,2,0,2,0,2][i%8]] for i in range(49,64)],actual_frame_qps=[q if i==0 else q+[0,2,1][[0,1,0,2,0,2,0,2][i%8]] for i in range(64)],cropped_RGB_sha256=[hashlib.sha256(a[y:y+256,x:x+256].copy().tobytes()).hexdigest() for a in arrays[domain]])
   digest=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   updates.append(dict(step=step,pair=unit,source_plan_step=first[domain]['step'],source_row=row,source_row_canonical_sha256=digest,source='new balanced crop/QP record on fixed first historical window'))
   counts[domain]+=1;vqp[(domain,q)]+=1
 assert counts=={'uvg':50,'ulong':50} and all(vqp[(d,q)]==5 for d in counts for q in range(10))
 for d in counts:assert len({(u['source_row']['crop_x'],u['source_row']['crop_y']) for u in updates if u['source_row']['domain']==d})==50
 plan=dict(experiment=ROOT.name,seed=20261034,local_RNG=True,historical_plan_path=str(OLD/'plans/first100.json'),historical_plan_sha256=sha(OLD/'plans/first100.json'),fixed_window_per_video=True,independent_time_windows=2,paired_units=50,updates=updates)
 p=ROOT/'plans/first100.json'
 if p.exists():assert load(p)==plan,'Frozen plan mismatch'
 else:dump(p,plan)
 dump(ROOT/'plans/counts.json',dict(status='PASS',plan_sha256=sha(p),video_counts=dict(counts),video_qp_counts={d:{str(q):vqp[(d,q)] for q in range(10)} for d in counts},unique_crops_per_video={d:50 for d in counts},distinct_time_windows_per_video={d:1 for d in counts}))
 dump(ROOT/'audits/training_windows.json',dict(status='PASS',windows=window_audit))
 streams=load(ROOT/'manifests/streams.json');assert len(streams)==6
 for x in streams:
  p=Path(x['bitstream_path']);assert p.stat().st_size==int(x['real_bytes']) and sha(p)==x['bitstream_sha256']
 dump(ROOT/'audits/input_streams.json',dict(status='PASS',streams=[dict(path=x['bitstream_path'],bytes=int(x['real_bytes']),sha256=x['bitstream_sha256']) for x in streams]))
 isolation=load(OLD/'audits/data_isolation.json');assert isolation['status']=='PASS'
 train_hashes={w['source_sha256'] for w in window_audit};test_hashes={v['source_sha256'] for v in load(ROOT/'eval_manifest.json')['hevc_b'] if v['video_index'] in (0,1)};assert not train_hashes & test_hashes
 dump(ROOT/'audits/data_isolation.json',dict(status='PASS',training_identities=[w['video'] for w in window_audit],evaluation_identities=['BasketballDrive','BQTerrace'],source_hash_overlap=[],inherited_path=str(OLD/'audits/data_isolation.json'),inherited_sha256=sha(OLD/'audits/data_isolation.json'),official_pretraining_exposure='unknown; not excluded'))
 dependencies=load(OLD/'manifests/code_dependencies.json');dependencies.update(load(OLD/'source_hashes.json'))
 for n in ['README.md','plans/first100.json','protocol.json','OBJECTIVE_COMPARISON.md','train.py','training32.py','training35.py','frames32.py','adapter35.py','evaluate.py','audits/checkpoint_step0_tensor_equality.json','audits/data_isolation.json']:dependencies[str(OLD/n)]=sha(OLD/n)
 dependencies[str(INIT)]=EXPECTED
 for n,h in dependencies.items():assert sha(n)==h,(n,'historical dependency changed')
 dump(ROOT/'source_hashes.json',dependencies);dump(ROOT/'manifests/code_dependencies.json',dependencies)
 dump(ROOT/'audits/preflight.json',dict(status='PASS',plan_sha256=sha(ROOT/'plans/first100.json'),source_sha256=EXPECTED,source_dependencies=len(dependencies),fixed_conditions=['O_original','G_basic','G_dists'],expected_training_updates=200,expected_results=18))
 dump(ROOT/'publication_status.json',dict(status='NOT_PUBLISHED',push_success=False))
 print('PREPARE PASS',dict(counts),dict(vqp),flush=True)
if __name__=='__main__':main()
