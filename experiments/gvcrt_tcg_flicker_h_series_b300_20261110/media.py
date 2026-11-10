"""Full-resolution FFV1 fixed development and historically exposed person_dog clips."""
from common_s2 import *
from execution_policy import h2_training_allowed
import numpy as np
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def ffmpeg_probe():
 from videos import open_lossless
 base=ROOT/'tmp/ffmpeg_probe';base.mkdir(parents=True,exist_ok=True);args=['ffmpeg','-v','error','-nostdin','-y']
 for i in range(2):
  p=base/f'{i}.mkv';w=open_lossless(p,16,16,30);w.stdin.write(np.full((2,16,16,3),i*50,np.uint8).tobytes());w.stdin.close();assert w.wait()==0;args+=['-i',str(p)]
 label=base/'label.txt';label.write_text('model step100 QP5 frame %{n}')
 graph=f'[0:v]pad=16:64:0:48:black,drawtext=fontfile={FONT}:textfile={label}:fontcolor=white:fontsize=8:x=0:y=0:start_number=48[a];[1:v]pad=16:64:0:48:black[b];[a][b]xstack=inputs=2:layout=0_0|16_0:shortest=1[out]'
 dest=base/'stack.mkv';subprocess.run(args+['-filter_complex_threads','1','-filter_complex',graph,'-map','[out]','-c:v','ffv1','-pix_fmt','bgr0','-threads','1',str(dest)],check=True);dump(ROOT/'audits/ffmpeg_compatibility.json',dict(status='PASS',graph=graph,unsupported_fill_omitted=True,SHA256=sha(dest),time=time.time()))

def cases():
 profiles=load(ROOT/'manifests/long_videos.json');out=[];rois=load(ROOT/'visual_ROIs.json')
 for seq in ['24747b35-04ca-416c-b1d0-0bc670fbac99','28a7c428-1a8d-4bd1-afbc-fed596ec6153']:
  profile=next(v for v in profiles if v['sequence']==seq);row=dict(profile['row']);row.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)));out.append(dict(sequence=seq,row=row,frames=128,fps=row['source_fps'],ROIs={'fixed_texture':rois[seq]},historical_exposure='development',selection_used=False))
 person=ORIGINAL_EVAL/'video/static_camera_examples_20261109/ULong_person_dog_QP5';sel=load(person/'selection.json');v=sel['video'];assert v['sequence']=='dce599d3-d196-4d87-b2b1-698acdbbb3d4';assert v['source_frame_indices']==list(range(128,192));roi={k:dict(x=z[0],y=z[1],width=z[2],height=z[3],first=0,last=63) for k,z in sel['ROIs'].items()};out.append(dict(sequence=v['sequence'],row=v,frames=64,fps=str(v['fps']),ROIs=roi,historical_exposure='historically viewed holdout; video only; excluded from training/calibration/selection',selection_used=False,source_identification=str(person/'selection.json'),source_identification_SHA256=sha(person/'selection.json')));return out

def export(case):
 import torch
 from cache_codec import cache_paths
 from pipeline import wait_rows
 from evaluation import Evaluator
 from experiment_model import reconstruct
 from model_context import assert_reset
 from videos import open_lossless
 row=case['row'];wait_rows([row],False,5);key,p,rp=cache_paths(row,5,False)
 with lock('locks/lease_'+key+'.lock',blocking=True):
  assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);info=d['info'];h,w=info['native_shape'];assert info['frames']==case['frames'];models=['GT','Original','A2000','B300',GROUPS[0],GROUPS[1]]
  if h2_training_allowed():models.append(GROUPS[2])
  ev=Evaluator(2);records=[]
  for model in models:
   step=100 if model in GROUPS else 300 if model=='B300' else 2000 if model=='A2000' else 0;dest=ROOT/f'media/native/{case["sequence"]}_QP5_{model}_step{step:04d}.mkv';manifest=ROOT/f'media/manifests/{case["sequence"]}_{model}.json'
   if manifest.exists():
    a=load(manifest);assert sha(a['path'])==a['SHA256'] and a['stream_SHA256']==info['stream_SHA256'];records.append(a);continue
   c=ev.context(model,step) if model not in ['GT','Original'] else None
   if c:assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
   temporary=dest.with_name(dest.stem+'.partial.mkv');writer=open_lossless(temporary,w,h,case['fps']);hashes=[];first=None
   with torch.no_grad():
    for i in range(info['frames']):
     if model=='GT':rgb=d['GT'][i].numpy().tobytes()
     else:
      assert (None if i==0 else tensor_sha(d['features'][i]))==info['codec_trace'][i]['codec_feature_SHA256'];raw=d['original'][i].to(c['device'] if c else 'cpu') if i==0 or c is None else reconstruct(c,d['features'][i],info['actual_qps'][i]);rgb=(((raw.float().clamp(-1,1)+1)/2)[0,:,:h,:w].permute(1,2,0)*255).round().clamp(0,255).byte().cpu().numpy().tobytes()
     if first is None:first=rgb
     writer.stdin.write(rgb);hashes.append(hashlib.sha256(rgb).hexdigest())
   writer.stdin.close();assert writer.wait()==0;temporary.replace(dest)
   decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(dest),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);assert decoded==first
   if c:assert before==tensor_hash(c['pm'].state_dict().items());assert_reset(c)
   a=dict(status='PASS',sequence=case['sequence'],model=model,step=step,QP=5,s=.5 if model not in ['GT','Original'] else None,frames=info['frames'],source_frame_indices=row['source_frame_indices'],width=w,height=h,fps=case['fps'],path=str(dest),SHA256=sha(dest),bytes=dest.stat().st_size,stream_SHA256=info['stream_SHA256'],stream_bytes=info['real_bytes'],codec_trace_SHA256=digest(info['codec_trace']),source_SHA256=row['source_sha256'],RGB8_frame_hashes=hashes,pixel_format='FFV1 bgr0 / exact RGB24 first-frame roundtrip',historical_exposure=case['historical_exposure'],used_for_training_or_selection=False)
   dump(manifest,a);records.append(a)
   ev.contexts.clear();del c;torch.cuda.empty_cache()
  return records

def roi_comparison(case,records,name,roi,anonymous=False):
 x,y,w,h=[roi[k] for k in ['x','y','width','height']];first,last=roi['first'],roi['last'];n=last-first+1;labelmode='anonymous' if anonymous else 'named';dest=ROOT/f'media/ROI/{case["sequence"]}_{name}_{labelmode}_QP5_FFV1.mkv';proof=ROOT/f'media/manifests/{case["sequence"]}_{name}_{labelmode}.json'
 if proof.exists():a=load(proof);assert sha(a['path'])==a['SHA256'];return a
 items=list(records);mapping={}
 if anonymous:
  rng=random.Random(202611102+int(digest(case['sequence']+name)[:8],16));rng.shuffle(items);mapping={chr(65+i):dict(model=r['model'],step=r['step'],source_SHA256=r['SHA256']) for i,r in enumerate(items)}
 args=['ffmpeg','-v','error','-nostdin','-y'];filters=[]
 for i,r in enumerate(items):
  args+=['-i',r['path']];label=ROOT/f'tmp/media_label_{i}.txt';identity=chr(65+i) if anonymous else LABELS.get(r['model'],r['model'])+f' step{r["step"]}';start=case['row']['source_frame_indices'][first];label.write_text(f'{identity}\nQP5 source frame %{{n}}\nROI ({x},{y},{w},{h})')
  filters.append(f'[{i}:v]trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y},pad={w}:{h+48}:0:48:black,drawtext=fontfile={FONT}:textfile={label}:fontcolor=white:fontsize=9:x=3:y=3:start_number={start}[v{i}]')
 layout='|'.join(f'{i*w}_0' for i in range(len(items)));filters.append(''.join(f'[v{i}]' for i in range(len(items)))+f'xstack=inputs={len(items)}:layout={layout}:shortest=1[out]');dest.parent.mkdir(parents=True,exist_ok=True);temporary=dest.with_name(dest.stem+'.partial.mkv');subprocess.run(args+['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(temporary)],check=True);temporary.replace(dest)
 raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(dest),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);grid=np.frombuffer(raw,np.uint8).reshape((h+48,len(items)*w,3))
 for i,r in enumerate(items):
  raw=subprocess.check_output(['ffmpeg','-v','error','-i',r['path'],'-vf',f'trim=start_frame={first}:end_frame={first+1},crop={w}:{h}:{x}:{y}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);assert np.array_equal(grid[48:,i*w:(i+1)*w],np.frombuffer(raw,np.uint8).reshape((h,w,3)))
 a=dict(status='PASS',path=str(dest),SHA256=sha(dest),sequence=case['sequence'],ROI_name=name,ROI=dict(x=x,y=y,width=w,height=h),source_frame_range=[case['row']['source_frame_indices'][first],case['row']['source_frame_indices'][last]],frames=n,fps=case['fps'],scale=1,pixels_unchanged=True,labels_outside_original_ROI=True,anonymous=anonymous,subjective_test_claimed=False)
 if anonymous:
  p=ROOT/f'media/anonymous_mapping/{case["sequence"]}_{name}.json';dump(p,dict(status='MAPPING_ONLY',mapping=mapping));a['mapping_path']=str(p);a['mapping_SHA256']=sha(p)
 dump(proof,a);return a

def main():
 env();assert load(ROOT/'selection.json')['status']=='FROZEN';ffmpeg_probe();clips=[];comparisons=[]
 for case in cases():
  rec=export(case);clips+=rec
  for name,roi in case['ROIs'].items():
   for anon in [False,True]:comparisons.append(roi_comparison(case,rec,name,roi,anon))
  dump(ROOT/'audits/media_progress.json',dict(status='RUNNING',native_clips=len(clips),comparisons=len(comparisons),time=time.time()))
 dump(ROOT/'video_manifest.json',dict(status='PASS',native_clips=clips,ROI_comparisons=comparisons,H2_missing_reason=None if h2_training_allowed() else 'H2_MECHANICAL_CHECK_FAILED; not replaced by another model',anonymous_labels_are_material_only=True,subjective_experiment_completed=False,all_paths_absolute=True,time=time.time()))
if __name__=='__main__':main()
