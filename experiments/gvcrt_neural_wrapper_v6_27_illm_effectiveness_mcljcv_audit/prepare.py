"""Freeze source identities, stratified diagnostic panels and full-resolution RGB pools."""
from io21 import *
import random,copy,concurrent.futures,re,zipfile
import numpy as np
from PIL import Image
DATA=Path('/Huang_group/zyyz/datasets/MCL-JCV')
PLAN_SHA='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
LABEL_SHA='3f58659bf2449b17f1f7eaf5ee64fd64d281da2f640d46148536ea6183eb9183'
def diagnostic_plan():
 rng=random.Random(20261027);shared=load(ROOT/'plans/shared.json')['updates'];seen=[];unseen=[];missing=[]
 for d in ('ulong','uvg'):
  for q in range(10):
   candidates=[r for r in shared if r['domain']==d and r['external_qp']==q];rng.shuffle(candidates);chosen=[candidates[0]]
   chosen.append(next((r for r in candidates[1:] if r['video']!=chosen[0]['video']),candidates[1]))
   for j,row in enumerate(chosen):seen.append(dict(copy.deepcopy(row),panel='seen',panel_id=f'seen_{d}_q{q}_{j}',original_plan_step=row['adaptation_step']))
 for d in ('ulong','uvg'):
  manifest=load(ROOT/f'train_manifest_{d}.json')['videos'];used={r['video'] for r in shared if r['domain']==d}
  for q in range(10):
   candidates=[v for v in manifest if v['filename' if d=='ulong' else 'name'] not in used and int(v['frames'])>=64];rng.shuffle(candidates)
   if not candidates:
    missing.append(dict(domain=d,QP=q,status='unavailable',reason='All permitted UVG canonical frame pools contain only64 frames and were included in shared plan; no non-overlapping64-frame interval exists' if d=='uvg' else 'no permitted unused source'));continue
   v=candidates[0];start=rng.randrange(int(v['frames'])-63);x=rng.randrange(int(v['width'])-255);y=rng.randrange(int(v['height'])-255)
   row=dict(panel='unseen',panel_id=f'unseen_{d}_q{q}_0',domain=d,video=v['filename'],source_path=v['path'],source_sha256=v['sha256'],canonical_indices=list(range(start,start+64)),source_frame_indices=list(range(start,start+64)),crop_x=x,crop_y=y,crop_size=256,external_qp=q,actual_qps=load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][49:64],supervised_window_positions=list(range(49,64)),normalization_denominator=15)
   import cv2
   cv2.setNumThreads(1);cap=cv2.VideoCapture(v['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,start);full=[];crop=[]
   try:
    for i in range(64):
     ok,a=cap.read();assert ok;(a)=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);full.append(hashlib.sha256(a.tobytes()).hexdigest());crop.append(hashlib.sha256(a[y:y+256,x:x+256].copy().tobytes()).hexdigest())
   finally:cap.release()
   row.update(full_RGB_sha256=full,cropped_RGB_sha256=crop);assert all(not (r['domain']==d and r['video']==row['video'] and set(r['source_frame_indices'])&set(row['source_frame_indices'])) for r in shared);unseen.append(row)
 rows=seen+unseen
 for i,r in enumerate(rows):r['adaptation_step']=i+1
 for r in rows:
  candidates=[s for s in rows if s['domain']==r['domain'] and s['external_qp']==r['external_qp'] and s['panel_id']!=r['panel_id']];r['shuffle_panel_id']=next((s['panel_id'] for s in candidates if s['video']!=r['video']),candidates[0]['panel_id'])
 dump(ROOT/'diagnostic_plan.json',dict(seed=20261027,shared_plan_SHA256=PLAN_SHA,seen_requested=40,seen_available=len(seen),unseen_requested=20,unseen_available=len(unseen),unavailable=missing,updates=rows,unseen_semantics='plan-window unseen only, video may have historical exposure',excluded_datasets=['hevc_b','uvg_holdout','uvg_validation','mcljcv']))
 dump(ROOT/'audits/diagnostic_sampling.json',dict(status='PASS',seen=len(seen),unseen=len(unseen),unavailable=missing,source_interval_disjoint=True,training_pool_only=True,seed=20261027,plan_sha256=sha(ROOT/'diagnostic_plan.json')))
def mcl_one(p):
 name=p.stem;m=re.fullmatch(r'videoSRC(\d+)_1920x1080_(24|25|30)',name);assert m
 size=p.stat().st_size;frame_bytes=1920*1080*3//2;assert size%frame_bytes==0 and size//frame_bytes>=64
 directory=ROOT/'cache/mcl_rgb'/name;directory.mkdir(parents=True,exist_ok=True)
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-f','rawvideo','-pixel_format','yuv420p','-video_size','1920x1080','-framerate',m[2],'-i',str(p),'-vf','scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=limited:out_range=full','-frames:v','64','-threads','1','-pix_fmt','rgb24','-start_number','0',str(directory/'frame_%06d.png')]
 if not all((directory/f'frame_{i:06d}.png').exists() for i in range(64)):subprocess.run(cmd,check=True)
 whole=hashlib.sha256();files=[];rgbs=[];stats=[]
 for i in range(64):
  f=directory/f'frame_{i:06d}.png'
  with Image.open(f) as im:assert im.mode=='RGB' and im.size==(1920,1080);a=np.asarray(im).copy()
  whole.update(a.tobytes());files.append(sha(f));rgbs.append(hashlib.sha256(a.tobytes()).hexdigest())
  if i in (0,16,48,63):stats.append(dict(frame=i,min=int(a.min()),max=int(a.max()),RGB_means=a.mean((0,1)).tolist(),fingerprint=hashlib.sha256(np.asarray(Image.fromarray(a).resize((32,32))).tobytes()).hexdigest()))
 out=dict(dataset='mcljcv',video_index=int(m[1])-1,name=name,input_dir=str(directory),source_path=str(p),source_sha256=sha(p),source_bytes=size,source_total_frames=size//frame_bytes,frames=64,width=1920,height=1080,source_fps_metadata=int(m[2]),rate_accounting_fps=int(m[2]),temporal_stride=1,source_frame_indices=list(range(64)),frame_file_sha256=files,frame_rgb_sha256=rgbs,rgb_sha256=whole.hexdigest(),pixel_format='yuv420p',bit_depth=8,source_kind='original YUV_source archive member',conversion_command=cmd,color_matrix='BT709',source_range_convention='limited',RGB_range='full',raw_color_tags='absent, historical conversion convention fixed',readability=True,color_channel_checks=stats)
 dump(ROOT/'manifests/individual'/f'{name}.json',out);print('MCL RGB READY',name,flush=True);return out

def main():
 assert sha(ROOT/'plans/shared.json')==PLAN_SHA;cfg=load(ROOT/'config.json');assert sha(cfg['source_checkpoint']['path'])==EXPECTED
 from illm_runtime import WEIGHT
 assert sha(WEIGHT)==LABEL_SHA
 deps=load(V26/'audits/dependencies.json');extra=[V26/'config.json',V26/'protocol.json',V26/'plans/shared.json',WEIGHT,*[V26/n for n in ('io21.py','illm_runtime.py','window_training.py','train.py','adapter.py','evaluate.py','qp_semantics_audit.json')]]
 checkpoint=[]
 import torch
 for b in BRANCHES:
  ix=load(V26/'branches'/b/'checkpoint_index.json');extra.append(V26/'branches'/b/'checkpoint_index.json');extra.append(V26/'branches'/b/'training_logs/train.jsonl');extra.append(V26/'branches'/b/'gradient_summary.csv')
  for step in (0,250,500,1000):
   item=ix[str(step)];assert sha(item['path'])==item['sha256'];extra.append(Path(item['path']))
   if b=='C_decoder_illm':
    s=torch.load(item['path'],map_location='cpu',weights_only=True);assert s['adaptation_step']==step and s['source_checkpoint_sha256']==EXPECTED and s['plan_sha256']==PLAN_SHA and 'projected' in s
    assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==item['module_hashes'];checkpoint.append(dict(step=step,**item,D_state_hash=tensor_hash(s['projected']),labeler_weight_sha256=s['labeler_weight_sha256'],D_updates=s['D_updates']));del s
 for p in extra:deps[str(p)]=sha(p)
 dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',C_training_checkpoints=checkpoint,source_B1_sha256=EXPECTED,shared_plan_SHA256=PLAN_SHA,labeler_SHA256=LABEL_SHA,no_missing_D=True,official_commit='3f122808255a2278eea049119367eb3abc434862'))
 diagnostic_plan()
 hashes={p:sha(ROOT/p) for p in ['config.json','protocol.json','plans/shared.json','diagnostic_plan.json','qp_semantics_audit.json']};dump(ROOT/'audits/protocol_hashes.json',hashes)
 dump(ROOT/'audits/diagnostic_preparation.json',dict(status='PASS',dependency_files=len(deps),no_training=True))
 entries={}
 for z in DATA.glob('*.zip'):
  with zipfile.ZipFile(z) as archive:
   for i in archive.infolist():
    if '/1080P/YUV_source/' in i.filename and i.filename.endswith('.yuv'):entries[Path(i.filename).name]=dict(archive=str(z),member=i.filename,uncompressed_bytes=i.file_size,CRC32=i.CRC)
 paths=sorted((DATA/'1080p_source').glob('*.yuv'));assert paths
 for p in paths:assert p.name in entries and p.stat().st_size==entries[p.name]['uncompressed_bytes']
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:videos=list(pool.map(mcl_one,paths))
 videos.sort(key=lambda v:v['video_index']);dump(ROOT/'manifests/mcljcv.json',dict(videos=videos,valid_N=len(videos),expected_points=len(videos)*50,archive_provenance=entries))
 # Exact source / decoded RGB overlap against available training and historical test records.
 histories=[]
 for p in (ROOT/'train_manifest_ulong.json',ROOT/'train_manifest_uvg.json',*[V26/'manifests'/f'{d}.json' for d in ('ulong','uvg_holdout','hevc_b','uvg_validation')]):
  for v in load(p)['videos']:histories.append(dict(manifest=str(p),identity=v.get('name',v.get('filename')),source_sha=v.get('source_sha256',v.get('sha256')),frame_hashes=v.get('frame_rgb_sha256',[])))
 overlap=[]
 for v in videos:
  matches=[h for h in histories if h['source_sha']==v['source_sha256'] or set(h['frame_hashes'])&set(v['frame_rgb_sha256'])]
  overlap.append(dict(sequence=v['name'],status='confirmed' if matches else 'no_confirmed_overlap_in_available_exact_evidence',matches=matches,unknown_pretraining_exposure=True))
 dump(ROOT/'audits/overlap_integrity.json',dict(status='PASS',records=overlap,confirmed=[r['sequence'] for r in overlap if r['status']=='confirmed'],limitations=['exact hashes cannot exclude re-encoded or unknown historical content','official pretraining exposure unknown'],full_and_confirmed_excluded_reporting=True))
 dump(ROOT/'audits/dataset_integrity.json',dict(status='PASS',valid_N=len(videos),expected_points=len(videos)*50,frames=64,source_indices=list(range(64)),pixel_format='yuv420p',bit_depth=8,source_identity='original YUV_source',byte_count_and_archive_member_verified=True,native_fps_from_filename=True,color_matrix='BT709',source_range='limited convention (untagged raw metadata)',RGB_range='full',ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],sources=['https://mcl.usc.edu/mcl-jcv-dataset/','https://github.com/microsoft/DCVC/blob/main/DCVC-family/test_conditions.md'],readability=True,manifest_sha256=sha(ROOT/'manifests/mcljcv.json')))
 dump(ROOT/'audits/data_preparation.json',dict(status='PASS'));print('PREPARATION PASS',flush=True)
if __name__=='__main__':main()
