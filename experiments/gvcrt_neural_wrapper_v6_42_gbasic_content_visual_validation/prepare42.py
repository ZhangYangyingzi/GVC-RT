from common42 import *
from PIL import Image
import numpy as np

def main():
 protocol=dict(experiment=ROOT.name,methods=list(METHODS),updates=0,checkpoint=str(CP),checkpoint_sha256=CP_SHA,
  datasets={'hevc_b':['Kimono1','ParkScene'],'uvg_holdout':['HoneyBee','Jockey']},QPs=[0,5,9],frames=64,
  IP=-1,native_I=True,padding=[1920,1088],output=[1920,1080],force_zero_thres=.12,
  metric_source='v6.40 float output',PSNR='-10log10(global RGB MSE)',actual_bpp='full bytes*8/(64*1920*1080)',
  FID=False,BD_rate=False,FPS_performance=False,allowed_GPUs=[4,5,6,7])
 if (ROOT/'protocol.json').exists():assert load(ROOT/'protocol.json')==protocol
 else:dump(ROOT/'protocol.json',protocol)
 assert sha(CP)==CP_SHA
 fin=load(REF/'branches/G_basic/final_integrity.json');assert fin['committed_updates']==fin['optimizer_internal_step']==100 and fin['active_modules']==['generator']
 assert fin['inference_checkpoint_sha256']==CP_SHA
 import torch
 cp=torch.load(CP,map_location='cpu',weights_only=True)
 assert tensor_hash(cp['bridge'])==fin['initial_module_hashes']['bridge']==fin['final_module_hashes']['bridge']
 assert tensor_hash(cp['generator'])==fin['final_module_hashes']['generator']
 dump(ROOT/'model_manifest.json',dict(status='PASS',checkpoint=str(CP),sha256=CP_SHA,bytes=CP.stat().st_size,
  step100_record_sha256=sha(REF/'branches/G_basic/receipts/100.json'),final_integrity_sha256=sha(REF/'branches/G_basic/final_integrity.json'),
  training_hashes=fin,original_deployment=load(ROOT/'evaluation_config.json'),only_generator_differs=True,optimizer_updates=0))
 src=load(V37/'eval_manifest.json');selected=[];source_index=[]
 for ds,names in protocol['datasets'].items():
  for name in names:
   v=next(x for x in src[ds] if x['name']==name);h=hashlib.sha256()
   for i in range(64):
    p=frame_path(v,i);a=np.asarray(Image.open(p).convert('RGB'));assert a.shape==(1080,1920,3)
    assert sha(p)==v['frame_png_sha256' if ds=='hevc_b' else 'frame_file_sha256'][i]
    assert hashlib.sha256(a.tobytes()).hexdigest()==v['frame_rgb_sha256'][i];h.update(a.tobytes())
    source_index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
   assert h.hexdigest()==v['rgb_sha256'];selected.append(v)
 dump(ROOT/'data_manifest.json',selected);dump(ROOT/'source_rgb_index.json',source_index)
 train=load(REF/'plans/first100.json')['updates'];train_names={r['source_row']['video'] for r in train};train_hash={r['source_row']['source_sha256'] for r in train}
 assert not train_names.intersection(v['name'] for v in selected) and not train_hash.intersection(v['source_sha256'] for v in selected)
 dump(ROOT/'audits/data_isolation.json',dict(status='PASS',train_identities=sorted(train_names),test_identities=[v['name'] for v in selected],
  official_pretraining_exposure='unknown',historical_evaluation_exposure='may have been evaluated in prior experiments',plan_sha256=sha(REF/'plans/first100.json')))
 originals=read(V37/'results/reused_points.csv');streams=[]
 for v in selected:
  for q in (0,5,9):
   r=next(r for r in originals if r['dataset']==v['dataset'] and int(r['video'])==v['video_index'] and int(r['QP'])==q and r['method']=='O_original')
   assert r['source_rgb_sha256']==v['rgb_sha256'] and json.loads(r['source_frame_indices'])==v['source_frame_indices']
   p=Path(r['bitstream_path']);assert p.stat().st_size==int(r['real_bytes']) and sha(p)==r['bitstream_sha256']
   streams.append(dict(dataset=v['dataset'],sequence=v['name'],QP=q,path=str(p),bytes=p.stat().st_size,sha256=sha(p),
    actual_qps=json.loads(r['actual_qps']),source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],
    original_reconstruction_sha256=r['reconstruction_sha256'],provenance=r,reference_replay_task=str(V37/'tasks'/f"replay_{v['dataset']}_{v['video_index']}_q{q}.json")))
 dump(ROOT/'stream_manifest.json',streams)
 # Coordinates were selected from GT frame32 only, before any reconstruction evaluation.
 rois={'Kimono1':[(520,180,'subject head and clothing edge'),(1200,300,'tree branches and foliage')],
  'ParkScene':[(800,560,'distant people and path'),(1350,250,'foliage and pavilion edges')],
  'HoneyBee':[(780,420,'flower and insect region'),(1300,180,'flower texture against background')],
  'Jockey':[(1320,180,'rider and horse head'),(400,540,'railing and hedge texture')]}
 visual=dict(selection='GT-only frame32 inspection before reconstruction; fixed thereafter',QPs=[0,9],frames=[16,32,48,63],
  PNG_rule='float RGB cropped to 1080x1920; clamp[0,1]; multiply255; round; uint8 RGB; identical for all models',
  metric_rule='original floating output; never PNG',overview='uniform 1/3 resize for all models',ROI_scale='1:1 pixels',
  sequences=[dict(dataset=v['dataset'],sequence=v['name'],fps=v['rate_accounting_fps'],source_frame_indices=v['source_frame_indices'],
   GT_selection_frame=32,GT_selection_sha256=v['frame_rgb_sha256'][32],rois=[dict(x=x,y=y,width=256,height=256,reason=reason) for x,y,reason in rois[v['name']]]) for v in selected])
 if (ROOT/'visual_manifest.json').exists():assert load(ROOT/'visual_manifest.json')==visual
 else:dump(ROOT/'visual_manifest.json',visual)
 dump(ROOT/'audits/preflight.json',dict(status='PASS',visual_manifest_sha256=sha(ROOT/'visual_manifest.json'),
  data_manifest_sha256=sha(ROOT/'data_manifest.json'),stream_manifest_sha256=sha(ROOT/'stream_manifest.json'),
  fixed_before_reconstruction=True,optimizer_updates=0,reference_eval_config_identical=sha(ROOT/'evaluation_config.json')==sha(REF/'evaluation_config.json')))
 dump(ROOT/'run_status.json',dict(status='PREPARED',optimizer_updates=0,completed=0,planned=24))
 print('PREPARE_PASS',flush=True)
if __name__=='__main__':main()
