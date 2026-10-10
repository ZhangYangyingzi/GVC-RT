"""Read-only historical verification; exact RGB extension and objective-source audit."""
from io21 import *
import re, shutil, traceback
B=ROOT.parent/'gvcrt_neural_wrapper_v6_21b_gvcrt_latent_interface_audit'
def bd(name,v):
 p=B/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def objective_search():
 hits=[];files=[]
 for p in sorted(REPO.rglob('*.py')):
  if any(x in p.parts for x in ('cache','__pycache__','.git')) or p.is_relative_to(ROOT) or p.is_relative_to(B):continue
  txt=p.read_text(errors='replace');matches=[]
  for i,line in enumerate(txt.splitlines(),1):
   if re.search(r'cosine_similarity|cosine_embedding|L_cos\b|L_margin\b|margin_loss|cos_loss',line,re.I):matches.append(dict(line=i,text=line))
  if p.is_relative_to(REPO/'src') or matches:files.append(dict(path=str(p),sha256=sha(p)))
  if matches:hits.append(dict(path=str(p),sha256=sha(p),matches=matches))
 # The public codec distribution supplies inference modules, not the original training objective.
 codec_hits=[h for h in hits if Path(h['path']).is_relative_to(REPO/'src')]
 assert not codec_hits,('original codec loss candidate needs inspection',codec_hits)
 bd('config.json',dict(no_training=True,seed=20261010,methods=list(load(V20/'config.json')['methods']),external_qps=[0,4,9],frames=64,expected_points=240))
 bd('protocol.json',dict(no_training=True,no_proxy_replacement=True,original_loss_required=True,checkpoint_source=str(V20),status='BLOCKED',missing_definitions=['original GVC-RT cosine alignment training loss','original GVC-RT margin training loss']))
 bd('audits/gvcrt_objective_source.json',dict(status='BLOCKED',search_root=str(REPO),searched_python_files=len(list(REPO.rglob('*.py'))),source_files=files,candidate_hits=hits,original_codec_hits=codec_hits,exact_formulas=None,exact_tensor_names=None,reason='Original GVC-RT training cosine/margin objective implementation is absent from the available codec source. Adaptation/proxy and teacher LFQ losses are not substituted.'))
 (B/'audits/gvcrt_objective_source_excerpt.txt').write_text('\n'.join(h['path']+'\n'+'\n'.join(str(m['line'])+': '+m['text'] for m in h['matches']) for h in hits)+'\n')
 for name in ['gvcrt_objective_integrity.json','latent_probe_integrity.json','codec_integrity.json','evaluation_integrity.json']:
  bd(name,dict(status='BLOCKED',no_training=True,reason='Exact original loss definition unavailable; no probe or evaluation using proxy losses.'))
 bd('blocking_error.json',dict(status='BLOCKED',reason='Original GVC-RT cosine alignment and margin training loss definitions unavailable in repository source',no_proxy_replacement=True,training_performed=False))
 bd('final_integrity.json',dict(status='BLOCKED',no_training=True,no_proxy_replacement=True,exact_gvcrt_alignment_loss_reused=False,completed_points=0,expected_points=240,blocking_error='blocking_error.json'))
 (B/'README_NUMBERS_ONLY.md').write_text('# Protocol\n\nNo training. Exact original GVC-RT loss required.\n\n| Item | Status |\n|---|---|\n| Original cosine/margin objective source | BLOCKED |\n| Completed points | 0 / 240 |\n\nFiles: audits/gvcrt_objective_source.json, audits/gvcrt_objective_source_excerpt.txt, blocking_error.json.\n')
def checkpoints():
 import torch
 cfg=load(V20/'evaluation/config.json');records=[]
 for method,branch in [('native_i_adapt_1000','native_i_adapt'),('wrapped_i_control_1000','wrapped_i_control'),('B_standard','B'),('B_i_bypass','B')]:
  origin=V18 if branch=='B' else V20;ix=load(origin/'branches'/branch/'checkpoint_index.json');r=ix.get('checkpoints',ix)['1000'];assert sha(r['path'])==r['sha256']
  s=torch.load(r['path'],map_location='cpu',weights_only=True);mh={k:tensor_hash(s[k])for k in ('wrapper','bridge','generator')};assert mh==r['module_hashes'];assert r['compression_hash']==cfg['compression_hash'];del s
  deploy=cfg['checkpoints'][method];assert sha(deploy['path'])==deploy['sha256'] and deploy['module_hashes']==mh
  records.append(dict(method=method,full_checkpoint_absolute_path=r['path'],full_checkpoint_sha256=r['sha256'],module_hashes=mh,compression_hash=r['compression_hash'],I_hash=cfg['I_hash'],deployment=deploy,index_path=str(origin/'branches'/branch/'checkpoint_index.json'),index_sha256=sha(origin/'branches'/branch/'checkpoint_index.json')))
 audit=dict(status='PASS',records=records,original=dict(compression_hash=cfg['compression_hash'],I_hash=cfg['I_hash'],receiver_hashes=cfg['original_receiver_hashes']),source_final_integrity=sha(V20/'final_integrity.json'))
 dump(ROOT/'checkpoint_integrity.json',audit);bd('checkpoint_integrity.json',audit)
 (ROOT/'evaluation').mkdir(exist_ok=True);dump(ROOT/'evaluation/config.json',cfg)
def extend(v):
 import numpy as np
 p=ROOT/'cache'/v['dataset']/v['name'];p.mkdir(parents=True,exist_ok=True);out=p/'rgb.raw';meta=p/'source.json'
 if meta.exists():
  d=load(meta);assert sha(out)==d['raw_rgb_sha256'];return d
 assert sha(v['source_path'])==v['source_sha256']
 ix=v['source_frame_indices'];stride=ix[1]-ix[0];assert all(y-x==stride for x,y in zip(ix,ix[1:]));start=ix[0]
 if v['dataset']=='ulong':
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',v['source_path']],text=True));total=int(probe['streams'][0]['nb_read_frames'])
  cmd=['ffmpeg','-v','error','-threads','1','-i',v['source_path'],'-vsync','0','-frames:v',str(min(total,start+128*stride)),'-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
 else:
  total=Path(v['source_path']).stat().st_size//(1920*1080*3//2)
  cmd=load(ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb/ffmpeg_conversion_audit.json')['inherited_command'];cmd=list(cmd);cmd[cmd.index('-i')+1]=v['source_path'];cmd[cmd.index('-framerate')+1]=str(v.get('source_fps_metadata',v['rate_accounting_fps']));cmd[cmd.index('-frames:v')+1]=str(min(total,start+128*stride))
 indices=list(range(start,min(total,start+128*stride),stride));assert len(indices)>=64 and indices[:64]==ix
 command(cmd);h=hashlib.sha256();h64=hashlib.sha256();hs=[];target=set(indices)
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=(p/'ffmpeg.log').open('w'))
 with out.with_suffix('.tmp').open('wb') as f:
  for i in range(indices[-1]+1):
   raw=proc.stdout.read(1920*1080*3);assert len(raw)==1920*1080*3,('short source',v['name'],i)
   if i in target:
    j=len(hs);digest=hashlib.sha256(raw).hexdigest()
    if j<64:assert digest==v['frame_rgb_sha256'][j],('RGB mismatch',v['name'],j);h64.update(raw)
    f.write(raw);h.update(raw);hs.append(digest)
  # Read unused trailing frames from the bounded historical conversion command.
  while proc.stdout.read(1<<20):pass
 assert proc.wait()==0 and h64.hexdigest()==v['rgb_sha256'];out.with_suffix('.tmp').replace(out)
 d={**v,'frames':len(indices),'source_frame_indices':indices,'frame_rgb_sha256':hs,'rgb_sha256':h.hexdigest(),'raw_rgb_path':str(out),'raw_rgb_sha256':sha(out),'historical_64_rgb_sha256':v['rgb_sha256'],'temporal_stride':stride,'source_total_frames':total,'target_frames':128,'maximum_legal_length_used':len(indices)<128,'conversion_command':cmd}
 dump(meta,d);print('SOURCE',v['dataset'],v['name'],len(indices),flush=True);return d

def main():
 if not (ROOT/'audits/historical_inventory.json').exists():dump(ROOT/'audits/historical_inventory.json',historical_inventory())
 objective_search();checkpoints()
 cfg=dict(no_training=True,seed=20261010,methods=METHODS,external_qps=list(range(10)),expected_points=480,target_frames=128,bootstrap_repeats=1000,sample_repeats=100,sample_sizes=[32,64,128,256,512,1024],source=str(V20),source_config_sha256=sha(V20/'evaluation/config.json'))
 dump(ROOT/'config.json',cfg);dump(ROOT/'protocol.json',dict(**cfg,reuse_historical_runtime=True,no_teacher_forcing=True,first_I_modes=load(V20/'evaluation/config.json')['i_frame_modes'],sample_rule='sequence-balanced counts; uniform without replacement inside sequence; same frame identities for GT and all methods/QPs',bootstrap_rule='within-sequence full-length sampling with replacement; paired GT and all methods and QPs; dataset-specific deterministic seed',FID='exact float64 sample-covariance low-rank FID',KID='unbiased polynomial MMD squared k(x,y)=(1+x dot y/2048)^3',equal_rate='three-method common measured bpp interval,100 uniform ln(bpp),PCHIP,no extrapolation',padding=[1920,1088],metric_crop=[1920,1080],codec_dtype='float16',wrapper_dtype='float32',force_zero_thres=.12))
 records=[]
 for d in DATASETS:
  vs=[extend(v)for v in load(V20/'manifests'/f'{d}.json')['videos']];dump(ROOT/'manifests'/f'{d}.json',dict(videos=vs,source_manifest=str(V20/'manifests'/f'{d}.json'),source_manifest_sha256=sha(V20/'manifests'/f'{d}.json')));records.extend(vs)
 split=load(V20/'dataset_split.json');assert split['training_source_disjoint'];train=load(V20/'train_manifest_ulong.json')['videos'];eval_ids={r['name']for r in records};assert not eval_ids.intersection({v.get('video_id',v.get('name',''))for v in train})
 audit=dict(status='PASS',videos=records,training_source_disjoint=True,source_split_path=str(V20/'dataset_split.json'),source_split_sha256=sha(V20/'dataset_split.json'),historical_pretraining_exposure='not claimed absent; retained historical audit records',expected_points=480)
 dump(ROOT/'dataset_integrity.json',audit);bd('dataset_integrity.json',audit)
 deps=load(V20/'audits/dependencies.json');deps.update({str(V20/x):sha(V20/x)for x in ['adapter.py','io20.py','evaluation/config.json','config.json','protocol.json','final_integrity.json']});dump(ROOT/'audits/dependencies.json',deps)
 dump(ROOT/'audits/protocol_hashes.json',{x:sha(ROOT/x)for x in ['config.json','protocol.json','evaluation/config.json',*[f'manifests/{d}.json'for d in DATASETS]]})
 dump(ROOT/'feature_extractor_integrity.json',dict(status='PASS',same_as_v620=True,metric_module_hashes=load(V20/'evaluation/config.json')['metric_module_hashes'],dependencies=deps,no_model_download=True))
 print('PREPARE PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'blocking_error.json',dict(status='FAIL',error=traceback.format_exc()));raise
