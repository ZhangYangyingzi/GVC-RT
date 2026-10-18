import csv,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]; EX=ROOT.parent
V34=EX/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500'; V33=EX/'gvcrt_neural_wrapper_v6_33_p_causal_rate_utility_ablation'; V24=EX/'gvcrt_neural_wrapper_v6_24_projected_adversarial_control'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvread(p): return list(csv.DictReader(p.open()))
def truth(v): return str(v).lower() in ('true','1')
def normalize_old(r,method,receipt,receipt_hash,feature_path,bitstream_path,source_path=''):
 b=int(r.get('real_bytes',r.get('real_RANS_bytes',0))); fps=float(r.get('rate_accounting_fps',r.get('original_rate_accounting_fps',0))); frames=int(r.get('frames',64)); H=W=0
 # Manifest lookup fills dimensions below.
 out=dict(dataset=r['dataset'],video=int(r['video_index'] if 'video_index' in r else r['video']),sequence=r.get('sequence',''),QP=int(r.get('QP',r.get('external_qp'))),
  method=method,adaptation_step=int(r.get('adaptation_step') or 0),actual_bpp=float(r.get('actual_bpp',r.get('bpp'))),real_bytes=b,total_bits=int(r.get('total_bits',b*8)),bits_per_frame=float(r.get('bits_per_frame',b*8/frames)),
  source_fps_metadata=float(r.get('source_fps_metadata',fps)),rate_accounting_fps=fps,kbps_20fps=float(r.get('kbps_20fps',b*8/frames*20/1000)),frames=frames,num_transitions=int(r.get('num_transitions',frames-1)),
  real_RANS=truth(r.get('real_RANS',True)),independent_decode_pass=truth(r.get('independent_decode_pass')),state_sync_pass=truth(r.get('state_sync_pass')),
  bitstream_path=str(bitstream_path),bitstream_sha256=sha(bitstream_path),feature_path=str(feature_path),feature_sha256=sha(feature_path),
  reconstruction_sha256=r.get('reconstruction_sha256',''),source_sha256=r.get('source_sha256',''),source_rgb_sha256=r.get('source_rgb_sha256',''),source_frame_indices=json.loads(r['source_frame_indices']) if isinstance(r.get('source_frame_indices'),str) else r.get('source_frame_indices'),
  LPIPS=float(r['LPIPS']),DISTS=float(r['DISTS']),FloLPIPS=float(r['FloLPIPS']),PSNR=float(r['PSNR']) if r.get('PSNR') not in (None,'') else None,SSIM=float(r['SSIM']) if r.get('SSIM') not in (None,'') else None,MS_SSIM=float(r['MS_SSIM']) if r.get('MS_SSIM') not in (None,'') else None,
  reuse=True,reuse_source_receipt=str(receipt),reuse_source_receipt_sha256=receipt_hash,protocol_source=r.get('protocol_sha256',''),actual_qps=json.loads(r['actual_qps']) if isinstance(r.get('actual_qps'),str) else r.get('actual_qps'))
 return out
# Source snapshots and fixed manifests.
v34=json.load(open(V34/'eval_manifest.json')); v33=json.load(open(V33/'eval_manifest.json'))
manifest={'ulong':v34['ulong'],'uvg_holdout':[v for v in v33['uvg_holdout'] if v['video_index'] in (2,3)],'hevc_b':v33['hevc_b']}
assert [v['name'] for v in manifest['uvg_holdout']]==['HoneyBee','Jockey']
assert len(manifest['ulong'])==6 and len(manifest['hevc_b'])==5
for ds,vs in manifest.items():
 for v in vs: assert v['frames']==64 and len(v['source_frame_indices'])==64 and len(v['frame_rgb_sha256'])==64 and v['width']==1920 and v['height']==1080
# ensure exact original manifests are unmodified by storing their content hashes
(ROOT/'eval_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Check official weights, trained inference checkpoints and source index records.
source_config=json.load(open(V34/'evaluation_config.json'))
source_idx={}; module_check={}
for old,new in [('N_noP_BG_FD','N_oursLoss_noP_500'),('P_PBG_FD','P_oursLoss_withP_500')]:
 ix=json.load(open(V34/'branches'/old/'checkpoint_index.json')); ent=ix.get('500')
 assert ent and ent['adaptation_step']==500
 cp=ent['inference']; assert Path(cp['path']).is_file() and sha(cp['path'])==cp['sha256']
 source_idx[new]=ix
 shutil.copy2(V34/'branches'/old/'checkpoint_index.json',ROOT/'branches'/new/'checkpoint_index.json')
 module_check[new]=dict(source_branch=old,step=500,train_checkpoint=ent.get('path'),train_checkpoint_sha256=ent.get('sha256'),inference_path=cp['path'],inference_sha256=cp['sha256'],module_hashes=cp['module_hashes'],deployment_receiver_hashes=ent['deployment_receiver_hashes'],optimizer_steps=ent.get('optimizer_internal_step'),EMA_counters=ent.get('EMA_counters'))
for f,expected in zip(source_config['source_model_files'],['c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e']):
 assert Path(f).is_file() and sha(f)==expected,(f,sha(f) if Path(f).exists() else 'missing')
source_hashes={'repository_HEAD':'d4e55e9d3b7de3e37245de0d81d05dae6eab10d1','v634b_commit':'a0e063a34f708453032df10acffecc9bf2f965e9','v634b_final_integrity_sha256':sha(V34/'final_integrity.json'),'v634b_eval_manifest_sha256':sha(V34/'eval_manifest.json'),'v633_eval_manifest_sha256':sha(V33/'eval_manifest.json'),'v634b_protocol_sha256':sha(V34/'effective_protocol.json'),'v634b_config_sha256':sha(V34/'config.json'),'v634b_plan_sha256':sha(V34/'plans/shared.json'),'expected_training_plan_sha256':'b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42','calibration_path':str(EX/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/calibration/final.json'),'calibration_sha256':sha(EX/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/calibration/final.json'),'official_model_files':[{ 'path':f,'sha256':sha(f),'bytes':Path(f).stat().st_size} for f in source_config['source_model_files']], 'checkpoints':module_check,'compression_hash':source_config['compression_hash'],'I_hash':source_config['I_hash'],'original_receiver_hashes':source_config['original_receiver_hashes'],'metric_module_hashes':source_config['metric_module_hashes'],'source_code':{str(p.relative_to(REPO)):sha(p) for p in [V34/'training32.py',V34/'distribution32.py',V34/'frames32.py',V34/'evaluate32.py',V34/'adapter32.py',V33/'train.py',V33/'evaluate.py'] if p.exists()}}
dump(ROOT/'source_hashes.json',source_hashes)
# Evaluation runtime config using canonical method labels.
methods={'original':[None,None],'N_oursLoss_noP_500':[None,'N_oursLoss_noP_500'],'P_oursLoss_withP_500':['P_oursLoss_withP_500','P_oursLoss_withP_500']}
checkpoints={}; deploy={}
for m,ix in source_idx.items():
 e=ix['500']; checkpoints[m]=e['inference'];deploy[m]=e['deployment_receiver_hashes']
evalcfg=dict(source_config);evalcfg.update(experiment='full_pipeline_gain_attribution_500',IP=-1,methods=methods,checkpoints=checkpoints,deployment_receiver_hashes=deploy,external_qps=[0,5,9],expected_points=117,datasets=['ulong','uvg_holdout','hevc_b'])
dump(ROOT/'evaluation_config.json',evalcfg)
# Historical point reuse. Each artifact is re-hashed and semantically matched to its immutable manifest row.
allrows=[]; reuse=[]; rowlookup={}
# 54 ULong rows, read from immutable task receipts and verify task + artifacts.
for m,oldm,branch in [('original','original',None),('N_oursLoss_noP_500','N_noP_BG_FD_500','N_noP_BG_FD'),('P_oursLoss_withP_500','P_PBG_FD_500','P_PBG_FD')]:
 for v in manifest['ulong']:
  for q in (0,5,9):
   rec=V34/'tasks'/f'adapt_ulong_{v["video_index"]}_{oldm}_{q}.json' if m!='original' else V34/'tasks'/f'baseline_ulong_{v["video_index"]}_original_{q}.json'
   if not rec.exists():
    # original receipts in v34b are baseline_*; adapt receipts for trained nodes.
    raise FileNotFoundError(rec)
   rj=json.load(open(rec));r=rj['result'];assert rj['status']=='PASS'
   assert int(r['QP'])==q and int(r['frames'])==64 and int(r['num_transitions'])==63 and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
   assert r.get('source_sha256',v['source_sha256'])==v['source_sha256'] and r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
   assert r['compression_hash_before']==r['compression_hash_after']==source_config['compression_hash']
   assert r['bitstream_sha256']==sha(r['bitstream_path']) and r['feature_sha256']==sha(r['feature_path'])
   assert Path(r['bitstream_path']).stat().st_size==int(r['real_bytes'])==int(r['bytes_consumed'])
   assert abs(float(r['bpp'])-int(r['real_bytes'])*8/(64*1920*1080))<1e-15
   out=normalize_old(r,m,rec,sha(rec),Path(r['feature_path']),Path(r['bitstream_path']))
   out['video']=v['video_index'];out['sequence']=v['name'];out['source_fps_metadata']=v['source_fps_metadata'];out['width']=1920;out['height']=1080;out['reuse']='v6.34b verified point';out['input_manifest_sha256']=sha(V34/'eval_manifest.json')
   if m!='original': assert r['loaded_receiver_module_hashes']==json.dumps(deploy[m],sort_keys=True) or r['runtime_audit']['receiver_hashes']==deploy[m]
   else: assert r['loaded_receiver_module_hashes']==source_config['original_receiver_hashes'] or json.loads(r['loaded_receiver_module_hashes'])==source_config['original_receiver_hashes']
   allrows.append(out);reuse.append(dict(dataset='ulong',video=v['video_index'],sequence=v['name'],method=m,QP=q,source_receipt=str(rec),source_receipt_sha256=sha(rec),bitstream_path=r['bitstream_path'],bitstream_sha256=r['bitstream_sha256'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256'],verified=True))
# 21 official original rows from v6.24: exact UVG and HEVC source/frame identity and all artifact hashes.
for ds in ('uvg_holdout','hevc_b'):
 csvp=V24/'results/ip_-1'/ds/'per_sequence_all_qp.csv'; rs=csvread(csvp)
 by={(int(r['video_index']),int(r['QP'])):r for r in rs if r['method']=='original'}
 for v in manifest[ds]:
  for q in (0,5,9):
   r=by[(v['video_index'],q)];assert r['source_sha256']==v['source_sha256'] and r['source_rgb_sha256']==v['rgb_sha256']
   assert json.loads(r['source_frame_indices'])==v['source_frame_indices'] and json.loads(r['actual_qps'])==json.load(open(ROOT/'qp_semantics_audit.json'))['actual_qps'][str(q)][:64]
   assert int(r['frames'])==64 and int(r['num_transitions'])==63 and truth(r['real_RANS']) and truth(r['independent_decode_pass']) and truth(r['state_sync_pass'])
   assert r['compression_hash_before']==r['compression_hash_after']==source_config['compression_hash']
   assert json.loads(r['loaded_receiver_module_hashes'])==source_config['original_receiver_hashes']
   bp=Path(r['bitstream_path']);fp=Path(r['feature_path']);rp=V24/'results/ip_-1'/ds/'per_sequence_all_qp.csv'
   assert bp.is_file() and fp.is_file() and sha(bp)==r['bitstream_sha256'] and sha(fp)==r['feature_sha256']
   assert int(r['real_bytes'])==int(r['bytes_consumed'])==bp.stat().st_size and abs(float(r['bpp'])-bp.stat().st_size*8/(64*1920*1080))<1e-15
   for k in ('frame_metrics','transitions'): assert sha(r[k+'_path'])==r[k+'_sha256']
   out=normalize_old(r,'original',csvp,sha(csvp),fp,bp);out.update(video=v['video_index'],sequence=v['name'],source_sha256=v['source_sha256'],source_fps_metadata=v.get('source_fps_metadata',v['rate_accounting_fps']),width=1920,height=1080,reuse='v6.24 verified original',input_manifest_sha256=sha(V33/'eval_manifest.json'))
   allrows.append(out);reuse.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method='original',QP=q,source_receipt=str(csvp),source_receipt_sha256=sha(csvp),bitstream_path=str(bp),bitstream_sha256=sha(bp),feature_path=str(fp),feature_sha256=sha(fp),verified=True))
# Existing point rows are the authoritative reuse set. Store a machine-readable provenance map.
assert len(allrows)==75,(len(allrows))
fields=list(dict.fromkeys(k for r in allrows for k in r));
with (ROOT/'results/all_rd_points.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(allrows)
dump(ROOT/'audits/historical_reuse.json',dict(status='PASS',verified_reuse_points=len(reuse),points=reuse,ULong_source='v6.34b',UVG_HEVC_original_source='v6.24 ip_-1 records',all_artifacts_rehashed=True))
# Fixed task queue for the only necessary new evaluations: N/P on UVG holdout and HEVC-B.
tasks=[]
for ds in ('uvg_holdout','hevc_b'):
 for v in manifest[ds]:
  for m,b in [('N_oursLoss_noP_500','N_oursLoss_noP_500'),('P_oursLoss_withP_500','P_oursLoss_withP_500')]:
   for q in (0,5,9): tasks.append(dict(stage='adapt',dataset=ds,video=v['video_index'],method=m,step=500,branch=b,qp=q))
dump(ROOT/'task_manifest.json',dict(tasks=tasks,expected_new=42,task_key=['method','checkpoint_sha256','dataset','video','QP','protocol_sha256']))
# Machine-readable fixed protocol and feasibility gate.
dump(ROOT/'config.json',dict(experiment='v6.35 full pipeline gain attribution, 500-step reuse-only plus fixed eval',repository_head='d4e55e9d3b7de3e37245de0d81d05dae6eab10d1',training={'N':{'steps':0,'reuse_step':500},'P':{'steps':0,'reuse_step':500},'A':{'steps':0,'status':'BLOCKED_MISSING_REFERENCE_OBJECTIVE'}},evaluation={'datasets':{'ulong':6,'uvg_holdout':2,'hevc_b':5},'qps':[0,5,9],'frames_per_video':64,'expected_logic_points':117,'historical_reuse_points':75,'new_real_encodes':42},gpu_policy={'allowed_physical':[4,5,6,7],'parallel':True,'observed_free_mib':{'4':7851,'5':6561,'6':34848,'7':34848},'evaluation_reservation_mib':9000},source_checkpoint_sha256={k:v['inference_sha256'] for k,v in module_check.items()}))
dump(ROOT/'protocol.json',dict(title='v6.35 fixed full-pipeline gain attribution',training='No new N/P training. Reuse only validated v6.34b step-500 checkpoints. A is omitted from execution because reference objective feasibility is blocked.',methods=['original','N_oursLoss_noP_500','P_oursLoss_withP_500'],datasets=['ulong','uvg_holdout','hevc_b'],videos=13,qps=[0,5,9],frames=64,IP=-1,real_RANS=True,independent_decode=True,rate='all persistent bitstream bytes*8/(64*W*H)',metrics=['LPIPS','DISTS','FloLPIPS','FID'],FID='pooled per dataset/method/QP, Inception2048, float64 ddof=1',training_objective='Reuse v6.34b effective complete objective including FD and legacy terms; no new loss branches.',source_manifest_hashes={'ulong':sha(V34/'eval_manifest.json'),'uvg_holdout_hevc_b':sha(V33/'eval_manifest.json')}))
missing=[
 {'item':'Official training implementation','status':'missing','evidence':'The official semcomm/GVC-RT README still lists training code release as TODO; no official training code is present in the repository snapshot.'},
 {'item':'Exact Stage-II/III lambda(q) table and normalization','status':'missing','evidence':'Paper gives a range/interpolation description but not a reproducible per-QP table and complete receiver-specific normalization.'},
 {'item':'Exact original latent teacher checkpoint/config/preprocessing','status':'missing','evidence':'No verifiable exact GVC-RT original LFQ teacher checkpoint and preprocessing package found; Open-MAGVIT2 proxy is not proven equivalent.'},
 {'item':'Exact margin loss definition/interface','status':'missing','evidence':'The paper names a margin term but the exact margin construction and tensor interface needed for implementation are unavailable in published training code.'},
 {'item':'Stage-III discriminator implementation/initialization/update rule','status':'missing','evidence':'No official implementation/config for discriminator and its training schedule is available.'},
 {'item':'Compatibility with this receiver-only frozen-compression adaptation','status':'unverified','evidence':'Published training objective may require codec/core optimization and modules not present in the fixed receiver-only comparison.'}
]
dump(ROOT/'reference_objective_feasibility.json',dict(status='BLOCKED_MISSING_REFERENCE_OBJECTIVE',branch='A_refLoss_noP_500',additional_updates=0,training_started=False,official_paper='https://arxiv.org/html/2608.04891v1',official_repository='https://github.com/semcomm/GVC-RT',checked_repository_snapshot='local source code available as of 2026-10-19',missing_requirements=missing,proxy_teacher_used=False,substitute_objective_created=False))
print('PREPARED',len(manifest['ulong']),len(manifest['uvg_holdout']),len(manifest['hevc_b']),'reuse',len(reuse),'new',len(tasks),'N/P hashes verified')
