"""Verify the complete bounded G11 delivery; no extra optimizer updates."""
from util import *
import torch,csv,concurrent.futures,inspect,gc
from model import load_cref,native_state,load_native,frozen_hashes,G11
from data import Data,cache_dir,Source
from runner import alive

def checkpoint_audit(done):
 step=done['committed_updates'];steps=[s for s in [0,50,100,200,300] if s<=step];out=[];expected_native=set(load(ROOT/'audits/preflight.json')['native_names'])
 for s in steps:
  p=ROOT/'checkpoints'/f'step_{s:04d}.pt';ck=read(p);assert ck['step']==ck['committed_updates']==ck['schedule_position']==s;assert ck['protocol_sha256']==protocol_sha() and ck['schedule_sha256']==sha(ROOT/'sample_schedule.json');assert set(ck['native_receiver'])==expected_native and sum(t.numel() for t in ck['native_receiver'].values())==988800;assert sum(t.numel() for t in ck['branch'].values())==1412251;assert set(['optimizer','RNG','mu','lambda_calibration','LR']).issubset(ck);assert torch.isfinite(ck['mu']).all() and ck['mu'].min()>=0 and ck['mu'].max()<=1
  if s>=50:assert ck['lambda_calibration']['status']=='PASS' and ck['lambda_calibration']['lambda_star']>0
  assert all(torch.isfinite(t).all() for t in list(ck['branch'].values())+list(ck['native_receiver'].values()));out.append(dict(step=s,path=str(p),bytes=p.stat().st_size,sha256=sha(p),native_parameters=988800,new_parameters=1412251,uploaded=False))
 assert read(ROOT/'checkpoints/latest.pt')['step']==step
 dump(ROOT/'checkpoint_manifest.json',dict(status='PASS',checkpoints=out,latest=str(ROOT/'checkpoints/latest.pt'),all_receiver_tensors_saved=True));return steps,out

def evaluation_audit(steps):
 proofs=[load(p) for p in (ROOT/'metrics/points').glob('*.json')];assert len(proofs)==6*(3+len(steps));source_ids={r['sequence'] for r in videos('validation')};grouped=collections.defaultdict(list);frame_count=pair_count=0
 for p in proofs:
  m=p['metrics'];assert p['status']=='PASS' and p['protocol_sha256']==protocol_sha() and m['sequence'] in source_ids and m['split']=='validation';assert m['frames']==15 and m['label'] in ['Original','B300','C_ref','G11'];assert m['label']!='G11' or m['step'] in steps
  for k in ['LPIPS','DISTS','RGB_MSE','highpass_signed_error_MSE','PSNR']:assert math.isfinite(m[k])
  name=f"{m['label']}_s{m['step']:04d}_{m['sequence']}";fr=list(csv.DictReader((ROOT/'metrics/frames'/f'{name}.csv').open()));pr=list(csv.DictReader((ROOT/'metrics/transitions'/f'{name}.csv').open()));assert len(fr)==15 and [int(r['frame']) for r in fr]==list(range(49,64));assert len(pr)==m['transitions']==14-p['reset_pairs_excluded'];frame_count+=15;pair_count+=len(pr);assert len({(r['from_frame'],r['to_frame']) for r in pr})==len(pr)
  for r in fr+pr:
   for key,val in r.items():
    if key in ['PSNR','RGB_MSE','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','highpass_signed_error_MSE','valid_fraction'] and val!='':assert math.isfinite(float(val))
  grouped[m['sequence']].append(p)
 for seq,ps in grouped.items():
  assert len(ps)==3+len(steps);assert len({p['stream_sha256'] for p in ps})==1;assert len({json.dumps(p['codec_feature_hashes']) for p in ps})==1;assert len({p['metrics']['bytes'] for p in ps})==len({p['metrics']['bpp'] for p in ps})==1
 for s in steps:
  d=load(ROOT/'evaluation'/f'decision_{s:04d}.json');assert d['status']=='PASS'
 return dict(points=len(proofs),frames=frame_count,transitions=pair_count,validation_sources=6,scored_frames=list(range(49,64)),no_final_test=True,shared_bitstreams_and_codec_features=True,no_duplicate_rows=True,finite_metrics=True)

def source_cache_audit():
 records=[]
 for row in videos():
  src=Source(row);dest=cache_dir(row['sequence']);cp=load(dest/'completion.json');assert cp['source_sha256']==row['source_sha256'] and src.info['source_row']['source_sha256']==row['source_sha256'];assert cp['stream_sha256']==src.info['stream_SHA256'];assert sha(src.info['stream_path'])==src.info['stream_SHA256'];assert cp['parent_metadata_sha256']==sha(PARENT/'cache/metadata'/f"{row['sequence']}.pt");assert cp['C_ref_branch_sha256']==load(ROOT/'weight_manifest.json')['D12_S_900']['sha256']
  for name,receipt in cp['files'].items():assert sha(dest/name)==receipt['sha256'] and (dest/name).stat().st_size==receipt['bytes']
  # Parent byte identities are checked against its own immutable completion receipts.
  for i in range(47,64):
   f=src.feature(i);assert thash([('f',f.half())])==load_tensor_feature_identity(dest,i);src.aux(i)
  records.append(dict(sequence=row['sequence'],split=row['split'],source_sha256=row['source_sha256'],stream_path=src.info['stream_path'],stream_sha256=src.info['stream_SHA256'],bytes=src.info['real_bytes'],bpp=src.info['bpp'],parent_metadata_sha256=cp['parent_metadata_sha256'],task_cache_manifest_sha256=sha(dest/'completion.json'),codec_feature_hashes=[load_tensor_feature_identity(dest,i) for i in range(47,64)]))
 dump(ROOT/'source_cache_manifest.json',dict(status='PASS',sources=records,train_sources=12,validation_sources=6,final_test_included=False));return records

def load_tensor_feature_identity(dest,i):
 d=torch.load(dest/f'frame_{i:02d}.pt',map_location='cpu',weights_only=True,mmap=True);return d['codec_feature_sha256']

def large_manifest():
 ps=[p for base in ['cache','checkpoints','videos'] for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ['.pt','.mkv','.mp4','.bin','.npz','.png']]
 def entry(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,reproduction='REPRODUCE.md; frozen protocol, schedule, source and weight manifests')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(entry,ps))
 dump(ROOT/'large_artifacts.json',dict(large_artifacts_uploaded=False,files=rows,total_bytes=sum(p['bytes'] for p in rows)));return len(rows)

def main():
 setup_torch();assert load(ROOT/'audits/C_ref_parent_replay.json')['status']=='PASS';done=load(ROOT/'completion.json');assert done['status']=='COMPLETE';assert done['committed_updates'] in [100,200,300];heartbeat('FINAL_VERIFY_CHECKPOINTS');steps,cps=checkpoint_audit(done);check=evaluation_audit(steps)
 c=load_cref();initial=native_state(c);cp=read(ROOT/'checkpoints'/f"step_{done['committed_updates']:04d}.pt");load_native(c,cp['native_receiver']);assert frozen_hashes(c)==done['frozen_hashes'];changed=[n for n,t in native_state(c).items() if not torch.equal(t,initial[n])];assert changed;assert not any('lora' in n for n in changed);del c,cp;gc.collect();torch.cuda.empty_cache();heartbeat('MEASURING_COMPONENT_COST')
 from performance import run
 run();heartbeat('VERIFYING_CACHE_IDENTITIES');sources=source_cache_audit()
 from report import tables,plots,write
 pts,agg,paired=tables();figs=plots(pts,agg);heartbeat('DELIVERING_LOSSLESS_MEDIA')
 from media import deliver
 vv=deliver();write(pts,agg,figs);heartbeat('HASHING_LOCAL_LARGE_ARTIFACTS');large=large_manifest()
 # Record the actual imported numerical dependencies, not just Git snapshots.
 import torchvision,lpips,DISTS_pytorch,importlib.metadata
 import DISTS_pytorch.DISTS_pt as dists_impl
 deps={}
 for name,module in [('torch',torch),('torchvision',torchvision),('lpips',lpips),('DISTS_pytorch',DISTS_pytorch),('DISTS_implementation',dists_impl)]:
  path=Path(module.__file__);deps[name]=dict(path=str(path),sha256=sha(path),version=str(getattr(module,'__version__','not exposed')))
 binary=Path(torchvision.__file__).parent/'_C.so'
 if binary.exists():deps['torchvision_C_extension']=dict(path=str(binary),sha256=sha(binary))
 dump(ROOT/'runtime_dependency_manifest.json',dict(status='PASS',dependencies=deps,device=device_info(),quality_weights='Per-evaluation metric_hashes; actual LPIPS/DISTS/PWC/FloLPIPS weights',RAFT_weight=load(ROOT/'weight_manifest.json')['RAFT']))
 # Copy actual sources, never assume Git HEAD represents this audited runtime.
 code={str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('*.py'),*ROOT.glob('vendor/*.py'),*ROOT.glob('runtime_tree/**/*.py'),*ROOT.glob('original_tree/**/*.py')]};dump(ROOT/'source_implementation_final.json',dict(code=code,protocol_sha256=protocol_sha(),parent_snapshot=str(ROOT/'dependency_snapshot.json'),training_per_checkpoint_implementation_retained=True))
 inspection=load(ROOT/'audits/runtime_inspection.json');pre=load(ROOT/'audits/preflight.json');summary=dict(status='LOCAL_COMPLETE_GITHUB_PENDING',complete=False,experiment_directory=str(ROOT),C_ref=dict(chain=load(ROOT/'PROTOCOL.json')['base_chain'],complete_sha256=pre['C_ref_complete_sha256'],D12_S_checkpoint_sha256=load(ROOT/'weight_manifest.json')['D12_S_900']['sha256']),trainable=dict(new_parameters=1412251,native_receiver_parameters=988800,total=2401051,native_names=inspection['native_names']),GPU=int(done['device']['physical_GPU']),committed_updates=done['committed_updates'],stop_reason=done['stop_reason'],validation={str(s):dict(completed_sources=6,planned_sources=6) for s in steps},evaluations=check,report_path=str(ROOT/'REPORT.md'),metrics_path=str(ROOT/'metrics'),figures_path=str(ROOT/'figures'),video_index_path=str(ROOT/'VIDEO_INDEX.md'),frozen='PASS excluding intentional native receiver update',bitstream='PASS',publication_status=str(ROOT/'publication_status.json'),remaining=['GitHub commit/push/remote verification'],resume_command=PYTHON+' -B '+str(ROOT/'manage.py')+' resume');dump(ROOT/'summary.json',summary)
 dump(ROOT/'publication_status.json',dict(status='LOCAL_COMPLETE_GITHUB_PENDING',branch='codex/g11-detail-temporal-pilot-20261113',remote_verified=False,large_artifacts_uploaded=False))
 dump(ROOT/'final_integrity.json',dict(status='PASS',local_complete=True,protocol_sha256=protocol_sha(),preflight_pass=True,**check,steps=steps,committed_updates=done['committed_updates'],stop_reason=done['stop_reason'],all_expected_checkpoints_present=True,native_receiver_checkpointed=True,changed_only_whitelist=True,changed_native_tensors=changed,frozen_excluding_whitelist_unchanged=True,codec_and_Bridge_unchanged=True,TCG_and_LoRA_frozen=True,reference_independent=True,step0_RGB_max_abs=pre['step0_RGB_max_abs'],cache_sources=len(sources),lossless_videos=len(vv),lossless_RGB8_roundtrip_all=True,large_artifacts=large,scientific_conclusion_not_made=True,time=stamp()))
 heartbeat('LOCAL_COMPLETE_GITHUB_PENDING',committed_updates=done['committed_updates'])
if __name__=='__main__':
 with lock('finalizer.lock'):
  try:main()
  except BaseException:
   import traceback
   dump(ROOT/'receipts/finalize.error.json',dict(status='LOCAL_DELIVERY_ERROR',error=traceback.format_exc(),time=stamp()));raise
