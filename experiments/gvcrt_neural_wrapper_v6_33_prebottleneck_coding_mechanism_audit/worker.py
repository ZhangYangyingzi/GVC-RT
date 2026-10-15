"""Resumable video/QP group; all seven methods, fixed receiver and real evaluator."""
import argparse,gc,time,traceback,fcntl,resource
from audit_io import *
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
ORDER=['B1_alpha0','Original','B1_alpha025','B1_alpha050','B1_alpha075','B1_alpha100','B1_alpha125']
def key(t):return f"{t['dataset']}/{t['video']:02d}/q{t['qp']}/{t['method']}"
def code_hashes():return {n:sha(ROOT/n) for n in ('audit_io.py','proxy_runtime.py','coding_hooks.py','worker.py')}
def valid(t,deep=False,sanity=False):
 p=ROOT/('sanity/points' if sanity else 'parts')/key(t)/'point.json'
 try:
  r=load(p)
  return r['protocol_sha256']==sha(ROOT/'protocol.json') and r['code_hashes']==code_hashes() and all((ROOT/f).stat().st_size==h['bytes'] and (sha(ROOT/f)==h['sha256'] if deep else (ROOT/f).stat().st_mtime_ns==h['mtime_ns']) for f,h in r['artifacts'].items())
 except (OSError,KeyError,ValueError):return False

def execute(t,v,targets,th,sanity=False):
 import torch,types
 from proxy_runtime import runtime,frames_for
 from coding_hooks import attach
 dest=ROOT/('sanity/points' if sanity else 'parts')/key(t);dest.mkdir(parents=True,exist_ok=True)
 ref=ROOT/('sanity/baseline' if sanity else 'cache/baseline')/t['dataset']/str(t['video'])/f"q{t['qp']}"
 count=4 if sanity else 64
 eng,rt=runtime(t['method'],targets);base_models=rt.models;rt.sample_frames=[];rt.coding=None
 def models(self,h,w):
  ia,pa=base_models(h,w)
  if self.models_count==1:self.coding=attach(pa,dest,ref,t['method']=='B1_alpha0')
  return ia,pa
 rt.models=types.MethodType(models,rt);tic=time.time();torch.cuda.reset_peak_memory_stats()
 result,quality=rt.run(frames_for(v,count),t['qp'],dest/'stream.bin',dest/'features.npz')
 assert len(rt.latent_rows)==len(quality)==count and len(rt.coding['rows'])==count-1 and len(rt.coding['entropy'])==2*(count-1)
 assert result['independent_decode_pass'] and result['state_sync_pass'] and result['model_hashes_unchanged']
 assert result['real_bytes']==(dest/'stream.bin').stat().st_size
 assert len(rt.wrapper_rows)==(count-1 if t['method']!='Original' else 0)
 wrapper=[dict(frame=0,RGB_residual_L1=0.,RGB_residual_RMSE=0.,RGB_residual_max_abs=0.,**{f'frequency_{b}_energy_delta':0. for b in ('low','mid','high')})]+rt.wrapper_rows
 if t['method']=='Original':wrapper=wrapper*count;wrapper=[dict(r,frame=i) for i,r in enumerate(wrapper)]
 transitions=list(csv.DictReader((dest/'features.transitions.csv').open()))
 for i,(q,l,w) in enumerate(zip(quality,rt.latent_rows,wrapper)):
  q.update(l);q.update(w);q.update(FloLPIPS=None if i==0 else float(transitions[i-1]['FloLPIPS']),source_frame=v['source_frame_indices'][i])
 write(str(dest.relative_to(ROOT))+'/frames.csv',quality)
 write(str(dest.relative_to(ROOT))+'/quantization.csv',rt.coding['rows'])
 write(str(dest.relative_to(ROOT))+'/entropy.csv',rt.coding['entropy'])
 pframes=quality[1:];area=1920*1080
 means={k:sum(float(r[k]) for r in pframes)/len(pframes) for k in ('cosine','L1','RMSE','normalized_L2','LPIPS','DISTS','PSNR','MS_SSIM','RGB_residual_L1','RGB_residual_RMSE','RGB_residual_max_abs','frequency_low_energy_delta','frequency_mid_energy_delta','frequency_high_energy_delta')}
 means['FloLPIPS']=sum(float(r['FloLPIPS']) for r in pframes)/len(pframes)
 qmean={k:sum(float(r[k]) for r in rt.coding['rows'] if r.get(k) is not None)/sum(r.get(k) is not None for r in rt.coding['rows']) for k in ('quant_error_L1','quant_error_RMSE','quant_error_relative_L2','distance_to_nearest_quantization_boundary','fraction_near_quantization_boundary','fraction_quantized_symbols_changed','prequant_shift_vs_alpha0_L1','prequant_shift_vs_alpha0_relative_L2','skip_fraction','saturation_fraction')}
 result.update(task=t,teacher_hash=th,teacher_label='paper_compatible_teacher_proxy',exact_original_teacher_verified=False,source_frame_indices=v['source_frame_indices'][:count],source_rgb_sha256=v['frame_rgb_sha256'][:count],protocol_sha256=sha(ROOT/'protocol.json'),code_hashes=code_hashes(),checkpoint_hashes=load(ROOT/'checkpoint_inventory.json'),alpha=None if t['method']=='Original' else load(ROOT/'protocol.json')['alphas'][load(ROOT/'protocol.json')['methods'][1:].index(t['method'])],P_frame_bpp=sum(r['real_bits'] for r in pframes)/(len(pframes)*area),payload_bpp=sum(r['payload_bits'] for r in quality)/(count*area),P_payload_bpp=sum(r['payload_bits'] for r in pframes)/(len(pframes)*area),payload_bits_per_frame=sum(r['payload_bits'] for r in quality)/count,metrics_P=means,quantization_P=qmean,cdf_entry_bits_per_P_frame=sum(r['cdf_entry_bits'] for r in rt.coding['entropy'])/(count-1),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,elapsed_seconds=time.time()-tic,optimizer_steps=0,quantization_frames=count-1)
 files=[p for p in dest.iterdir() if p.is_file() and p.name!='point.json']
 if t['method']=='B1_alpha0':files+=list(ref.glob('frame*.npz'))
 result['artifacts']={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p)) for p in files}
 dump(str(dest.relative_to(ROOT))+'/point.json',result)
 del rt,eng;gc.collect();torch.cuda.empty_cache()
 print('POINT_PASS',key(t),round(time.time()-tic,1),flush=True)
 return result

def group(dataset,video,qp,gpu,sanity=False):
 import torch
 from proxy_runtime import backend,teacher_targets
 backend();v=next(v for v in load(ROOT/'dataset_manifest.json')['datasets'][dataset] if v['video_index']==video)
 lockpath=ROOT/'cache/locks'/f'{dataset}_{video}_{qp}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  targets,th=teacher_targets(v,4 if sanity else 64)
  checks=[]
  for m in ORDER:
   t=dict(dataset=dataset,video=video,qp=qp,method=m)
   r=load(ROOT/('sanity/points' if sanity else 'parts')/key(t)/'point.json') if valid(t,sanity=sanity) else execute(t,v,targets,th,sanity)
   checks.append(r);dump(f'worker_status/{dataset}_{video}_{qp}.json',dict(status='RUNNING',PID=os.getpid(),gpu=gpu,last_method=m,updated=time.time(),completed_methods=len(checks),peak_memory_MiB=r['peak_memory_MiB'],peak_RSS_MiB=r['peak_RSS_MiB']))
  zero,original=checks[:2]
  assert zero['bitstream_sha256']==original['bitstream_sha256'],'alpha0 vs Original: fixed codec bits changed with receiver'
  assert len({r['I_refresh_audits'][0]['native_I_sha256'] for r in checks})==1,'I0 bypass mismatch'
  dump(f'worker_status/{dataset}_{video}_{qp}.json',dict(status='PASS',PID=os.getpid(),gpu=gpu,completed_methods=7,alpha0_original_stream_identical=True,native_I_identical=True,peak_memory_MiB=max(r['peak_memory_MiB'] for r in checks),peak_RSS_MiB=max(r['peak_RSS_MiB'] for r in checks)))
  if sanity:
   # Compare no-op alpha1 instrumentation against the already audited historical 4-frame B1 stream.
   historical=EXPS/'gvcrt_neural_wrapper_v6_31a_latent_fidelity_vs_generative_utility_audit/sanity/proxy_points'/f'{dataset}/{video:02d}/B1/q{qp}/stream.bin'
   noop=next(r for r in checks if r['task']['method']=='B1_alpha100')
   match=None
   if historical.exists():match=sha(historical)==noop['bitstream_sha256'];assert match
   dump('sanity/sanity.json',dict(status='PASS',methods=7,frames=4,real_RANS=True,fresh_independent_decode=True,alpha0_original_stream_identical=True,native_I_identical=True,alpha1_historical_noop_stream_identical=match,internal_y_yhat_symbols=True,actual_cdf_capture=True,models_unchanged=True,peak_memory_MiB=max(r['peak_memory_MiB'] for r in checks),peak_RSS_MiB=max(r['peak_RSS_MiB'] for r in checks),training_steps=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--qp',type=int,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--sanity',action='store_true');a=p.parse_args()
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1')
 try:group(a.dataset,a.video,a.qp,a.gpu,a.sanity)
 except Exception:
  dump(f'worker_status/error_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc(),args=vars(a)));raise
