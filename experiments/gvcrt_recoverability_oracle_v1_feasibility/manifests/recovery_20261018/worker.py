from base import *
import argparse,traceback
START_SOURCE_SHA={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}

def run(job,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from codec_control import runtime,unpack
 cfg=load(ROOT/'configs/experiment.json');torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
 v=next(r for r in load(ROOT/'manifests/videos.json') if r['dataset']==job['dataset'])
 folder=ROOT/'artifacts/runs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(V32));from adapter32 import frames_for
 all_frames=frames_for(v)
 class Frames:
  def __len__(self):return cfg['frames']
  def __getitem__(self,i):
   if i>=len(self):raise IndexError(i)
   return all_frames[i]
 e,rt=runtime(job,folder);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 stream=folder/'stream.gro';features=folder/'features.npz'
 recdir=folder/'reconstruction' if job['kind']=='baseline' else None
 result,fr=rt.run(Frames(),job['qp'],stream,features,recon_dir=recdir)
 plan,native,header=unpack(stream.read_bytes());assert plan==job['plan']
 (folder/'native.bin').write_bytes(native)
 assert len(rt.weight_checks)==3 and len(rt.decoded_groups)==4 and all(len(g)==2 for g in rt.decoded_groups.values())
 capfiles={}
 for f,data in rt.captures.items():
  p=folder/f'frame{f:02d}_encoder_symbols.npz';np.savez_compressed(p,**data);capfiles[str(f)]=dict(path=str(p),sha256=sha(p),shapes={k:list(a.shape) for k,a in data.items()})
 actual_qps=[r['actual_qp'] for r in fr];expected=load(V32/'qp_semantics_audit.json')['actual_qps'][str(job['qp'])][:cfg['frames']]
 assert actual_qps==expected
 payload=sum(r['payload_bits'] for r in fr);nativebits=len(native)*8;total=stream.stat().st_size*8
 accounting=dict(actual_total_bits=total,actual_total_bpp=result['bpp'],actual_total_kbps=result['kbps'],native_codec_stream_bits=nativebits,RANS_payload_bits=payload,native_SPS_IP_header_bits=nativebits-payload,control_header_bits=header*8,definition='complete persisted experiment stream; includes all RANS payloads, native codec headers and signalled control',estimated_bits_used=False)
 assert total==payload+accounting['native_SPS_IP_header_bits']+accounting['control_header_bits']==sum(r['real_bits'] for r in fr)
 result.update(native_codec_stream_bytes=len(native),real_RANS_bytes=len(native),control_header_bytes=header,RANS_payload_bytes=payload//8,native_bitstream_sha256=sha(folder/'native.bin'),actual_qps=actual_qps,FID=None,FID_status=cfg['FID'])
 if job['kind']=='baseline':
  old=load(v['historical_baselines'][str(job['qp'])])['result'];assert native==Path(old['bitstream_path']).read_bytes()[:len(native)],'fresh native baseline differs from verified historical 64-frame prefix'
  result['historical_native_prefix_exact']=True
  movie=folder/'reconstruction_ffv1.mkv'
  command=['ffmpeg','-y','-v','error','-framerate',str(cfg['fps']),'-i',str(recdir/'frame_%06d.png'),'-c:v','ffv1','-threads','2','-level','3','-pix_fmt','bgr0',str(movie)]
  subprocess.run(command,check=True)
  result['reconstruction_video']=dict(path=str(movie),sha256=sha(movie),codec='FFV1 lossless RGB display export; metrics use float reconstructions',command=command)
 write('metrics/per_frame/'+job['id']+'.csv',fr)
 write('metrics/timing/'+job['id']+'.csv',rt.timings)
 times=dict(encode_ms=sum(r['ms'] for r in rt.timings if r['role']==0),decode_ms=sum(r['ms'] for r in rt.timings if r['role']==2),paired_verification_decode_ms=sum(r['ms'] for r in rt.timings if r['role']==1),scope='synchronized codec calls with read-only tensor observers; excludes model loads, metrics, video export and disk I/O; full task wall time separate')
 base=None;delta={};repeat=None
 if job['kind']!='baseline':
  base=load(ROOT/'manifests/receipts'/f'{job["baseline_id"]}.json')
  delta=dict(delta_actual_bits=total-base['rate']['actual_total_bits'],delta_actual_bpp=result['bpp']-base['result']['bpp'],**{'delta_'+m.lower():result[m]-base['result'][m] for m in ('LPIPS','DISTS','FloLPIPS')},delta_fid=None)
  if job['kind']=='repeat':
   repeat={k:result[k]==base['result'][k] for k in ('bitstream_sha256','reconstruction_sha256','LPIPS','DISTS','FloLPIPS')};assert all(repeat.values()),repeat
  if job['kind']=='noop':assert result['native_bitstream_sha256']==base['result']['native_bitstream_sha256']
 roi=[]
 for a in job['plan'] if job['plan'] else cfg['blocks']:
  f,y,x,h,w=(a[k] for k in ('frame','y','x','h','w'));cur=rt.captures[f];s=cur['symbols'][:,:,y:y+h,x:x+w]
  record=dict(block=a,shape=list(s.shape),modified_symbols=s.flatten().tolist(),original_effective_q_min=float(cur['original_q'][:,:,y:y+h,x:x+w].min()),modified_effective_q_min=float(cur['effective_q'][:,:,y:y+h,x:x+w].min()),effective_q_changed_fraction=float((cur['effective_q'][:,:,y:y+h,x:x+w]!=cur['original_q'][:,:,y:y+h,x:x+w]).mean()))
  if base:
   p=base['tensor_artifacts'][str(f)]['path']
   with np.load(p) as z:original=z['symbols'][:,:,y:y+h,x:x+w].copy()
   record.update(original_symbols=original.flatten().tolist(),symbol_changed_fraction=float((s!=original).mean()),original_tensor_path=p)
  else:record['original_symbols']=record['modified_symbols']
  roi.append(record)
 dump('oracle/local_sensitivity/symbols/'+job['id']+'.json',roi)
 source_hashes=START_SOURCE_SHA
 receipt=dict(status='PASS',job=job,job_sha256=sha(ROOT/'manifests/jobs'/f'{job["id"]}.json'),gpu=gpu,pid=os.getpid(),result=result,rate=accounting,delta=delta,reproducibility=repeat,timings=times,wall_seconds=time.perf_counter()-start,tensor_artifacts=capfiles,decoder_symbol_checks=[dict(role=k[0],frame=k[1],groups=g) for k,g in rt.decoded_groups.items()],frozen_weight_checks=rt.weight_checks,reference_state_exact_all_P_frames=True,independent_decode=True,causal_suffix_reencoded=True,CUDAsynchronize=True,CUDA_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,CUDA_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,effective_torch_seed=torch.initial_seed(),code_sha256=source_hashes,FID_status=cfg['FID'])
 dump('manifests/receipts/'+job['id']+'.json',receipt)
 if job['kind'] in ('baseline','repeat'):dump('baseline/'+job['id']+'.json',receipt)
 else:dump('oracle/'+('local_sensitivity' if job['kind'] in ('local','noop','joint') else 'rate_allocation')+'/'+job['id']+'.json',receipt)
 print('PASS',job['id'],accounting['actual_total_bits'],receipt['wall_seconds'],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();job=load(a.job)
 try:run(job,a.gpu)
 except Exception:
  rec=dict(status='FAILED',job=job,gpu=a.gpu,pid=os.getpid(),traceback=traceback.format_exc());dump('manifests/receipts/'+job['id']+'.json',rec);print(rec['traceback'],file=sys.stderr);sys.exit(1)
