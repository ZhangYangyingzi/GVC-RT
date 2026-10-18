"""Real RANS encode/decode/evaluate, with immutable per-attempt artifacts."""
from protocol import *
import argparse,traceback

def run(job,gpu,attempt):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from codec_control import runtime,unpack
 torch.set_num_threads(2);torch.manual_seed(cfg()['seed'])
 video=load(ROOT/'manifests/videos.json')[job['dataset']][job['phase']]
 sys.path.insert(0,str(V32));from adapter32 import frames_for
 inherited=frames_for(video)
 class Frames:
  def __len__(self):return 16
  def __getitem__(self,i):
   if i>=16:raise IndexError(i)
   return inherited[i]
 folder=ROOT/'artifacts/runs'/job['id']/str(attempt)
 folder.mkdir(parents=True,exist_ok=False)
 started=time.time();e,rt=runtime(job,folder);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 stream=folder/'stream.gro';features=folder/'features.npz'
 recon=folder/'reconstruction' if job['kind']=='test_baseline' else None
 result,rows=rt.run(Frames(),job['qp'],stream,features,recon_dir=recon)
 plan,native,header=unpack(stream.read_bytes());assert plan==job['plan']
 (folder/'native.bin').write_bytes(native)
 assert len(rt.weight_checks)==3 and len(rt.decoded_groups)==4
 assert all(len(g)==2 and all(x['exact'] for x in g) for g in rt.decoded_groups.values())
 payload=sum(r['payload_bits'] for r in rows);total=stream.stat().st_size*8
 rate=dict(actual_total_bits=total,native_codec_stream_bits=len(native)*8,RANS_payload_bits=payload,native_SPS_IP_header_bits=len(native)*8-payload,control_header_bits=header*8,actual_total_bpp=total/(16*1920*1080),actual_total_kbps=total*30/(16*1000))
 assert total==sum(rows[i]['real_bits'] for i in range(16))==rate['RANS_payload_bits']+rate['native_SPS_IP_header_bits']+rate['control_header_bits']
 assert [r['actual_qp'] for r in rows]==load(V32/'qp_semantics_audit.json')['actual_qps'][str(job['qp'])][:16]
 result.update(native_bitstream_sha256=sha(folder/'native.bin'),actual_qps=[r['actual_qp'] for r in rows],FID=None,FID_status=cfg()['FID'])
 tensors={}
 for f,data in rt.captures.items():
  path=folder/f'frame{f:02d}_encoder_symbols.npz';np.savez_compressed(path,**data)
  tensors[str(f)]=dict(path=str(path),sha256=sha(path),shapes={k:list(a.shape) for k,a in data.items()})
 if job['kind']=='test_baseline':
  command=['ffmpeg','-y','-v','error','-framerate','30','-i',str(recon/'frame_%06d.png'),'-c:v','ffv1','-threads','2','-level','3','-pix_fmt','bgr0',str(folder/'reconstruction_ffv1.mkv')]
  subprocess.run(command,check=True)
  result['reconstruction_video']=dict(path=str(folder/'reconstruction_ffv1.mkv'),sha256=sha(folder/'reconstruction_ffv1.mkv'),command=command)
 write('metrics/per_frame/'+job['id']+'_'+str(attempt)+'.csv',rows)
 write('metrics/timing/'+job['id']+'_'+str(attempt)+'.csv',rt.timings)
 transition=features.with_suffix('.transitions.csv')
 with transition.open() as f:transitions=list(csv.DictReader(f))
 assert len(transitions)==15
 write('metrics/per_transition/'+job['id']+'_'+str(attempt)+'.csv',transitions)
 timing=dict(encode_ms=sum(r['ms'] for r in rt.timings if r['role']==0),decode_ms=sum(r['ms'] for r in rt.timings if r['role']==2),paired_decode_ms=sum(r['ms'] for r in rt.timings if r['role']==1),scope='synchronized codec calls with read-only symbol observers; full wall time separate')
 roi=[]
 for a in job['plan']:
  z=rt.captures[a['frame']];y,x,h,w=(a[k] for k in ('y','x','h','w'));ix=(slice(None),slice(None),slice(y,y+h),slice(x,x+w))
  roi.append(dict(block=a,symbols=z['symbols'][ix].flatten().tolist(),effective_step_changed_fraction=float((z['original_q'][ix]!=z['effective_q'][ix]).mean())))
 dump('oracle/local_sensitivity/symbols/'+job['id']+'_'+str(attempt)+'.json',roi)
 repeat=None
 if job.get('repeat_source'):
  old=load(job['repeat_source']);repeat={k:result[k]==old['result'][k] for k in ('bitstream_sha256','native_bitstream_sha256','reconstruction_sha256','LPIPS','DISTS','FloLPIPS')}
  assert all(repeat.values()),repeat
 standalone=None
 if job.get('standalone'):
  torch.cuda.empty_cache()
  output='manifests/standalone/'+job['id']+'_'+str(attempt)+'.json'
  command=[PYTHON,'-B',str(ROOT/'src/decode_stream.py'),'--stream',str(stream),'--gpu',str(gpu),'--output',output]
  with (ROOT/'logs'/('standalone_'+job['id']+'_'+str(attempt)+'.log')).open('w') as f:
   process=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
  dump('logs/standalone_exits/'+job['id']+'_'+str(attempt)+'.json',dict(command=command,exit_code=process.returncode,gpu=gpu))
  process.check_returncode();standalone=load(ROOT/output)
  assert standalone['reconstruction_sha256']==result['reconstruction_sha256'] and standalone['frames']==16
 record=dict(status='PASS',job=job,job_sha256=sha(ROOT/'manifests/jobs'/(job['id']+'.json')),attempt=attempt,pid=os.getpid(),gpu=gpu,start=started,result=result,rate=rate,timings=timing,wall_seconds=time.perf_counter()-start,tensor_artifacts=tensors,decoder_symbol_checks=[dict(role=k[0],frame=k[1],groups=v) for k,v in rt.decoded_groups.items()],frozen_weight_checks=rt.weight_checks,reproducibility=repeat,standalone=standalone,reference_state_exact_all_P_frames=True,independent_decode=True,causal_suffix_reencoded=True,effective_torch_seed=torch.initial_seed(),CUDA_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},config_sha256=sha(ROOT/'configs/experiment.json'))
 dump('manifests/attempts/'+job['id']+'_'+str(attempt)+'.json',record)
 dump('manifests/receipts/'+job['id']+'.json',record)
 print('PASS',job['id'],job['phase'],job['dataset'],job['qp'],total,record['wall_seconds'],flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--gpu',required=True,type=int,choices=(4,5,6,7));parser.add_argument('--attempt',required=True,type=int);args=parser.parse_args();job=load(args.job)
 try:run(job,args.gpu,args.attempt)
 except Exception:
  r=dict(status='FAILED',job=job,pid=os.getpid(),gpu=args.gpu,attempt=args.attempt,traceback=traceback.format_exc(),time=time.time())
  dump('manifests/attempts/'+job['id']+'_'+str(args.attempt)+'.json',r);dump('manifests/receipts/'+job['id']+'.json',r);print(r['traceback'],file=sys.stderr);sys.exit(1)
