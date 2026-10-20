"""Fresh independent decode of immutable audited native-I, no-P RANS streams."""
from io32 import *
import argparse,fcntl,traceback,math,io

def run(method,video,qp,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from adapter35 import runtime,frames_for,backend
 backend();key=f'{method}_v{video}_q{qp}';dest=ROOT/'tasks'/f'{key}.json';lock=dest.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if dest.exists() and load(dest)['status']=='PASS':return
 old=next(r for r in load(ROOT/'manifests/streams.json') if int(r['video'])==video and int(r['QP'])==qp)
 path=Path(old['bitstream_path']);assert sha(path)==old['bitstream_sha256'] and path.stat().st_size==int(old['real_bytes'])
 v=next(v for v in load(ROOT/'eval_manifest.json')['hevc_b'] if v['video_index']==video)
 assert v['rgb_sha256']==old['source_rgb_sha256'] and v['source_frame_indices']==json.loads(old['source_frame_indices'])
 eng,rt=runtime(dict(method='original',stage='baseline'));ia,pa=rt.models(1088,1920)
 checkpoint_sha=None
 if method!='O':
  assert load(ROOT/f'branches/{method}/final_integrity.json')['status']=='PASS'
  cp=ROOT/f'evaluation/checkpoints/{method}_100.pt';checkpoint_sha=sha(cp);s=torch.load(cp,map_location='cpu',weights_only=True)
  pa.recon_generation_net.mlp.load_state_dict(s['bridge'],strict=True);pa.recon_generation_net.decoder.load_state_dict(s['generator'],strict=True)
 pa.eval().requires_grad_(False);before=[eng.core.module_hash(m) for m in (ia,pa)]
 frames=frames_for(v);flow,perceptual,_=rt.metric_models;stream=io.BytesIO(path.read_bytes());helper=eng.eco.SPSHelper();rows=[];trans=[];hashes=[];feature_hashes=[];previous_reference=previous_output=None;cap={}
 hooks=[pa.dec.register_forward_pre_hook(lambda m,a:cap.update(y=a[0])),pa.dec.register_forward_hook(lambda m,a,o:cap.update(f=o)),pa.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a:cap.update(q=a[1]))]
 torch.cuda.reset_peak_memory_stats();tic=time.time()
 with torch.inference_mode():
  for i,cpu in enumerate(frames):
   pos=stream.tell();header=eng.eco.read_header(stream)
   while header['nal_type']==eng.eco.NalType.NAL_SPS:helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
   sp=helper.get_sps_by_id(header['sps_id']);q,bits=eng.eco.read_ip_remaining(stream)
   assert (header['nal_type']==eng.eco.NalType.NAL_I)==(i==0) and q==json.loads(old['actual_qps'])[i]
   if i==0:
    decoded=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat'])
   else:
    decoded=pa.decompress(bits,sp,q)
    feature_hashes.append(dict(frame=i,**{k:dict(sha256=eng.eco.tensor_sha(cap[k]),shape=list(cap[k].shape),dtype=str(cap[k].dtype)) for k in ('y','f','q')}))
   hashes.append(eng.eco.tensor_sha(decoded['x_hat']));out=eng.core.unit(decoded['x_hat'],1080,1920).float();gt=cpu.cuda();m=eng.eco.frame_metrics(out,gt,rt.quality);m['L1']=float((out-gt).abs().mean())
   assert all(math.isfinite(float(x)) for x in m.values())
   rows.append(dict(frame=i,external_qp=qp,actual_qp=q,real_bits=(stream.tell()-pos)*8,**m))
   if previous_reference is not None:
    fd=flow(previous_reference,gt)-flow(previous_output,out);mag=fd.square().sum(1,keepdim=True).sqrt().sum();assert torch.isfinite(mag) and mag>0
    val=float(perceptual(previous_reference,previous_output,fd,normalize=True).item());assert math.isfinite(val);trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=val))
   previous_reference,previous_output=gt,out
 for h in hooks:h.remove()
 assert len(rows)==64 and len(trans)==63 and stream.tell()==path.stat().st_size and sum(r['real_bits'] for r in rows)==path.stat().st_size*8
 recon=eng.eco.sha256_bytes(''.join(hashes).encode())
 if method=='O':assert recon==old['reconstruction_sha256']
 assert before==[eng.core.module_hash(m) for m in (ia,pa)]
 raw=ROOT/'results/frames'/f'{key}.csv';write(raw,rows);tr=ROOT/'results/transitions'/f'{key}.csv';write(tr,trans)
 result=dict(method=method,video=video,sequence=v['name'],QP=qp,frames=64,transitions=63,LPIPS=sum(r['LPIPS'] for r in rows)/64,DISTS=sum(r['DISTS'] for r in rows)/64,FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,L1=sum(r['L1'] for r in rows)/64,PSNR=-10*math.log10(max(sum(r['pixel_SSE'] for r in rows)/(64*1080*1920*3),1e-15)),PSNR_mean_frame=sum(r['PSNR'] for r in rows)/64,actual_bpp=path.stat().st_size*8/(64*1080*1920),real_bytes=path.stat().st_size,bitstream_sha256=sha(path),bitstream_path=str(path),reuse_bitstream=True,real_RANS=True,independent_decode_pass=True,bytes_consumed=stream.tell(),source_rgb_sha256=v['rgb_sha256'],checkpoint_sha256=checkpoint_sha,reconstruction_sha256=recon,frame_csv_sha256=sha(raw),transition_csv_sha256=sha(tr),code_sha256=sha(ROOT/'evaluate.py'))
 dump(dest,dict(status='PASS',result=result,decoded_features=feature_hashes,frozen_model_hashes=before,physical_GPU=gpu,started_unix=tic,ended_unix=time.time(),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('EVAL_PASS',key,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--method',choices=['O','B','G','BG'],required=True);a.add_argument('--video',type=int,choices=[0,1],required=True);a.add_argument('--qp',type=int,choices=[0,5,9],required=True);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);x=a.parse_args()
 try:run(x.method,x.video,x.qp,x.gpu)
 except Exception:error(dict(stage='evaluation',method=x.method,video=x.video,qp=x.qp,time=time.time(),traceback=traceback.format_exc()));raise
