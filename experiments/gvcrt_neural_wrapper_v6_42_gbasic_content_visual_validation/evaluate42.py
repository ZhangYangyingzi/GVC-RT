"""Two independent no-update real-RANS decodes with explicit receiver-input hooks."""
from common42 import *
import argparse,fcntl,io,math,traceback,gc

def run(dataset,sequence,qp,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 import torch,numpy as np
 from PIL import Image
 from adapter35 import runtime,frames_for,backend
 backend();key=f'{dataset}/{sequence}/q{qp}';dest=ROOT/'tasks'/key/'pair.json';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  sig=signature42()
  if dest.exists():
   old=load(dest);assert old['signature']==sig
   if old['status']=='PASS':return
  assert sha(CP)==CP_SHA
  def forbidden(*args,**kwargs):raise AssertionError('Optimizer update forbidden in evaluation-only experiment')
  torch.optim.Adam.step=forbidden;torch.optim.AdamW.step=forbidden;torch.optim.SGD.step=forbidden
  v=next(v for v in load(ROOT/'data_manifest.json') if (v['dataset'],v['name'])==(dataset,sequence))
  stream_info=next(s for s in load(ROOT/'stream_manifest.json') if (s['dataset'],s['sequence'],s['QP'])==(dataset,sequence,qp))
  path=Path(stream_info['path']);assert path.stat().st_size==stream_info['bytes'] and sha(path)==stream_info['sha256']
  frames=frames_for(v);eng,rt=runtime(dict(method='original',stage='baseline'))
  metrics=(*rt.quality,*rt.metric_models)
  for m in metrics:m.eval().requires_grad_(False)
  metric_before=[eng.core.module_hash(m) for m in metrics]
  results=[];proofs={};I_hashes=[];frozen=[];tic=time.time()
  visual=qp in (0,9)
  def save_png(t,p):
   a=t.detach().clamp(0,1).mul(255).round().byte()[0].permute(1,2,0).cpu().numpy()
   p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');Image.fromarray(a).save(tmp,format='PNG',compress_level=1);tmp.replace(p)
  for method in METHODS:
   ia,pa=rt.models(1088,1920)
   initial_bridge=eng.core.module_hash(pa.recon_generation_net.mlp)
   if method=='G_basic':
    s=torch.load(CP,map_location='cpu',weights_only=True)
    pa.recon_generation_net.mlp.load_state_dict(s['bridge'],strict=True);pa.recon_generation_net.decoder.load_state_dict(s['generator'],strict=True);del s
   for m in (ia,pa):m.eval().requires_grad_(False)
   assert eng.core.module_hash(pa.recon_generation_net.mlp)==initial_bridge
   assert eng.core.compression_hash(ia,pa)==load(ROOT/'evaluation_config.json')['compression_hash']
   assert {str(p.dtype) for m in (ia,pa) for p in m.parameters()}=={'torch.float16'}
   assert all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()) for m in (ia,pa,*metrics))
   before=[eng.core.module_hash(m) for m in (ia,pa)];cap={}
   hooks=[pa.dec.register_forward_pre_hook(lambda m,a:cap.update(y=a[0])),
    pa.dec.register_forward_hook(lambda m,a,o:cap.update(f=o)),
    pa.recon_generation_net.mlp[3].register_forward_hook(lambda m,a,o:cap.update(bridge_output=o)),
    pa.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a:cap.update(z=a[0],q=a[1]))]
   stream=io.BytesIO(path.read_bytes());helper=eng.eco.SPSHelper();rows=[];trans=[];features=[];hashes=[];prevgt=prevout=None
   with torch.inference_mode():
    for i,cpu in enumerate(frames):
     pos=stream.tell();header=eng.eco.read_header(stream)
     while header['nal_type']==eng.eco.NalType.NAL_SPS:
      helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
     sp=helper.get_sps_by_id(header['sps_id']);q,bits=eng.eco.read_ip_remaining(stream)
     assert sp['height']==1088 and sp['width']==1920 and q==stream_info['actual_qps'][i]
     assert (header['nal_type']==eng.eco.NalType.NAL_I)==(i==0)
     if i==0:
      decoded=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat']);I_hashes.append(eng.eco.tensor_sha(decoded['x_hat']))
     else:
      decoded=pa.decompress(bits,sp,q)
      assert torch.equal(cap['bridge_output'],cap['z'])
      features.append(dict(frame=i,**{k:dict(sha256=eng.eco.tensor_sha(cap[k]),shape=list(cap[k].shape),dtype=str(cap[k].dtype)) for k in ('y','f','q','z','bridge_output')}))
     hashes.append(eng.eco.tensor_sha(decoded['x_hat']));out=eng.core.unit(decoded['x_hat'],1080,1920).float();gt=cpu.cuda()
     m=eng.eco.frame_metrics(out,gt,rt.quality);m['L1']=float((out-gt).abs().mean());assert all(math.isfinite(float(x)) for x in m.values())
     rows.append(dict(frame=i,source_frame=v['source_frame_indices'][i],actual_qp=q,frame_type='I' if i==0 else 'P',real_bits=(stream.tell()-pos)*8,**m))
     if i:
      fd=rt.metric_models[0](prevgt,gt)-rt.metric_models[0](prevout,out)
      value=float(rt.metric_models[1](prevgt,prevout,fd,normalize=True).item());assert math.isfinite(value)
      trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))
     if visual:
      base=ROOT/'server_media'/key
      save_png(out,base/method/f'{i:03d}.png')
      if method=='O_original':save_png(gt,base/'GT'/f'{i:03d}.png')
     prevgt,prevout=gt,out
   for h in hooks:h.remove()
   assert len(rows)==64 and len(trans)==63 and stream.tell()==path.stat().st_size and sum(r['real_bits'] for r in rows)==8*path.stat().st_size
   assert before==[eng.core.module_hash(m) for m in (ia,pa)]
   assert all(p.grad is None for m in (ia,pa,*metrics) for p in m.parameters())
   reconstruction=eng.eco.sha256_bytes(''.join(hashes).encode())
   if method=='O_original':assert reconstruction==stream_info['original_reconstruction_sha256'],('original reconstruction mismatch',key)
   proofs[method]=features;frozen.append(dict(method=method,before=before,after=before,all_eval=True,parameter_gradients_absent=True,precision='float16 codec/receiver, float32 metrics input'))
   f=ROOT/'results'/key/f'{method}_frames.csv';t=ROOT/'results'/key/f'{method}_transitions.csv';write(f,rows);write(t,trans)
   result=dict(dataset=dataset,sequence=sequence,QP=qp,method=method,frames=64,transitions=63,
    LPIPS=sum(r['LPIPS'] for r in rows)/64,DISTS=sum(r['DISTS'] for r in rows)/64,L1=sum(r['L1'] for r in rows)/64,
    FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,PSNR=-10*math.log10(max(sum(r['pixel_SSE'] for r in rows)/(64*1920*1080*3),1e-15)),
    actual_bpp=path.stat().st_size*8/(64*1920*1080),bytes_consumed=stream.tell(),real_bytes=path.stat().st_size,
    bitstream_sha256=sha(path),reconstruction_sha256=reconstruction,frame_csv=str(f),frame_csv_sha256=sha(f),transition_csv=str(t),transition_csv_sha256=sha(t))
   results.append(result);del ia,pa,prevgt,prevout;gc.collect();torch.cuda.empty_cache()
  assert proofs['O_original']==proofs['G_basic'] and I_hashes[0]==I_hashes[1]
  assert metric_before==[eng.core.module_hash(m) for m in metrics]
  dump(dest,dict(status='PASS',signature=sig,task_key=key,results=results,input_proof=proofs['O_original'],
   frozen=frozen,metric_hashes=metric_before,metrics_unchanged=True,I_output_identical=True,optimizer_updates=0,
   visual_manifest_sha256=sha(ROOT/'visual_manifest.json'),physical_GPU=gpu,started=tic,finished=time.time(),
   stream_sha256=sha(path),same_y_f_q_z_bridge_output=True))
  print('PAIR_PASS',key,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--sequence',required=True);p.add_argument('--qp',type=int,choices=(0,5,9),required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
 try:run(a.dataset,a.sequence,a.qp,a.gpu)
 except Exception:append(ROOT/'errors.jsonl',dict(time=time.time(),task=vars(a),error=traceback.format_exc()));raise
