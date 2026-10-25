"""One real native 64-frame stream, a fresh decoder, full maps and state receipts."""
from util import *
import argparse,io,math,time,gc,weakref
def digest(t):return None if t is None else hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def states(pm,result,i,decoding=False):
 tr=pm.decoding_trace if decoding and i else pm.coding_trace if i else {}
 return dict(frame=i,reference=digest(pm.dpb[0].feature if i else result['x_hat']),latent=digest(tr.get('y_hat')),qmap=digest(pm.q_actual) if i else None,rho=digest(tr.get('rho')))
def models(train,torch,mode,native=False):
 b='A' if mode=='A' or native else 'H11'
 im,pm=train.load_models(b)
 path=checkpoint('A' if native else mode);st=torch.load(path,map_location='cpu',weights_only=False)
 assert st['step']==300 and st['branch']==b
 pm.recon_generation_net.decoder.load_state_dict(st['G'],strict=True)
 if b=='H11':pm.quant_adjust.load_state_dict(st['H'],strict=True)
 if not native:
  from ablation import upgrade
  upgrade(pm,mode)
 for m in (im,pm):m.eval().requires_grad_(False)
 return im,pm
def entropy(pm):
 out={}
 for name,m in [('z',pm.bit_estimator_z),('y',pm.gaussian_encoder)]:
  h=hashlib.sha256()
  for a in m.get_cdf_info():h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
  out[name]=h.hexdigest()
 return out
def configure(im,pm,sps):
 im.set_use_two_entropy_coders(bool(sps['ec_part']));pm.set_use_two_entropy_coders(bool(sps['ec_part']))
 pm.clear_dpb();pm.set_curr_poc(0)
def valid(path):
 if not path.exists():return False
 r=load(path)
 if r.get('status')!='PASS' or r.get('signature')!=signature():return False
 return all(Path(p).is_file() and sha(p)==h for p,h in r['artifact_hashes'].items())
def run(mode,source_id,qp,gpu):
 assert mode in MODES and source_id in SOURCES and qp in (2,7)
 k=key(mode,source_id,qp);root=P/'server_artifacts';receipt=root/'evaluation'/f'{k}.json'
 if valid(receipt):print('RESUME_PASS',k,flush=True);return load(receipt)
 train,torch=setup(gpu)
 from core import codec_input,unit,quality_models
 from gvc_hooks import INDEX_MAP
 from src.utils import stream_helper as stream
 import evaluate as v2_eval
 import numpy as np
 assert Path(v2_eval.__file__).resolve()==V2/'evaluate.py'
 source=next(r for r in load(P/'evaluation_sources.json') if r['id']==source_id)
 frames=v2_eval.frames_for(source);h,w=source['height'],source['width']
 sps=dict(sps_id=0,height=(h+63)//64*64,width=(w+63)//64*64,ec_part=int(((h+63)//64*64)*((w+63)//64*64)>1280*720),use_ada_i=0)
 im,pm=models(train,torch,mode);configure(im,pm,sps);enc_cdf=entropy(pm)
 torch.cuda.reset_peak_memory_stats();buffer=io.BytesIO();stream.write_sps(buffer,sps)
 enc=[];estimates=[];diagnostics=[];map_records=[];enc_seconds=0.;payload_bytes=0;i_hash=None
 map_dir=root/'maps'/k;map_dir.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for i,cpu in enumerate(frames):
   x=codec_input(cpu.cuda());q=qp if i==0 else pm.shift_qp(qp,INDEX_MAP[i%8]);yb=zb=0.
   torch.cuda.synchronize();started=time.time()
   if not i:
    result=im.compress(x,q);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,result['x_hat'])
    i_hash=hashlib.sha256(result['bit_stream']).hexdigest()
   else:
    captured={};hook=pm.hyper_enc.register_forward_hook(lambda m,args,out:captured.update(z=out.detach()))
    original=pm.compress_prior_2x
    def capture(y,common,spatial):
     out=original(y,common,spatial);captured['prior']=tuple(t.detach() for t in out[:4]);return out
    pm.compress_prior_2x=capture
    try:result=pm.compress(x,q)
    finally:hook.remove();pm.compress_prior_2x=original
    z=captured['z'].round().clamp(-128,127).float();zb=float(pm.get_z_bits(z,q,pm.bit_estimator_z).sum())
    y0,y1,s0,s1=captured['prior'];d=dict(mode=mode,sequence=source_id,QP=qp,frame=i,clamp_fraction=pm.clamp_fraction)
    for j,(sym,scale) in enumerate(((y0,s0),(y1,s1))):
     active=scale.float()>.12;d[f'pass{j}_skip_fraction']=float((~active).float().mean())
     s=scale.float()[active].clamp(.11,16);v=sym.float()[active];normal=torch.distributions.Normal(torch.zeros_like(s),s)
     yb+=float(-torch.log2((normal.cdf(v+.5)-normal.cdf(v-.5)).clamp_min(1e-9)).sum())
    assert torch.equal(pm.maps['rho'],pm.coding_trace['rho'])
    for name,t in pm.maps.items():
     f=t.detach().float();d[name+'_mean']=float(f.mean());d[name+'_std']=float(f.std(unbiased=False));d[name+'_min']=float(f.min());d[name+'_max']=float(f.max())
    file=map_dir/f'frame_{i:02d}.npz';tmp=file.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**{name:t.detach().cpu().numpy() for name,t in pm.maps.items()})
    tmp.replace(file)
    map_records.append(dict(frame=i,path=str(file),sha256=sha(file),tensor_hashes={name:digest(t) for name,t in pm.maps.items()},shapes={name:list(t.shape) for name,t in pm.maps.items()}))
    diagnostics.append(d)
   enc.append(states(pm,result,i));estimates.append((yb,zb));payload_bytes+=len(result['bit_stream'])
   stream.write_ip(buffer,i==0,0,q,result['bit_stream'])
   torch.cuda.synchronize();enc_seconds+=time.time()-started
 path=root/'streams'/f'{k}.bin';path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
 temp.replace(path);total=path.stat().st_size
 assert entropy(pm)==enc_cdf
 ref=weakref.ref(pm);del original,capture,hook,captured,im,pm,buffer,result
 gc.collect();assert ref() is None,'encoder retained';torch.cuda.empty_cache()
 ia,pa=models(train,torch,mode);configure(ia,pa,sps);dec_cdf=entropy(pa);assert enc_cdf==dec_cdf
 quality=quality_models(torch.device('cuda:0'))
 compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback');sys.path.insert(0,str(compat))
 import flolpips_compat
 flow,perceptual=flolpips_compat.load_models(torch.device('cuda:0'));flow.eval().requires_grad_(False);perceptual.eval().requires_grad_(False)
 from experiments.gvcrt_neural_wrapper_v4_progressive_joint.eval_core import frame_metrics
 buffer=io.BytesIO(path.read_bytes());helper=stream.SPSHelper();rows=[];trans=[];dec=[];previous_gt=previous_out=None;dec_seconds=0.
 with torch.inference_mode():
  for i,cpu in enumerate(frames):
   begin=buffer.tell();head=stream.read_header(buffer)
   while head['nal_type']==stream.NalType.NAL_SPS:
    helper.add_sps_by_id(stream.read_sps_remaining(buffer,head['sps_id']));head=stream.read_header(buffer)
   sp=helper.get_sps_by_id(head['sps_id']);assert sp==sps
   q,bits=stream.read_ip_remaining(buffer);assert q==(qp if not i else pa.shift_qp(qp,INDEX_MAP[i%8]))
   assert (head['nal_type']==stream.NalType.NAL_I)==(i==0)
   torch.cuda.synchronize();started=time.time()
   if not i:
    result=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,result['x_hat'])
   else:result=pa.decompress(bits,sp,q)
   dec.append(states(pa,result,i,True));assert dec[-1]==enc[i],(k,i,'state mismatch')
   if i:assert {n:digest(t) for n,t in pa.maps.items()}==map_records[i-1]['tensor_hashes'],(k,i,'map mismatch')
   torch.cuda.synchronize();dec_seconds+=time.time()-started
   out=unit(result['x_hat'],h,w).float();gt=cpu.cuda();metric=frame_metrics(out,gt,quality)
   rows.append(dict(frame=i,source_frame_index=source['indices'][i],actual_qp=q,real_bytes=buffer.tell()-begin,payload_bytes=len(bits),estimated_y_bits=estimates[i][0],estimated_z_bits=estimates[i][1],**metric))
   if previous_gt is not None:
    motion=flow(previous_gt,gt)-flow(previous_out,out);val=float(perceptual(previous_gt,previous_out,motion,normalize=True).item());assert math.isfinite(val)
    trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=val))
   previous_gt,previous_out=gt,out
   if i==15 and qp==2:
    from PIL import Image
    preview=P/'previews';preview.mkdir(exist_ok=True)
    for label,t in [(k,out),(source_id+'_GT',gt)]:
     a=t[0,:,200:456,400:656].permute(1,2,0).clamp(0,1).cpu().numpy()
     target=preview/f'{label}_frame15.png';tmp=target.with_name(target.stem+f'.{os.getpid()}.tmp')
     Image.fromarray((a*255).round().astype('uint8')).save(tmp,format='PNG');tmp.replace(target)
 assert len(rows)==64 and len(trans)==63 and buffer.tell()==total
 native_parity=None
 if mode in ('A','IDENTITY'):
  # Independently restore the original A/v2 runtime, with no new granularity class.
  ni,npmodel=models(train,torch,'A',native=True);configure(ni,npmodel,sps)
  native=io.BytesIO(path.read_bytes());nh=stream.SPSHelper();native_states=[]
  with torch.inference_mode():
   for i in range(64):
    head=stream.read_header(native)
    while head['nal_type']==stream.NalType.NAL_SPS:
     nh.add_sps_by_id(stream.read_sps_remaining(native,head['sps_id']));head=stream.read_header(native)
    sp=nh.get_sps_by_id(head['sps_id']);q,bits=stream.read_ip_remaining(native)
    if not i:
     r=ni.decompress(bits,sp,q);npmodel.clear_dpb();npmodel.set_curr_poc(0);npmodel.add_ref_frame(None,r['x_hat'])
    else:r=npmodel.decompress(bits,sp,q)
    native_states.append(states(npmodel,r,i,True));assert native_states[-1]==enc[i],(k,i,'native mismatch')
  old=load(V2/'server_artifacts/evaluation'/f'A_s300_{source_id}_q{qp}.json')
  assert sha(path)==old['stream_sha256'],'native byte parity'
  native_parity=dict(status='PASS',native_runtime='original v2 A; v1 A step300 G',stream_sha256=old['stream_sha256'],all_64_states_exact=True)
 root.mkdir(exist_ok=True);hashes={str(path):sha(path)}
 for folder,data in [('frames',rows),('transitions',trans),('diagnostics',diagnostics)]:
  f=root/folder/f'{k}.csv';table(f,data);hashes[str(f)]=sha(f)
 f=root/'states'/f'{k}.json';dump(f,dict(status='PASS',encoder=enc,decoder=dec,maps=map_records));hashes[str(f)]=sha(f)
 hashes.update({r['path']:r['sha256'] for r in map_records})
 result=dict(status='PASS',mode=mode,sequence=source_id,external_qp=qp,step=300,alpha=1.,training_updates=0,signature=signature(),physical_gpu=gpu,frames=64,IP=-1,height=h,width=w,source_sha256=source['source_sha256'],source_frame_indices=source['indices'],checkpoint_path=str(checkpoint(mode)),checkpoint_sha256=sha(checkpoint(mode)),stream_path=str(path),stream_sha256=sha(path),I_payload_sha256=i_hash,real_total_bytes=total,payload_bytes=payload_bytes,header_bytes=total-payload_bytes,P_payload_bytes=sum(r['payload_bytes'] for r in rows[1:]),estimated_P_y_bits=sum(v[0] for v in estimates[1:]),estimated_P_z_bits=sum(v[1] for v in estimates[1:]),actual_bpp=total*8/(64*h*w),kbps_20fps=total*8*20/(64*1000),PSNR=-10*math.log10(sum(r['pixel_SSE'] for r in rows)/(64*h*w*3)),LPIPS=sum(r['LPIPS'] for r in rows)/64,DISTS=sum(r['DISTS'] for r in rows)/64,FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,encoding_seconds=enc_seconds,decoding_seconds=dec_seconds,peak_MiB=torch.cuda.max_memory_reserved()/2**20,encoder_released=True,independent_decode=True,all64_reference_latent_qmap_rho_equal=True,all_maps_encoder_decoder_equal=True,encoder_CDF=enc_cdf,decoder_CDF=dec_cdf,native_parity=native_parity,artifact_hashes=hashes,FID='not_computed',actual_y_z_partition='combined_RANS_payload',mode_information='fixed shared experiment configuration; native stream has no new side information')
 dump(receipt,result);print('TASK_PASS',k,result['actual_bpp'],result['LPIPS'],flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=MODES,required=True);a.add_argument('--source',choices=SOURCES,required=True);a.add_argument('--qp',type=int,choices=[2,7],required=True);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
 v=a.parse_args();run(v.mode,v.source,v.qp,v.gpu)
