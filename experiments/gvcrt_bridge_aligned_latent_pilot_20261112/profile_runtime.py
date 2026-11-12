from util import *
import torch,time,types,io,statistics
from model import *
from restore import restore_b300

def run():
 setup();seed();c=restore_b300();stats=readtensor(ROOT/'cache/statistics.pt');m=AlignedLatent(stats,'T_aligned').cuda();m.load_state_dict(readtensor(ROOT/'checkpoints/T_aligned/step_0300.pt')['model']);m.eval();d=waitcache(manifest()['dev'][0]['id'],'profile',5);f=d['features'][8].cuda().float();old=d['features'][7].cuda().float();l=bridge(c,f).detach();past=bridge(c,old).detach();h,w=l.shape[-2:]
 def measure(fn,n=20):
  with torch.no_grad():
   for _ in range(3):fn()
   torch.cuda.synchronize();a=[]
   for _ in range(n):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();a.append((time.perf_counter()-t)*1000)
  return dict(mean_ms=statistics.mean(a),median_ms=statistics.median(a),samples_ms=a)
 with torch.no_grad():
  u,e,flows=m.aligner(f,old);u16=F.interpolate(u,size=(h,w),mode='bilinear',align_corners=False)/2;coords=grid(h,w,'cuda')+u16;v=bounds(coords,h,w).float();a=torch.where(v.bool(),sample(past,coords),l);e16=F.interpolate(e,size=(h,w),mode='bilinear',align_corners=False)
 timings=dict(matching=measure(lambda:m.aligner(f,old)),warp=measure(lambda:torch.where(bounds(grid(h,w,'cuda')+F.interpolate(u,size=(h,w),mode='bilinear',align_corners=False)/2,h,w),sample(past,grid(h,w,'cuda')+F.interpolate(u,size=(h,w),mode='bilinear',align_corners=False)/2),l)),fusion=measure(lambda:m.fusion(l,a,e16,v,5,6)),Bridge=measure(lambda:bridge(c,f)))
 # Actual bitstream decode with original codec weights/precision. Feature-only P decoding
 # avoids generating or putting candidate RGB in the codec DPB. Feature identities are checked.
 pm=c['pm'];im=c['im'];raw=Path(d['stream_path']).read_bytes();assert sha(d['stream_path'])==d['stream_sha256']
 from src.utils import stream_helper as sh
 def feature_only(self,y_hat,ctx,q_decoder,q_recon):return None,self.dec(y_hat,ctx,q_decoder)
 pm.get_recon_and_feature=types.MethodType(feature_only,pm)
 result={}
 with torch.no_grad():
  for label in ['B300','T300']:
   trials=[];recontimes=[];codectimes=[]
   for repeat in range(3):
    pm.clear_dpb();pm.set_curr_poc(0);m.reset();reader=io.BytesIO(raw);helper=sh.SPSHelper();values=[];ct=[];rt=[]
    for i in range(16):
     torch.cuda.synchronize();t=time.perf_counter();header=sh.read_header(reader)
     while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
     sps=helper.get_sps_by_id(header['sps_id']);qp,payload=sh.read_ip_remaining(reader);assert qp==d['actual_qps'][i]
     dec=im.decompress(payload,sps,qp) if i==0 else pm.decompress(payload,sps,qp)
     if i==0:pm.add_ref_frame(None,dec['x_hat'])
     torch.cuda.synchronize();t1=time.perf_counter()
     if i>0:
      fhat=pm.dpb[0].feature;current=bridge(c,fhat.float());prime,_=m(current,fhat.float(),qp,d['id'],i) if label=='T300' else (current,None);rgb=decode(c,prime,qp);assert pm.dpb[0].frame is None
     torch.cuda.synchronize();end=time.perf_counter();values.append((end-t)*1000);ct.append((t1-t)*1000);rt.append((end-t1)*1000)
     if i>0:assert thash([('f',pm.dpb[0].feature)])==d['codec_feature_hashes'][i]
    assert reader.tell()==len(raw)
    if repeat:trials.extend(values);codectimes.extend(ct);recontimes.extend(rt)
   result[label]=dict(end_to_end_ms_per_frame=statistics.mean(trials),codec_ms_per_frame=statistics.mean(codectimes),external_reconstruction_ms_per_frame=statistics.mean(recontimes),end_to_end_samples=trials,scope='RAM bitstream header+entropy+codec features+Bridge+candidate+full de-tokenizer; I frame includes Original image decoding; P DPB stores only unmodified raw codec f and frame=None; no candidate RGB/latent in DPB; hash checks excluded from timers',feature_bitwise_verified=True)
 counts={g:sum(p.numel() for p in AlignedLatent(stats,g).parameters()) for g in ['S_current','U_unaligned','T_aligned']}
 dump(ROOT/'audits/runtime_profile.json',dict(status='PASS',precision='FP32 reference reconstruction; original codec FP16; PyTorch dispatch, not optimized deployment',native_shape=d['native_shape'],latent_shape=list(l.shape),feature_shape=list(f.shape),new_parameter_counts=counts,new_state_bytes_FP32=f.numel()*4+l.numel()*4,new_state_components=['one original f_hat','one uncorrected Bridge latent','QP/frame metadata'],component_timings=timings,actual_bitstream_pipeline=result))
 heartbeat('profile','COMPLETE')
if __name__=='__main__':run()
