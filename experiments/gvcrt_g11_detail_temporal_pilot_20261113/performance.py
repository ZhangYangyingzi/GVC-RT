"""G11 component timing on native features; scope explicitly excludes entropy decode."""
from util import *
import torch,gc,copy
from model import load_cref,load_native,G11
from data import Data,load_raft
from original_reference import build_original
from alignment import grid,sample,valid_flow,geometry
from evaluate import highpass

def run():
 if (ROOT/'audits/performance.json').exists():return
 setup_torch();c=load_cref();cp=read(ROOT/'checkpoints'/f"step_{load(ROOT/'completion.json')['committed_updates']:04d}.pt");load_native(c,cp['native_receiver']);b=G11().cuda();b.load_state_dict(cp['branch']);b.eval();data=Data(c);row=videos('validation')[0];seq=row['sequence'];d=data.tensors(seq,56);prev=data.tensors(seq,55);src=data.source(seq);rt,im,pm=build_original();raft,transform=load_raft();f=src.feature(56).cuda().float();fp=src.feature(55).cuda().float();q=src.info['actual_qps'][56];pq=src.info['actual_qps'][55];res={}
 def bench(name,fn,n=5):
  fn();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();vals=[]
  for _ in range(n):
   t=time.perf_counter();z=fn();torch.cuda.synchronize();vals.append((time.perf_counter()-t)*1000);del z
  res[name]=dict(mean_ms=sum(vals)/len(vals),samples_ms=vals,peak_allocated_bytes=torch.cuda.max_memory_allocated(),resident_before_bytes=resident,incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident)
 with torch.no_grad():
  def rgb(f,qp):return (pm.recon_generation_net(f,pm.q_scale_recon[qp:qp+1].float()).clamp(-1,1)+1)/2
  bench('independent_Original_FP32_one_RGB',lambda:rgb(f,q));a,z=rgb(f,q),rgb(fp,pq)
  # The actual parent inference-reference precision remains FP16.
  native_h,native_w=src.info['native_shape'];original16=copy.deepcopy(pm.recon_generation_net).half().eval().requires_grad_(False)
  def rgb16(feat,qp):return ((original16(feat.half(),pm.q_scale_recon[qp:qp+1].half()).float().clamp(-1,1)+1)/2)[...,:native_h,:native_w]
  cached_current=src.aux(56)['original'].cuda().float();cached_past=src.aux(55)['original'].cuda().float();cache_difference=max(float((rgb16(f,q)-cached_current).abs().max()),float((rgb16(fp,pq)-cached_past).abs().max()));assert cache_difference<=1e-6,('ACTUAL_ORIGINAL_FP16_REFERENCE_MISMATCH',cache_difference)
  def native_flow(current,previous):
   aa,bb=transform(current,previous);return raft(aa,bb,num_flow_updates=12)[-1],raft(bb,aa,num_flow_updates=12)[-1]
  def actual_parent_path():return native_flow(rgb16(f,q),rgb16(fp,pq))
  bench('actual_parent_Original_FP16_one_RGB',lambda:rgb16(f,q));bench('actual_parent_Original_FP16_pair_plus_RAFT12',actual_parent_path,3);bench('actual_parent_native_bidirectional_RAFT12',lambda:native_flow(cached_current,cached_past),3)
  def actual_reference_metadata():
   back,forward=actual_parent_path();coords=grid(native_h,native_w,back.device)+back;error=(back+sample(forward,coords)).square().sum(1,keepdim=True).sqrt();valid=valid_flow(back,forward);return geometry(back,valid,error,*d['current_c'].shape[-2:],available=not d['meta']['scene_cut'])
  bench('actual_parent_Original_FP16_pair_RAFT_FB_H8_metadata',actual_reference_metadata,3)


  def flow():
   aa,bb=transform(a,z);back=raft(aa,bb,num_flow_updates=12)[-1];fw=raft(bb,aa,num_flow_updates=12)[-1];return back,fw
  bench('independent_Original_FP32_pair_plus_bidirectional_RAFT',lambda:(rgb(f,q),rgb(fp,pq),flow()),3);bench('bidirectional_RAFT12_only',flow,3);bench('frozen_S_encoder',lambda:c['S'].E(f));bench('G11_DCN_predictor_alignment',lambda:b.align(d['current_c'],d['previous_c'],d['meta']));aligned,v,aux=b.align(d['current_c'],d['previous_c'],d['meta'])
  bench('G11_projection_fusion_output',lambda:b.output(b.fusion(torch.cat((b.P_h(d['h_s']),d['current_c'],aligned,d['current_c']-aligned,v,d['meta']['fb_error']),1))));bench('G11_full_new_branch',lambda:b(d['h_s'],d['current_c'],d['previous_c'],d['meta']));hs,_=b(d['h_s'],d['current_c'],d['previous_c'],d['meta']);bench('complete_receiver_suffix',lambda:data.receiver.suffix(hs,d['x'],d['q'],None,False))
 dump(ROOT/'audits/performance.json',dict(status='PASS',device=device_info(),sequence=seq,frame=56,measurements=res,actual_parent_Original_FP16_RGB_max_abs_to_audited_cache=cache_difference,scope='FP32 native padded features and RGB; Original+RAFT measured directly; codec feature/metadata inputs are existing verified cached data. This is NOT full entropy/codec end-to-end decode latency; no realtime claim.',flow_reference_scope='Parent inference metadata is Original FP16; this supplementary timing uses independent Original FP32 RGB, explicitly named and does not replace inference metadata.',prior_reference_production_seconds={p.parent.name:{k:load(p)[k] for k in ['prefix_seconds','GT_flow_seconds']} for p in (ROOT/'cache/frozen').glob('*/completion.json')}))
if __name__=='__main__':run()
