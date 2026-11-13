from util import *
import torch,time,statistics,io,types
from model import DeformableHistory,bridge,decode
from reference import coexist_original,load_raft,decoded_flow
from stream import raw_frames,stream_file,decode_frames

def run():
 setup();from restore import restore_b300
 c=restore_b300();rt,im,pm=coexist_original();raft,tf=load_raft();row=devrows()[0];arrays=raw_frames(row);sp,meta=stream_file(rt,im,pm,row,5,arrays);current=previous=feat=None
 with torch.no_grad():
  for i,q,f,r,sps,_ in decode_frames(rt,im,pm,sp,meta):
   if i==7:previous=r.clone();previous_feature=f.clone()
   if i==8:current=r.clone();feat=f.clone();actual_qp=q;break
  l=bridge(c,feat.float());previous_latent=bridge(c,previous_feature.float());back,fw,e=decoded_flow(raft,tf,current,previous);parts={}
  def measure(fn,n=3):
   vals=[];peaks=[];baselines=[]
   for _ in range(n):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();baselines.append(torch.cuda.memory_allocated());torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();vals.append((time.perf_counter()-t)*1000);peaks.append(torch.cuda.max_memory_allocated());del out
   return dict(mean_ms=statistics.mean(vals),samples_ms=vals,peak_allocated_bytes=max(peaks),resident_before_bytes=max(baselines),max_incremental_bytes=max(p-b for p,b in zip(peaks,baselines)))
  # Replay P9 with the actual P8 codec state, excluding the Original RGB receiver.
  sh=rt['stream'];reader=io.BytesIO(Path(sp).read_bytes());helper=sh.SPSHelper()
  for index in range(10):
   head=sh.read_header(reader)
   while head['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,head['sps_id']));head=sh.read_header(reader)
   current_sps=helper.get_sps_by_id(head['sps_id']);packet_q,packet=sh.read_ip_remaining(reader)
  dpb=list(pm.dpb);poc=pm.curr_poc;original=pm.get_recon_and_feature
  pm.get_recon_and_feature=types.MethodType(lambda self,y,ctx,qd,qr:(None,self.dec(y,ctx,qd)),pm)
  def codec_only():
   pm.dpb=list(dpb);pm.curr_poc=poc;return pm.decompress(packet,current_sps,packet_q)
  parts['codec_feature_only_P9']=measure(codec_only);assert thash([('f',pm.dpb[0].feature)])==meta['features'][9];pm.get_recon_and_feature=original;pm.dpb=dpb;pm.curr_poc=poc
  parts['independent_reference_RGB']=measure(lambda:pm.recon_generation_net(feat,pm.q_scale_recon[actual_qp:actual_qp+1]));parts['bidirectional_RAFT_and_evidence']=measure(lambda:decoded_flow(raft,tf,current,previous));parts['Bridge']=measure(lambda:bridge(c,feat.float()))
  for group in ['A_zero_base','B_flow_guided']:
   m=DeformableHistory(stats(),group).cuda();m.load_state_dict(readtensor(ROOT/'checkpoints'/group/'step_0300.pt')['model']);m.eval();m(previous_latent,back,e,meta['actual_qps'][7],row['id'],7);_,a=m(l,back,e,actual_qp,row['id'],8);ln=l/m.l_rms;old=previous_latent/m.l_rms;qs=(l.new_tensor([actual_qp,5]).view(1,2,1,1).expand(1,-1,*l.shape[-2:])-m.qp_mean)/m.qp_std;flow=a['base_flow'];an=a['aligned_l']/m.l_rms;z=torch.cat((ln,an,(ln-an).abs()),1);ev=torch.nn.functional.interpolate(e,size=l.shape[-2:],mode='bilinear',align_corners=False)/m.evidence_rms;zz=torch.cat((z,ev,qs,a['v']),1)
   parts[group+'.DCN_and_offset']=measure(lambda:m.align(ln,old,flow,qs));parts[group+'.fusion_adapter']=measure(lambda:m.fusion.adapter(z)*m.l_rms);parts[group+'.q_head']=measure(lambda:m.fusion.reliability(zz).sigmoid()*a['v']);parts[group+'.final_decoder']=measure(lambda:decode(c,a['corrected_l'],actual_qp));del m
  dump(ROOT/'audits/performance_components.json',dict(status='PASS',precision='FP32, real native padded inputs, fixed final checkpoints, no updates',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],component_peaks=parts,codec_peak_scope='full noncached decode process peak recorded in performance.json; codec/entropy shares resident weights with reference decoder',cumulative_pipeline_peak_scope='performance.json peak_GPU_GiB is cumulative within that process, not a per-component peak',implementation_sha256=sha(ROOT/'profile_parts.py')))
 heartbeat('profile_parts','COMPLETE')
if __name__=='__main__':run()
