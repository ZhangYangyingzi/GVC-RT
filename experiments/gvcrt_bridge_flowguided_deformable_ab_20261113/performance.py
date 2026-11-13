"""Noncached deployed pipeline measurement with Original reference and two RAFT passes."""
from util import *
import torch,time,statistics,types,io
from model import DeformableHistory,bridge,decode
from reference import coexist_original,load_raft,decoded_flow
from stream import raw_frames,stream_file,decode_frames

def run():
 setup();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);rt,im,pm=coexist_original();raft,transform=load_raft();row=devrows()[0];arrays=raw_frames(row);path,meta=stream_file(rt,im,pm,row,5,arrays);fixed=load(ROOT/'evaluation/fixed_points.json')['last_common_checkpoint'];results={};st=stats()
 for group in ['A_zero_base','B_flow_guided']:
  m=DeformableHistory(st,group).cuda();m.load_state_dict(readtensor(ROOT/'checkpoints'/group/f'step_{fixed:04d}.pt')['model']);m.eval();m.reset();c['models']['generator'].reset_temporal_buffer();parts={k:[] for k in ['codec','independent_reference_RGB','bidirectional_RAFT_and_evidence','DCN_and_offset','fusion','final_decoder','Bridge','end_to_end']};previous=None;ref_seconds={'value':0.};original_get=pm.get_recon_and_feature
  def split(self,yhat,ctx,qdec,qrecon):
   feature=self.dec(yhat,ctx,qdec);torch.cuda.synchronize();t=time.perf_counter();rgb=self.recon_generation_net(feature,qrecon);torch.cuda.synchronize();ref_seconds['value']=time.perf_counter()-t;return rgb,feature
  pm.get_recon_and_feature=types.MethodType(split,pm);hooks=[];active={};component={}
  def pre(name):
   def hook(mod,args):torch.cuda.synchronize();active[name]=time.perf_counter()
   return hook
  def post(name):
   def hook(mod,args,result):torch.cuda.synchronize();component[name]=time.perf_counter()-active[name]
   return hook
  # Offset and DCN are fused in align(); measure align as one pure block, exact weights/shape.
  align=m.align
  def timed_align(*a,**kw):
   torch.cuda.synchronize();t=time.perf_counter();out=align(*a,**kw);torch.cuda.synchronize();component['DCN_and_offset']=time.perf_counter()-t;return out
  m.align=timed_align
  for name,mod in [('adapter',m.fusion.adapter),('q',m.fusion.reliability)]:hooks.extend([mod.register_forward_pre_hook(pre(name)),mod.register_forward_hook(post(name))])
  pm.clear_dpb();pm.set_curr_poc(0);reader=io.BytesIO(Path(path).read_bytes());sh=rt['stream'];helper=sh.SPSHelper()
  with torch.no_grad():
   for i in range(64):
    heartbeat('performance','MEASURING_NONCACHED',group=group,frame=i);torch.cuda.synchronize();t0=time.perf_counter();header=sh.read_header(reader)
    while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
    sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader);ref_seconds['value']=0.;dec=im.decompress(payload,sps,q) if i==0 else pm.decompress(payload,sps,q)
    if i==0:pm.add_ref_frame(None,dec['x_hat'])
    ref=(dec['x_hat'].float().clamp(-1,1)+1)/2;torch.cuda.synchronize();t1=time.perf_counter();codec=t1-t0-ref_seconds['value'];flow_s=dcn_s=fusion_s=decoder_s=bridge_s=0.
    if i>0:
     ft=pm.dpb[0].feature;tb=time.perf_counter();l=bridge(c,ft.float());torch.cuda.synchronize();bridge_s=time.perf_counter()-tb;tf=time.perf_counter();back,fw,e=decoded_flow(raft,transform,ref,previous);torch.cuda.synchronize();flow_s=time.perf_counter()-tf;component.clear();lp,a=m(l,back,e,q,row['id'],i);torch.cuda.synchronize();dcn_s=component.get('DCN_and_offset',0.);fusion_s=component.get('adapter',0.)+component.get('q',0.);td=time.perf_counter();rgb=decode(c,lp,q);torch.cuda.synchronize();decoder_s=time.perf_counter()-td;assert thash([('f',pm.dpb[0].feature)])==meta['features'][i]
    torch.cuda.synchronize();end=time.perf_counter();previous=ref.detach()
    if i>=1:
     for k,val in dict(codec=codec,independent_reference_RGB=ref_seconds['value'],bidirectional_RAFT_and_evidence=flow_s,DCN_and_offset=dcn_s,fusion=fusion_s,final_decoder=decoder_s,Bridge=bridge_s,end_to_end=end-t0).items():parts[k].append(val*1000)
  pm.get_recon_and_feature=original_get
  for hook in hooks:hook.remove()
  results[group]=dict(step=fixed,frames=63,scope='native P1..P63; RAM bitstream header+entropy/raw-feature decode+independent Original FP32 RGB+two frozen12-update RAFT forwards+Bridge+DCN+fusion+full final decoder; excludes GT/loss/metrics/cache; hash transfers are included and therefore separately disclosed',mean_ms={k:statistics.mean(v) for k,v in parts.items()},samples_ms=parts,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)
 dump(ROOT/'audits/performance.json',dict(status='PASS',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],results=results,matching_and_fusion_not_complete_decode_speed=True,inference_requires_decoded_reference_not_GT=True));heartbeat('performance','COMPLETE')
if __name__=='__main__':run()
