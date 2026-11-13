from util import *
import torch,numpy as np,io,gc
from reference import independent_original,load_raft,decoded_flow
from model import resize_flow
import torch.nn.functional as F

def verify_replay(rt,im,pm,d,refs):
 sh=rt['stream'];reader=io.BytesIO(Path(d['stream_path']).read_bytes());helper=sh.SPSHelper();pm.clear_dpb();pm.set_curr_poc(0);records=[]
 with torch.no_grad():
  for i in range(len(d['features'])):
   header=sh.read_header(reader)
   while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
   sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader);assert q==d['actual_qps'][i];dec=im.decompress(payload,sps,q) if i==0 else pm.decompress(payload,sps,q)
   if i==0:pm.add_ref_frame(None,dec['x_hat'])
   else:assert thash([('f',pm.dpb[0].feature)])==d['codec_feature_hashes'][i]
   rgb=(dec['x_hat'].float().clamp(-1,1)+1)/2;assert torch.equal(rgb.cpu(),refs[i]),('ORIGINAL_REFERENCE_REPLAY_MISMATCH',d['id'],i);records.append(dict(frame=i,RGB_SHA256=thash([('RGB',rgb)])))
 assert reader.tell()==Path(d['stream_path']).stat().st_size;return records

def run():
 setup();rt,im,pm=independent_original();raft,transform=load_raft();before=thash(im.state_dict().items())+thash(pm.state_dict().items());rms=[];ids=plan()['stats_ids'];verified_qp=set();old=load(PREV/'audits/preflight.json');keys={k:sha(ROOT/k) for k in ['reference.py','cache_train.py']};base={k:old['sources'][k] for k in ['RAFT']};base.update(original_source_protocol_sha256=sha(PROJECT/'experiments/gvcrt_original_independent_rd_20261105/protocol.json'),reference_precision='FP32 P receiver, inherited FP16 I codec; FP32 output tensors',flow_iterations=12,flow_input='independent Original RGB, never GT/candidate')
 for row in training_rows():
  id=row['id'];path=sidepath(id)
  if path.with_suffix('.npz.json').exists():
   meta=load(path.with_suffix('.npz.json'))['meta'];assert meta['implementation']['reference.py']==keys['reference.py'];continue
  heartbeat('cache_train','BUILDING',sample=id);t=time.time();d=oldtrain(id);assert len(d['features'])==16 and len(d['original'])==1,'OLD_TRAIN_ORIGINAL_MUST_ONLY_SUPPLY_I'
  assert sha(d['stream_path'])==d['stream_sha256'];refs=[d['original'][0].clone()]
  with torch.no_grad():
   for i in range(1,16):refs.append(((pm.recon_generation_net(d['features'][i].cuda(),pm.q_scale_recon[d['actual_qps'][i]:d['actual_qps'][i]+1])+1)/2).cpu())
   if row['external_qp'] not in verified_qp:
    records=verify_replay(rt,im,pm,d,refs);dump(ROOT/'audits'/f'original_reference_replay_q{row["external_qp"]}.json',dict(status='PASS',id=id,reference_count=16,records=records,source_stream=d['stream_path'],stream_sha256=d['stream_sha256'],all_P_frames_bitwise_equal=True));verified_qp.add(row['external_qp'])
   back=[np.zeros((1,2,16,16),np.float32)];forward=[back[0].copy()];evidence=[back[0].copy()]
   for i in range(1,16):
    heartbeat('cache_train','REFERENCE_FLOW',sample=id,frame=i);b,f,e=decoded_flow(raft,transform,refs[i].cuda(),refs[i-1].cuda());back.append(resize_flow(b,(16,16)).cpu().numpy());forward.append(resize_flow(f,(16,16)).cpu().numpy());evidence.append(F.interpolate(e,size=(16,16),mode='bilinear',align_corners=False).cpu().numpy())
  meta=dict(id=id,source=row['source_path'],source_sha256=row['source_sha256'],source_frames=row['source_frame_indices'],crop=row['crop'],external_QP=row['external_qp'],actual_QPs=d['actual_qps'],stream_sha256=d['stream_sha256'],codec_features=d['codec_feature_hashes'],implementation=keys,reference_identity=base,flow_grid='latent16 pixels; dx,dy; current->previous; vectors rescaled by16/256; evidence bilinear downsampled before fixed RMS',frames=16)
  npzsave(path,dict(decoded_ref_RGB=np.stack([r.numpy() for r in refs]),decoded_ref_backward=np.stack(back),decoded_ref_forward=np.stack(forward),decoded_ref_evidence=np.stack(evidence)),meta)
  event('DECODED_REFERENCE_CACHE_COMMITTED',id=id,seconds=time.time()-t,bytes=path.stat().st_size,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  if all(sidepath(k).with_suffix('.npz.json').exists() for k in ids) and not (ROOT/'cache/evidence_stats.pt.json').exists():
   vals=[]
   for k in ids:
    a=npzload(sidepath(k))['decoded_ref_evidence'][1:];vals.append(np.mean(a.astype('float64')**2,axis=(0,1,3,4)))
   scales=np.sqrt(np.mean(vals,axis=0)).clip(.01);save(ROOT/'cache/evidence_stats.pt',dict(evidence_rms=torch.tensor(scales).float(),ids=ids,train_only=True));dump(ROOT/'audits/evidence_statistics.json',dict(ids=ids,RMS=scales.tolist(),floor=.01,train_only=True));event('EVIDENCE_STATS_FROZEN',RMS=scales.tolist())
  del d,refs,back,forward,evidence;gc.collect();torch.cuda.empty_cache()
 assert before==thash(im.state_dict().items())+thash(pm.state_dict().items());heartbeat('cache_train','COMPLETE',committed_caches=len(training_rows()))
if __name__=='__main__':
 with lock('cache_train.lock'):run()
