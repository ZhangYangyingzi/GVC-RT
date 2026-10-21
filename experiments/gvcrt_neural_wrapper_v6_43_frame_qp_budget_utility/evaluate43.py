"""Fixed G_basic full RANS encoding and fresh independent decoding; no updates."""
from common43 import *
import argparse,fcntl,gc,io,math,traceback

def run(task,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 import torch
 from adapter35 import backend,runtime,frames_for
 backend()
 dest=result_path(task);dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if validate_result(task):print('REUSED',key(task),flush=True);return
  started=time.time();sig=signature43();assert sha(CP)==CP_SHA
  for f in load(ROOT/'source_manifest.json'):assert sha(f['path'])==f['sha256'],f['path']
  def forbidden(*a,**k):raise AssertionError('Optimizer updates forbidden')
  torch.optim.Adam.step=torch.optim.AdamW.step=torch.optim.SGD.step=forbidden
  v=next(v for v in load(ROOT/'data_manifest.json') if (v['dataset'],v['name'])==(task['dataset'],task['sequence']))
  old=next(s for s in load(ROOT/'baseline_manifest.json') if s['sequence']==task['sequence'])
  if task['kind']!='baseline':
   bt=next(t for t in load(ROOT/'intervention_plan.json') if t['sequence']==task['sequence'] and t['kind']=='baseline')
   assert validate_result(bt);base=load(result_path(bt))
  eng,rt=runtime(dict(method='original',stage='baseline'));frames=frames_for(v)
  metrics=(*rt.quality,*rt.metric_models)
  for m in metrics:m.eval().requires_grad_(False)
  metric_hashes=[eng.core.module_hash(m) for m in metrics]
  def models():
   im,pm=rt.models(1088,1920);original_bridge=eng.core.module_hash(pm.recon_generation_net.mlp)
   cp=torch.load(CP,map_location='cpu',weights_only=True)
   pm.recon_generation_net.mlp.load_state_dict(cp['bridge'],strict=True)
   pm.recon_generation_net.decoder.load_state_dict(cp['generator'],strict=True)
   assert eng.core.module_hash(pm.recon_generation_net.mlp)==original_bridge
   assert eng.core.compression_hash(im,pm)==load(ROOT/'evaluation_config.json')['compression_hash']
   for m in (im,pm):m.eval().requires_grad_(False)
   assert {str(p.dtype) for m in (im,pm) for p in m.parameters()}=={'torch.float16'}
   return im,pm
  def frozen(mods,before):
   assert before==[eng.core.module_hash(m) for m in mods]
   assert all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()) for m in (*mods,*metrics))
  def state(pm):
   ref=pm.dpb[0]
   return dict(poc=ref.poc,curr_poc=pm.curr_poc,feature=None if ref.feature is None else eng.eco.tensor_sha(ref.feature),frame=None if ref.frame is None else eng.eco.tensor_sha(ref.frame))
  def capture(pm):
   cap={}
   hooks=[pm.dec.register_forward_pre_hook(lambda m,a:cap.update(y=a[0])),pm.dec.register_forward_hook(lambda m,a,o:cap.update(f=o)),pm.recon_generation_net.mlp[3].register_forward_hook(lambda m,a,o:cap.update(bridge=o)),pm.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a:cap.update(z=a[0],q=a[1]))]
   return cap,hooks
  def hashes(cap,keys):return {k:eng.eco.tensor_sha(cap[k]) for k in keys}
  sps=dict(sps_id=0,height=1088,width=1920,ec_part=1,use_ada_i=0)
  stream_path=ROOT/'bitstreams'/task['dataset']/task['sequence']/(task['variant']+'.bin');stream_path.parent.mkdir(parents=True,exist_ok=True)
  encoder=[];packets=[];buf=io.BytesIO();eng.eco.write_sps(buf,sps)
  with torch.inference_mode():
   ie,pe=models();before=[eng.core.module_hash(m) for m in (ie,pe)];cap,hooks=capture(pe)
   pe.clear_dpb();pe.set_curr_poc(0)
   for i,cpu in enumerate(frames):
    q=task['actual_qps'][i];assert 0<=q<ie.get_qp_num() if i==0 else 0<=q<pe.q_scale_enc.shape[0]
    x=eng.replicate_pad(cpu.cuda().half(),8,0)*2.-1.
    start=0 if i==0 else buf.tell()
    if i==0:
     encoded=ie.compress(x,q);pe.clear_dpb();pe.set_curr_poc(0);pe.add_ref_frame(None,encoded['x_hat'])
     proof=dict(frame=i,state=state(pe),reconstruction_sha256=eng.eco.tensor_sha(encoded['x_hat']))
    else:
     encoded=pe.compress(x,q);proof=dict(frame=i,state=state(pe),**hashes(cap,('y','f')))
    eng.eco.write_ip(buf,i==0,0,q,encoded['bit_stream'])
    packet=buf.getvalue()[start:buf.tell()]
    packets.append(dict(frame=i,actual_qp=q,frame_type='I' if i==0 else 'P',packet_bytes=len(packet),payload_bytes=len(encoded['bit_stream']),header_bytes=len(packet)-len(encoded['bit_stream']),packet_sha256=eng.eco.sha256_bytes(packet),payload_sha256=eng.eco.sha256_bytes(encoded['bit_stream'])))
    encoder.append(proof)
    if i%16==0:print('ENC',key(task),i,flush=True)
   for h in hooks:h.remove()
   frozen((ie,pe),before);del ie,pe,x,cap,encoded;gc.collect();torch.cuda.empty_cache()
   raw=buf.getvalue();tmp=stream_path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(stream_path)
   if task['kind']=='baseline':assert len(raw)==old['bytes'] and sha(stream_path)==old['sha256'],('baseline stream mismatch',len(raw),sha(stream_path),old['sha256'])
   ia,pa=models();before=[eng.core.module_hash(m) for m in (ia,pa)];cap,hooks=capture(pa)
   stream=io.BytesIO(stream_path.read_bytes());helper=eng.eco.SPSHelper();rows=[];trans=[];decoder=[];rec_hash=[];prevgt=prevout=None
   for i,cpu in enumerate(frames):
    start=stream.tell();header=eng.eco.read_header(stream)
    while header['nal_type']==eng.eco.NalType.NAL_SPS:
     helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
    sp=helper.get_sps_by_id(header['sps_id']);q,bits=eng.eco.read_ip_remaining(stream)
    assert sp==sps and q==task['actual_qps'][i] and (header['nal_type']==eng.eco.NalType.NAL_I)==(i==0)
    if i==0:
     decoded=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat'])
     proof=dict(frame=i,state=state(pa))
     assert eng.eco.tensor_sha(decoded['x_hat'])==encoder[i]['reconstruction_sha256']
    else:
     decoded=pa.decompress(bits,sp,q);assert torch.equal(cap['bridge'],cap['z']);proof=dict(frame=i,state=state(pa),**hashes(cap,('y','f','bridge','z','q')))
     assert proof['y']==encoder[i]['y'] and proof['f']==encoder[i]['f'] and proof['state']['feature']==encoder[i]['state']['feature'],('state sync',i)
    assert proof['state']['poc']==encoder[i]['state']['poc'] and proof['state']['curr_poc']==encoder[i]['state']['curr_poc']
    rec_hash.append(eng.eco.tensor_sha(decoded['x_hat']));proof['reconstruction_sha256']=rec_hash[-1];decoder.append(proof)
    out=eng.core.unit(decoded['x_hat'],1080,1920).float();gt=cpu.cuda();m=eng.eco.frame_metrics(out,gt,rt.quality);m['L1']=float((out-gt).abs().mean());assert all(math.isfinite(float(x)) for x in m.values())
    assert stream.tell()-start==packets[i]['packet_bytes'] and len(bits)==packets[i]['payload_bytes']
    rows.append(dict(dataset=task['dataset'],sequence=task['sequence'],variant=task['variant'],source_frame=v['source_frame_indices'][i],**packets[i],real_bits=packets[i]['packet_bytes']*8,**m))
    if i:
     fd=rt.metric_models[0](prevgt,gt)-rt.metric_models[0](prevout,out);fl=float(rt.metric_models[1](prevgt,prevout,fd,normalize=True).item());assert math.isfinite(fl)
     trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=fl))
    prevgt,prevout=gt,out
    if i%16==0:print('DEC',key(task),i,flush=True)
   for h in hooks:h.remove()
   frozen((ia,pa),before)
   assert metric_hashes==[eng.core.module_hash(m) for m in metrics]
  reconstruction=eng.eco.sha256_bytes(''.join(rec_hash).encode())
  assert stream.tell()==len(raw) and sum(p['packet_bytes'] for p in packets)==len(raw)
  result=dict(dataset=task['dataset'],sequence=task['sequence'],variant=task['variant'],kind=task['kind'],LPIPS=sum(r['LPIPS'] for r in rows)/64,DISTS=sum(r['DISTS'] for r in rows)/64,L1=sum(r['L1'] for r in rows)/64,PSNR=-10*math.log10(max(sum(r['pixel_SSE'] for r in rows)/(64*1080*1920*3),1e-15)),FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,total_bytes=len(raw),total_bits=len(raw)*8,actual_bpp=len(raw)*8/(64*1080*1920),reconstruction_sha256=reconstruction,stream_sha256=sha(stream_path),bytes_consumed=stream.tell())
  audits=dict(freeze='PASS',bit_accounting='PASS',encoder_decoder_state_sync='PASS',qp_in_stream='PASS',no_extra_reference_reset=True,optimizer_updates=0)
  if task['kind']=='baseline':
   ref=old['G_basic'];assert reconstruction==ref['reconstruction_sha256'],('baseline output mismatch',reconstruction,ref['reconstruction_sha256'])
   for k in ('LPIPS','DISTS','L1','PSNR','FloLPIPS','actual_bpp'):assert result[k]==ref[k],(k,result[k],ref[k])
   historical_rows=read(ref['frame_csv']);historical_trans=read(ref['transition_csv'])
   assert sha(ref['frame_csv'])==ref['frame_csv_sha256'] and sha(ref['transition_csv'])==ref['transition_csv_sha256']
   for a,b in zip(rows,historical_rows):
    for k in ('LPIPS','DISTS','L1','pixel_SSE','PSNR','real_bits','actual_qp'):assert float(a[k])==float(b[k]),(i,k,a[k],b[k])
   assert all(a['FloLPIPS']==float(b['FloLPIPS']) for a,b in zip(trans,historical_trans))
   audits['baseline_reproduction']='PASS'
  else:
   first=min(map(int,task['changes']))
   bp=load(ROOT/'tasks'/task['dataset']/task['sequence']/'baseline/proofs.json')
   assert encoder[:first]==bp['encoder'][:first] and decoder[:first]==bp['decoder'][:first] and packets[:first]==bp['packets'][:first]
   audits.update(prefix_equality='PASS',prefix_frames=first,variant_references_evolve_through_63=True)
  files=[stream_path]
  for name,obj in [('proofs.json',dict(encoder=encoder,decoder=decoder,packets=packets)),('audits.json',audits)]:
   p=dest.parent/name;dump(p,obj);files.append(p)
  for name,obj in [('frames.csv',rows),('transitions.csv',trans)]:
   p=dest.parent/name;write(p,obj);files.append(p)
  dump(dest,dict(status='PASS',signature=sig,task=task,result=result,audits=audits,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files],physical_gpu=gpu,started=started,finished=time.time(),model_hashes=before,metric_hashes=metric_hashes))
  print('PASS',key(task),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--variant',required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
 try:
  tasks=load(ROOT/'intervention_plan.json')+(load(ROOT/'transfer_plan.json') if (ROOT/'transfer_plan.json').exists() else [])
  task=next(t for t in tasks if t['sequence']==a.sequence and t['variant']==a.variant);run(task,a.gpu)
 except Exception:append(ROOT/'errors.jsonl',dict(time=time.time(),args=vars(a),error=traceback.format_exc()));raise
