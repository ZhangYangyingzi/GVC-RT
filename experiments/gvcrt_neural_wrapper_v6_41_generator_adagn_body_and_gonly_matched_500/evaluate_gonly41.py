"""Read-only G-only checkpoint evaluation on a verified historical stream."""
import argparse,csv,fcntl,json,math,os,sys,time,traceback,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500'
sys.path.insert(0,str(OLD))
from receiver37 import runtime_original,metadata,sourcepoint,parse,tag,th,sha
from intervene41 import atomic_json,atomic_csv
METHOD='G_fullLoss_Gonly_500'
ADA=('ada1','ada2','ada3','ada4','ada_final')

def run(task,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 import numpy as np,torch
 from adapter35 import frames_for,backend
 backend();torch.set_num_threads(2)
 key=tag(task);lockpath=HERE/'tasks'/f'gonly_{key}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  dest=HERE/'tasks'/f'gonly_{key}.json'
  if dest.exists() and json.loads(dest.read_text()).get('status')=='PASS':return
  idx=json.loads((HERE/'branches/G_fullLoss_Gonly_500/checkpoint_index.json').read_text())['500']
  cp=Path(idx['inference_path']);assert sha(cp)==idx['inference_sha256']
  source_path=OLD/'tasks'/f'replay_{key}.json';old=json.loads(source_path.read_text());assert old['status']=='PASS'
  archive=Path(old['latent_archive']['path']);assert sha(archive)==old['latent_archive']['sha256']
  lat=np.load(archive);assert lat['z0'].shape[0]==63
  eng,rt=runtime_original();v=metadata(task);frames=frames_for(v)
  im,pm=rt.models(1088,1920);decoder=copy.deepcopy(pm.recon_generation_net.decoder)
  weights=torch.load(cp,map_location='cpu',weights_only=True)
  bridge=copy.deepcopy(pm.recon_generation_net.mlp)
  bridge.load_state_dict(weights['bridge'],strict=True)
  assert eng.core.module_hash(bridge)==eng.core.module_hash(pm.recon_generation_net.mlp)
  decoder.load_state_dict(weights['generator'],strict=True);decoder.eval().requires_grad_(False)
  before=eng.core.module_hash(decoder);assert all(not p.requires_grad for p in decoder.parameters())
  original=sourcepoint(task,'O_original');stream=Path(original['bitstream_path'])
  assert sha(stream)==original['bitstream_sha256'] and stream.stat().st_size==int(original['real_bytes'])
  quality=rt.quality;flow,perceptual,inception=rt.metric_models
  frozen_before=[eng.core.module_hash(m) for m in (im,pm,*quality,flow,perceptual,inception)]
  rows=[];trans=[];features=[];real=[];hashes=[];previous=None;prevgt=None;paths=[];qchecks=[]
  tic=time.time();torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   for frame,sp,q,bits,realbits in parse(stream,eng.eco):
    assert q==json.loads((OLD/'qp_semantics_audit.json').read_text())['actual_qps'][str(task['qp'])][frame]
    target=frames[frame].cuda()
    if frame==0:raw=im.decompress(bits,sp,q)['x_hat']
    else:
     z=torch.from_numpy(lat['z0'][frame-1]).cuda();qr=pm.q_scale_recon[q:q+1]
     proof=old['tensor_equality'][frame-1]['decoded']['O_original']['q'];assert th(qr)==proof
     qchecks.append(dict(frame=frame,actual_qp=q,q_hash=proof['sha256']))
     seen=[];hooks=[]
     def gen_pre(_m,a):
      assert th(a[0])==th(z) and th(a[1])==th(qr)
     hooks.append(decoder.register_forward_pre_hook(gen_pre))
     for layer in ADA:
      def pre(_m,a,layer=layer):
       assert len(a)==2 and th(a[1])==th(z)
       seen.append(layer);paths.append(dict(frame=frame,layer=layer,stats_sha256=th(a[1])['sha256']))
      hooks.append(getattr(decoder,layer).register_forward_pre_hook(pre))
     try:raw=decoder(z,qr)
     finally:
      for h in hooks:h.remove()
     assert seen==list(ADA)
    hashes.append(eng.eco.tensor_sha(raw))
    output=eng.core.unit(raw,1080,1920);assert output.shape==target.shape and torch.isfinite(output).all()
    rows.append(dict(frame=frame,actual_qp=q,real_bits=realbits,**eng.eco.frame_metrics(output,target,quality)))
    features.append(inception(output).cpu().numpy());real.append(inception(target).cpu().numpy())
    if frame:
     gtflow=flow(prevgt,target);fd=gtflow-flow(previous,output)
     val=float(perceptual(prevgt,previous,fd,normalize=True).item());assert math.isfinite(val)
     trans.append(dict(from_frame=frame-1,to_frame=frame,FloLPIPS=val))
    previous=output;prevgt=target
    if v['video_index']==json.loads((OLD/'eval_manifest.json').read_text())[task['dataset']][0]['video_index'] and frame in (16,32,48,63):
     eng.eco.save_png(output,HERE/'artifacts/examples'/key/METHOD/f'{frame}.png')
  assert len(rows)==64 and len(trans)==len(qchecks)==63 and len(paths)==315
  assert before==eng.core.module_hash(decoder)
  assert frozen_before==[eng.core.module_hash(m) for m in (im,pm,*quality,flow,perceptual,inception)]
  fp=HERE/'features'/f'gonly_{key}.npz';fp.parent.mkdir(parents=True,exist_ok=True)
  tmp=fp.with_name(fp.name+f'.{os.getpid()}.tmp')
  with tmp.open('wb') as f:np.savez(f,real=np.concatenate(real),reconstruction=np.concatenate(features))
  tmp.replace(fp)
  atomic_csv(HERE/'raw'/f'gonly_{key}_frames.csv',rows)
  atomic_csv(HERE/'raw'/f'gonly_{key}_transitions.csv',trans)
  result=dict(dataset=task['dataset'],video=task['video'],sequence=v['name'],QP=task['qp'],method=METHOD,
   method_full=METHOD,actual_bpp=stream.stat().st_size*8/(64*1920*1080),real_bytes=stream.stat().st_size,
   frames=64,transitions=63,LPIPS=sum(r['LPIPS'] for r in rows)/64,
   DISTS=sum(r['DISTS'] for r in rows)/64,FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,
   reconstruction_sha256=eng.eco.sha256_bytes(''.join(hashes).encode()),
   feature_path=str(fp),feature_sha256=sha(fp),bitstream_path=str(stream),bitstream_sha256=sha(stream),
   source_rgb_sha256=v['rgb_sha256'],generator_module_hash=before,bridge_source='original z0',
   reuse_bitstream=True,new_real_encodes=0)
  atomic_json(dest,dict(status='PASS',task=task,result=result,source_replay_receipt=str(source_path),
   source_replay_sha256=sha(source_path),latent_archive=str(archive),latent_archive_sha256=sha(archive),
   training_checkpoint=str(cp),training_checkpoint_sha256=sha(cp),actual_qp_and_q_recon=qchecks,
   adagn_path_checks=len(paths),generator_hash_before=before,generator_hash_after=eng.core.module_hash(decoder),
   frozen_hashes_before=frozen_before,frozen_hashes_after=[eng.core.module_hash(m) for m in (im,pm,*quality,flow,perceptual,inception)],
   result_sha256=sha(fp),new_optimizer_updates=0,new_real_encodes=0,seconds=time.time()-tic,
   peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
  print('GONLY_EVAL_PASS',key,round(time.time()-tic,1),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);v=a.parse_args()
 try:run(json.loads(v.task),v.gpu)
 except Exception:
  with (HERE/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(stage='G_only_evaluation',task=json.loads(v.task),traceback=traceback.format_exc(),time=time.time()))+'\n')
  raise
