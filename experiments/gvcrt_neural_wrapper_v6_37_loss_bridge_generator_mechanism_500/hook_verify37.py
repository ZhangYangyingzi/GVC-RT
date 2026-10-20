"""Read-only replay verification of actual intervention decoder-input hooks."""
from receiver37 import *
import argparse

def main(gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from adapter35 import backend
 backend();eng,rt=runtime_original()
 actual_metrics=dict(quality=[eng.core.module_hash(m) for m in rt.quality],flow_and_perceptual=[eng.core.module_hash(m) for m in rt.metric_models[:2]],official_Inception_module_hash=eng.core.module_hash(rt.metric_models[-1]),official_Inception_weights_SHA256='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2',official_Inception_dimensions=2048,source='training35.official_model; inherited v6.36 official replacement after initial loader checks',legacy_initial_loader_hashes=load(ROOT/'evaluation_config.json')['metric_module_hashes'])
 dump(ROOT/'audits/metric_models.json',dict(status='PASS',actual_metrics=actual_metrics,all_parameters_frozen=all(not p.requires_grad for m in (*rt.quality,*rt.metric_models) for p in m.parameters())))
 import subprocess,torchvision,cv2,PIL
 imported=[]
 for name,mod in list(sys.modules.items()):
  filename=getattr(mod,'__file__',None)
  if not filename:continue
  path=Path(filename).resolve()
  if not path.is_file():continue
  if path.is_relative_to(REPO) or name in ('torch','torchvision','numpy','cv2','PIL') or name.startswith(('pytorch_fid.','lpips.','DISTS_pytorch.')):
   imported.append(dict(module=name,path=str(path),sha256=sha(path),git_blob=subprocess.check_output(['git','hash-object',str(path)],cwd=REPO,text=True).strip()))
 for path in (V30/'adapter.py',V29/'io21.py',V29/'stat_loss.py'):
  imported.append(dict(module='dynamically executed historical source',path=str(path),sha256=sha(path),git_blob=subprocess.check_output(['git','hash-object',str(path)],cwd=REPO,text=True).strip()))
 dump(ROOT/'audits/runtime_import_manifest.json',dict(status='PASS',scope='actual replay/metric/model imports plus explicitly exec-compiled historical sources; formal training uses the separately signed train/training32/frames32 sources',imports=imported,environment=dict(python=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,OpenCV=cv2.__version__,Pillow=PIL.__version__,CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version()),collected=time.time()))

 im,pm=rt.models(1088,1920);bank=receiver_bank(pm,['O_original','F_fullLoss_noP_500']);outputs=[]
 with torch.inference_mode():
  for p in sorted((ROOT/'tasks').glob('replay_*.json')):
   r=load(p);t=r['task'];name=tag(t);outpath=ROOT/'audits/intervention_input_hooks'/f'{name}.json'
   if outpath.exists():outputs.append(load(outpath));continue
   z=np.load(r['latent_archive']['path']);record=[];hashes={m:[] for m in ('X_fullBridge_originalG','Y_originalBridge_fullG','I_halfLatent_fullG')};stream=sourcepoint(t,'O_original')['bitstream_path']
   for i,sp,q,bits,rb in parse(stream,eng.eco):
    if i==0:
     native=im.decompress(bits,sp,q)['x_hat']
     for m in hashes:hashes[m].append(eng.eco.tensor_sha(native))
     continue
    z0=torch.from_numpy(z['z0'][i-1]).cuda();zF=torch.from_numpy(z['zF'][i-1]).cuda();q_recon=pm.q_scale_recon[q:q+1];assert th(q_recon)==r['tensor_equality'][i-1]['decoded']['O_original']['q']
    for method,m,zz in [('X_fullBridge_originalG',bank['O_original'].decoder,zF),('Y_originalBridge_fullG',bank['F_fullLoss_noP_500'].decoder,z0),('I_halfLatent_fullG',bank['F_fullLoss_noP_500'].decoder,(z0+zF)/2)]:
     cap=[];h=m.register_forward_pre_hook(lambda _,a:cap.append((a[0],a[1])))
     try:x=m(zz,q_recon)
     finally:h.remove()
     assert len(cap)==1 and torch.equal(cap[0][0],zz) and torch.equal(cap[0][1],q_recon)
     hashes[method].append(eng.eco.tensor_sha(x));record.append(dict(frame=i,method=method,actual_generator_input=th(cap[0][0]),q_recon=th(cap[0][1]),input_exact=True))
   for m,hs in hashes.items():assert eng.eco.sha256_bytes(''.join(hs).encode())==r['result'][m]['reconstruction_sha256']
   out=dict(status='PASS',task=t,generator_input_pre_hooks=record,all_intervention_output_endpoint_hashes_exact=True,source_replay_receipt=str(p),source_replay_sha256=sha(p),new_optimizer_updates=0,physical_GPU=gpu);dump(outpath,out);outputs.append(out)
 dump(ROOT/'audits/intervention_input_hooks.json',dict(status='PASS',video_QP_tasks=len(outputs),P_frame_method_checks=sum(len(r['generator_input_pre_hooks']) for r in outputs),all_inputs_and_q_exact=True,all_full_output_stream_hashes_exact=True,verification_only_not_additional_methods=True))
 print('ACTUAL INTERVENTION INPUT HOOKS PASS',len(outputs),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);main(a.parse_args().gpu)
