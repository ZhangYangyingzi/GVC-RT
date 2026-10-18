from preflight import *
from data import Frames,load
import argparse,csv,math,importlib.util,resource
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def metric_models(device):
 from runtime import quality
 path=GVC/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py';s=importlib.util.spec_from_file_location('glc_transfer_official_fid',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 sys.path.insert(0,'/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback');import flolpips_compat as flo
 return quality(device),flo.load_models(device),m.InceptionPool3().to(device).eval().requires_grad_(False)
def run(dataset,video,branch,q,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from runtime import models,Sender,step,state_hash
 from src.entropy_coder import GaussianRansCodec,make_gop_stream,parse_gop_stream
 ident=f'{dataset}_{video}_{branch}_{q}';receipt=R/'results/points'/f'{ident}.json'
 v=load(R/'eval_manifest.json')[dataset][video];sig=dict(manifest=sha(R/'eval_manifest.json'),protocol=sha(R/'protocol.json'),runtime=sha(R/'runtime.py'),evaluation=sha(R/'evaluate.py'),models=load(R/'audits/checkpoint_audit.json')['models'])
 p=None
 if branch!='A_original':
  path=R/'final_P'/f'{branch}_1000.pt';sig['P_sha256']=sha(path);p=Sender('C_latent_P').cuda().eval();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['P'],strict=True)
 if receipt.exists():
  old=load(receipt)
  if old.get('signature')==sig and old['status']=='PASS' and all(Path(k).exists() and sha(k)==h for k,h in old['artifacts'].items()):return
 im,pm=models('cuda:0');frozen=[state_hash(m.state_dict()) for m in (im,pm)];ql,(flow,perceptual),inception=metric_models('cuda:0');f=Frames(v);coder=GaussianRansCodec();packets=[];hashes=[];dpb=None;tic=time.time()
 with torch.inference_mode():
  for i in range(64):
   e,out,dpb,pad=step(im,pm,f.tensor(i,'cuda:0'),dpb,q,i,p,mode='compress',coder=coder);packets.append(e['bit_stream']);hashes.append(state_hash({k:t for k,t in dpb.items() if k in ('ref_frame','ref_latent','ref_y') and t is not None}))
 data=make_gop_stream(v['height'],v['width'],packets);stream=R/'bitstreams'/f'{ident}.glc';stream.parent.mkdir(parents=True,exist_ok=True);stream.write_bytes(data);hh,ww,packets=parse_gop_stream(stream.read_bytes());assert (hh,ww)==(v['height'],v['width'])
 # No source/sender data are passed into the independent receiver.
 del im,pm;torch.cuda.empty_cache();im,pm=models('cuda:0');dpb=None;rows=[];trans=[];real=[];rec=[];prev=None
 with torch.inference_mode():
  for i,packet in enumerate(packets):
   e=im.decompress(packet,coder) if i==0 else pm.decompress(packet,dpb,coder)
   dpb=dict(ref_frame=e['x_hat'],ref_latent=e['ref_latent'],ref_y=None) if i==0 else e['dpb'];assert state_hash({k:t for k,t in dpb.items() if t is not None})==hashes[i]
   out=((dpb['ref_frame'].clamp(-1,1)+1)/2)[:,:,:v['height'],:v['width']];gt=f.tensor(i,'cuda:0');lp=float(ql[0](out,gt,normalize=True).mean());di=float(ql[1](out,gt).mean());rows.append(dict(frame=i,frame_type='I' if i==0 else 'P',q=q,LPIPS=lp,DISTS=di,L1=float((out-gt).abs().mean()),bytes=len(packet)))
   real.append(inception(gt).cpu().numpy());rec.append(inception(out).cpu().numpy())
   if prev is not None:
    diff=flow(prev[0],gt)-flow(prev[1],out);mag=diff.square().sum(1,keepdim=True).sqrt().sum();assert torch.isfinite(mag) and mag>0
    trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=float(perceptual(prev[0],prev[1],diff,normalize=True).item())))
   prev=(gt,out)
   if i in (16,32,63) and video==0 and q==28:
    from PIL import Image
    dest=R/'previews'/dataset;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(out[0].mul(255).round().byte().permute(1,2,0).cpu().numpy()).save(dest/f'{branch}_{i}.png')
 assert frozen==[state_hash(m.state_dict()) for m in (im,pm)]
 assert len(rows)==64 and len(trans)==63 and all(math.isfinite(float(vv)) for row in rows+trans for k,vv in row.items() if k in ('LPIPS','DISTS','FloLPIPS'))
 feature=R/'features'/f'{ident}.npz';feature.parent.mkdir(parents=True,exist_ok=True);np.savez(feature,real=np.concatenate(real),reconstruction=np.concatenate(rec));frame=R/'results/raw'/f'{ident}.frames.csv';transition=frame.with_suffix('.transitions.csv');csvwrite(frame,rows);csvwrite(transition,trans)
 result=dict(dataset=dataset,video=video,name=v['name'],method=branch,q=q,frames=64,transitions=63,bytes=len(data),bpp=len(data)*8/(64*v['height']*v['width']),rate_type='real_local_nonofficial_rans',rate_accounting_fps=v['rate_accounting_fps'],kbps=len(data)*8*v['rate_accounting_fps']/64/1000,LPIPS=sum(x['LPIPS'] for x in rows)/64,DISTS=sum(x['DISTS'] for x in rows)/64,LPIPS_P_only=sum(x['LPIPS'] for x in rows[1:])/63,DISTS_P_only=sum(x['DISTS'] for x in rows[1:])/63,FloLPIPS=sum(x['FloLPIPS'] for x in trans)/63,feature=str(feature),per_frame=str(frame),per_transitions=str(transition),seconds=time.time()-tic,peak_MiB=torch.cuda.max_memory_reserved()/2**20,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 save(str(receipt.relative_to(R)),dict(status='PASS',signature=sig,result=result,independent_decode=True,frozen_unchanged=True,artifacts={str(x):sha(x) for x in (feature,stream,frame,transition)}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);args=a.parse_args()
 try:run(**json.loads(args.task),gpu=args.gpu)
 except Exception:print(traceback.format_exc());sys.exit(1)
