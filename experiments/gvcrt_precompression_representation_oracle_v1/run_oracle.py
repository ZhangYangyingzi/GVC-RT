#!/usr/bin/env python3
import argparse,csv,gc,hashlib,io,json,math,os,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];MATCH=ROOT.parent/'gvcrt_vs_dcvc_rt_matched_rate'
V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src'
for p in (REPO,V9):sys.path.insert(0,str(p))
from common import sha256
from gvc_hooks import load_models,INDEX_MAP
from src.layers.cuda_inference import round_and_to_int8
from src.utils.stream_helper import write_sps,write_ip

CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text())
SPS={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0}
def tag(v):return f"{v['dataset']}_{int(v['video_id']):02d}"
def digest(b):return hashlib.sha256(b).hexdigest()
def write_csv(p,rows):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def model_hash(*ms):
 h=hashlib.sha256()
 for m in ms:
  for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def source_frames(v,dev):
 d=MATCH/'source_frames'/tag(v);out=[]
 for i in range(1,17):
  a=np.asarray(Image.open(d/f'im{i}.png').convert('RGB'),dtype=np.uint8).copy()
  out.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(dev).float()/255)
 return out
def codec_input(x):return F.pad(x,(0,0,0,8),mode='replicate').to(torch.float16)*2-1
def unit(x):return (x[:,:,:1080,:1920].float().clamp(-1,1)+1)/2
def basic(x,gt):
 d=x-gt;sse=float(d.square().sum());mse=float(d.square().mean());psnr=-10*math.log10(max(mse,1e-15));ux=x.mean();uy=gt.mean();vx=((x-ux)**2).mean();vy=((gt-uy)**2).mean();cov=((x-ux)*(gt-uy)).mean();ss=float(((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2)));return sse,mse,psnr,ss
def perceptual(a,b,mods,resize=None):
 if resize:a=F.interpolate(a,resize,mode='bilinear',align_corners=False);b=F.interpolate(b,resize,mode='bilinear',align_corners=False)
 return mods[0](a,b,normalize=True).mean()+mods[1](a,b).mean()
def prepare_p(m,qp):
 qf=m.q_scale_feature[qp:qp+1];qe=m.q_scale_enc[qp:qp+1];qd=m.q_scale_dec[qp:qp+1];qr=m.q_scale_recon[qp:qp+1]
 f=m.apply_feature_adaptor();ctx,ct=m.feature_extractor(f,qf);return ctx,ct,qe,qd,qr
def ste_forward_p(m,xmaster,qp,gt,mods):
 ctx,ct,qe,qd,qr=prepare_p(m,qp);x=codec_input(xmaster);y=m.enc(x,ctx,qe);z=m.hyper_enc(m.pad_for_y(y));zh=torch.clamp(torch.round(z),-128,127);zs=z+(zh-z).detach();params=m.res_prior_param_decoder(zs,ct)
 qdec,sc,mu=params.chunk(3,1);qdec=torch.clamp_min(qdec,.5);ys=y*torch.reciprocal(qdec);B,C,H,W=ys.shape;m0,m1=m.get_mask_2x(B,C,H,W,ys.dtype,ys.device)
 def proc(a,s,u,mask):
  sh=s*mask;uh=u*mask;r=(a-uh)*mask;hard=torch.clamp(torch.round(r),-128,127);active=mask.bool()&(sh>.12);hard=hard*active;ste=r+(hard-r).detach();return hard,ste,sh,uh,active
 h0,q0,s0,u0,a0=proc(ys,sc,mu,m0);yh0=q0+u0;sc1,mu1=m.y_spatial_prior(torch.cat((yh0,params),1)).chunk(2,1);h1,q1,s1,u1,a1=proc(ys,sc1,mu1,m1);yhat=(yh0+q1+u1)*qdec
 out=m.recon_generation_net(m.dec(yhat,ctx,qd),qr);uo=unit(out);d=perceptual(uo,gt,mods,tuple(CFG['optimization_metric_resize']))
 zu=m.bit_estimator_z.get_cdf(zs.float()+.5,qp);zl=m.bit_estimator_z.get_cdf(zs.float()-.5,qp);rate=(-torch.log2((zu-zl).clamp_min(1e-9))).sum()
 def gb(q,s,act):
  q=q.float()[act];s=s.float()[act].clamp(.11,16);idx=((torch.log(s)-m.gaussian_encoder.log_scale_min)*m.gaussian_encoder.log_step_recip).long().clamp(0,127);sq=m.gaussian_encoder.scale_table.to(s.device)[idx];n=torch.distributions.Normal(torch.zeros_like(sq),sq);return (-torch.log2((n.cdf(q+.5)-n.cdf(q-.5)).clamp_min(1e-9))).sum()
 rate=rate+gb(q0,s0,a0)+gb(q1,s1,a1)
 with torch.no_grad():
  zt,_=round_and_to_int8(z);pp=m.res_prior_param_decoder(zt,ct);_,_,_,_,true=m.compress_prior_2x(y,pp,m.y_spatial_prior);diff=float((yhat.detach()-true).abs().max())
 return {'D':d,'R':rate,'out':uo,'ste_diff':diff}
def metric_models(dev):
 import lpips
 from DISTS_pytorch import DISTS
 return lpips.LPIPS(net='alex',verbose=False).to(dev).eval().requires_grad_(False),DISTS().to(dev).eval().requires_grad_(False)
def eval_frame(out,gt,mods):
 sse,mse,psnr,ss=basic(out,gt)
 with torch.inference_mode():lp=float(mods[0](out,gt,normalize=True));di=float(mods[1](out,gt))
 return {'pixel_SSE':sse,'MSE':mse,'PSNR':psnr,'SSIM':ss,'LPIPS':lp,'DISTS':di}
def save_png(x,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);a=torch.clamp(x*255,0,255).round().byte()[0].permute(1,2,0).cpu().numpy();Image.fromarray(a).save(p,compress_level=1)
def aggregate(rows,v,nbytes):
 pixels=16*3*1080*1920;sse=sum(float(x['pixel_SSE']) for x in rows);mse=sse/pixels
 return {'bytes':nbytes,'bits':nbytes*8,'bpp':nbytes*8/(16*1080*1920),'kbps':nbytes*8*float(v['fps'])/16/1000,'total_pixel_SSE':sse,'aggregate_MSE':mse,'sequence_PSNR':-10*math.log10(max(mse,1e-15)),'mean_PSNR':sum(float(x['PSNR']) for x in rows)/16,'SSIM':sum(float(x['SSIM']) for x in rows)/16,'LPIPS':sum(float(x['LPIPS']) for x in rows)/16,'DISTS':sum(float(x['DISTS']) for x in rows)/16}
def run_stream(v,frames,dev,mods,kind,param):
 im,enc=load_models(dev);_,dec=load_models(dev);before=model_hash(im,enc,dec);enc.clear_dpb();dec.clear_dpb();enc.set_curr_poc(0);dec.set_curr_poc(0)
 stream=io.BytesIO();write_sps(stream,SPS);rows=[];tr=[];sync=[];proxies=[];recons=[]
 for fi,gt in enumerate(frames):
  aq=1 if fi==0 else enc.shift_qp(1,INDEX_MAP[fi%8])
  if fi==0:
   proxy=gt.detach();e=im.compress(codec_input(proxy),aq);payload=e['bit_stream'];enc.add_ref_frame(None,e['x_hat']);out=im.decompress(payload,SPS,aq)['x_hat'];dec.add_ref_frame(None,out);is_i=True;ste=0.0
  else:
   if kind=='oracle':
    proxy=torch.nn.Parameter(gt.detach().clone().float());base=ste_forward_p(enc,proxy,aq,gt,mods);r0=base['R'].detach();d0=base['D'].detach();opt=torch.optim.Adam([proxy],lr=CFG['learning_rate'])
    for step in range(CFG['optimization_steps']):
     opt.zero_grad(set_to_none=True);z=ste_forward_p(enc,proxy,aq,gt,mods);rn=z['R']/(1080*1920);loss=z['D']+float(param)*rn
     if not torch.isfinite(loss):raise RuntimeError('nonfinite oracle loss')
     loss.backward();torch.nn.utils.clip_grad_norm_([proxy],CFG['gradient_clip_norm']);opt.step()
     with torch.no_grad():proxy.clamp_(0,1)
     tr.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'kind':kind,'parameter':param,'frame':fi,'step':step,'loss':float(loss),'D_perceptual':float(z['D']),'R_est_bits':float(z['R']),'R_est_bpp':float(rn),'grad_norm':float(proxy.grad.norm()) if proxy.grad is not None else 0,'proxy_delta_l2':float((proxy-gt).norm()),'ste_forward_max_abs':z['ste_diff']})
    ste=ste_forward_p(enc,proxy,aq,gt,mods)['ste_diff'];proxy=proxy.detach()
   elif kind=='blur':
    sigma=float(param);a=torch.clamp(gt*255,0,255).round().byte()[0].permute(1,2,0).cpu().numpy();b=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=sigma)),dtype=np.uint8).copy();proxy=torch.from_numpy(b).permute(2,0,1).unsqueeze(0).to(dev).float()/255;ste=''
   else:proxy=gt.detach();ste=''
   e=enc.compress(codec_input(proxy),aq);payload=e['bit_stream'];out=dec.decompress(payload,SPS,aq)['x_hat'];is_i=False
   state=float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max())
   sync.append({'dataset':v['dataset'],'video':v['name'],'kind':kind,'parameter':param,'frame':fi,'state_max_abs':state,'status':'PASS' if state==0 else 'FAIL'})
  write_ip(stream,is_i,0,aq,payload);u=unit(out);m=eval_frame(u,gt,mods);pm=eval_frame(proxy,gt,mods);rows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'kind':kind,'parameter':param,'frame':fi,'actual_qp':aq,'payload_bytes':len(payload),**m,'proxy_PSNR':pm['PSNR'],'proxy_LPIPS':pm['LPIPS'],'proxy_DISTS':pm['DISTS'],'ste_forward_max_abs':ste});proxies.append(proxy.detach().cpu());recons.append(u.detach().cpu())
 data=stream.getvalue();after=model_hash(im,enc,dec)
 if before!=after:raise RuntimeError('model parameter hash changed')
 return data,rows,tr,sync,proxies,recons,before
def run_video(v,dev):
 t=tag(v);done=ROOT/'parts'/f'{t}_done.json'
 if done.exists() and json.loads(done.read_text()).get('status')=='PASS':print('reuse',t,flush=True);return
 frames=source_frames(v,dev);mods=metric_models(dev);allrows=[];traces=[];sync=[];summ=[];art={}
 specs=[('baseline','original')]+[('oracle',b) for b in CFG['betas']]+[('blur',s) for s in CFG['blur_sigmas']]
 for kind,param in specs:
  print(t,kind,param,flush=True);data,rows,tr,sy,px,rc,mh=run_stream(v,frames,dev,mods,kind,param);name=f'{t}_{kind}_{param}'.replace('.','p');bp=ROOT/'bitstreams'/f'{name}.bin';bp.parent.mkdir(parents=True,exist_ok=True);bp.write_bytes(data);ag=aggregate(rows,v,len(data));summ.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'kind':kind,'parameter':param,**ag,'bitstream_path':str(bp),'bitstream_sha256':digest(data),'decode_status':'PASS','model_hash':mh});allrows+=rows;traces+=tr;sync+=sy;art[(kind,str(param))]=(px,rc,name)
  write_csv(ROOT/'parts'/f'{t}_summaries.partial.csv',summ)
 base=next(x for x in summ if x['kind']=='baseline')
 for x in summ:x['actual_rate_ratio']=float(x['bytes'])/float(base['bytes'])
 selected=[]
 for target in CFG['target_rate_ratios']:
  feasible=[x for x in summ if x['kind']=='oracle' and float(x['bytes'])<=target*float(base['bytes'])]
  if feasible:
   best=min(feasible,key=lambda x:float(x['LPIPS'])+float(x['DISTS']));status=True
   px,rc,name=art[('oracle',str(best['parameter']))]
   for fi,(pimg,rimg) in enumerate(zip(px,rc)):save_png(pimg,ROOT/'proxy_frames'/t/f'target_{target}'/f'frame_{fi:06d}.png');save_png(rimg,ROOT/'reconstruction_frames'/t/f'target_{target}'/f'frame_{fi:06d}.png')
  else:best={k:'' for k in base};status=False
  selected.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'target_ratio':target,'baseline_bytes':base['bytes'],'baseline_kbps':base['kbps'],'baseline_LPIPS':base['LPIPS'],'baseline_DISTS':base['DISTS'],'baseline_PSNR':base['sequence_PSNR'],'proxy_parameter':best.get('parameter',''),'proxy_bytes':best.get('bytes',''),'proxy_kbps':best.get('kbps',''),'actual_rate_ratio':best.get('actual_rate_ratio',''),'proxy_output_LPIPS':best.get('LPIPS',''),'proxy_output_DISTS':best.get('DISTS',''),'proxy_output_FloLPIPS':'','proxy_output_PSNR':best.get('sequence_PSNR',''),'delta_LPIPS_vs_baseline':float(best['LPIPS'])-float(base['LPIPS']) if status else '','delta_DISTS_vs_baseline':float(best['DISTS'])-float(base['DISTS']) if status else '','delta_PSNR_vs_baseline':float(best['sequence_PSNR'])-float(base['sequence_PSNR']) if status else '','target_reached':status,'bitstream_path':best.get('bitstream_path',''),'decode_status':best.get('decode_status','')})
 write_csv(ROOT/'parts'/f'{t}_summaries.csv',summ);write_csv(ROOT/'parts'/f'{t}_frames.csv',allrows);write_csv(ROOT/'parts'/f'{t}_traces.csv',traces);write_csv(ROOT/'parts'/f'{t}_sync.csv',sync);write_csv(ROOT/'parts'/f'{t}_selected.csv',selected)
 done.write_text(json.dumps({'status':'PASS','video':t,'candidates':len(summ),'model_hashes_unchanged':True,'causal_sync':all(x['status']=='PASS' for x in sync)},indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--videos',required=True);ap.add_argument('--gate-only',action='store_true');a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.manual_seed(CFG['random_seed']);dev=torch.device('cuda:0');lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
 for s in a.videos.split(','):
  ds,vid=s.split(':');v=lookup[(ds,int(vid))]
  try:run_video(v,dev)
  except Exception as e:(ROOT/'parts'/f'{tag(v)}_FAILED.json').write_text(json.dumps({'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
if __name__=='__main__':main()
