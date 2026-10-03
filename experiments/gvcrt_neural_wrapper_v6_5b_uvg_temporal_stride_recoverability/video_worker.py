"""One isolated video worker. Every pair gets fresh codec state and stream decode."""
import argparse,hashlib,math,traceback,time
from stride_io import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--video',required=True,choices=VIDEOS);p.add_argument('--gpu',required=True,type=int,choices=(4,5,6,7));a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 import torch,numpy as np,cv2
 from PIL import Image
 torch.set_num_threads(2);cv2.setNumThreads(1)
 v=next(v for v in videos() if v['name']==a.video)
 conf=cfg();e=engine();device=torch.device('cuda:0')
 arrays=source_frames(v)
 eq=[]
 for i in (0,4,32,64,128,192):
  canonical=V52/'canonical_sources/uvg'/v['name']/f'frame_{i//4:06d}.png'
  with Image.open(canonical) as im:expected=np.asarray(im,dtype=np.uint8)
  actual=arrays[i];maxerr=int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max())
  eq.append(dict(video=v['name'],original_frame=i,canonical_index=i//4,exact_equality=bool(np.array_equal(actual,expected)),max_abs_error=maxerr,original_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),canonical_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
 assert all(x['exact_equality'] and x['max_abs_error']==0 and x['original_sha256']==x['canonical_sha256']==v['frame_rgb_sha256'][x['canonical_index']] for x in eq)
 dump(ROOT/'audits'/f'rgb_equivalence_{v["name"]}.json',dict(status='PASS',checks=eq,conversion_command=load(V52/'ffmpeg_conversion_audit.json')['inherited_command']))
 tensor=lambda x:torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().div_(255)
 frames={i:tensor(x) for i,x in arrays.items()};gray,gradient,camera=motion_functions()
 quality=e.core.quality_models(device);flow,_=e.metrics.flo.load_models(device)
 wrappers={}
 for theta in ('V62','V64'):
  _,wrappers[theta]=e.eco.load_joint(conf['checkpoints'][theta]['path'],device)
  assert e.core.module_hash(wrappers[theta])==conf['checkpoints'][theta]['module_hashes']['wrapper']
 runtimes={m:e.Runtime(conf,m,device) for m in METHODS}
 visual=v['name'] in ('Beauty','Jockey','ShakeNDry')
 u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
 for t in ANCHORS:
  ref=frames[t].to(device);refa=arrays[t]
  for s in STRIDES:
   path=part(v,t,s)
   if path.exists():
    existing=load(path)
    if existing.get('status')=='PASS' and all(sha(r['bitstream_path'])==r['bitstream_sha256'] for r in existing['factorial']):continue
   target=frames[t+s].to(device);targeta=arrays[t+s];base=dict(video=v['name'],anchor=t,stride=s,equivalent_fps=120//s,delta_t_seconds=s/120)
   with torch.inference_mode():
    f=flow(ref,target);cam=camera(refa,targeta)
    assert cam['status']=='PASS',(v['name'],t,s,cam['status'])
    dif=target-ref
    difficulty=dict(base,temporal_RGB_L1=float(dif.abs().mean()),temporal_RGB_MSE=float(dif.square().mean()),flow_magnitude=float(f.square().sum(1).sqrt().mean()),compensated_residual_L1=cam['compensated_residual_L1'],compensated_residual_MSE=cam['compensated_residual_MSE'],camera_status=cam['status'])
    prox=[];gx=gray(targeta.astype(np.float32)/255);ex=gradient(gx)
    lx=float(cv2.Laplacian(gx,cv2.CV_32F,ksize=3).var());hx=float(np.square(gx-cv2.GaussianBlur(gx,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean());ee=float(np.square(ex).mean())
    for theta,wrapper in wrappers.items():
     y=wrapper(target);metrics=e.eco.frame_metrics(y,target,quality)
     yy=y[0].permute(1,2,0).float().cpu().numpy();gy=gray(yy);ey=gradient(gy)
     ly=float(cv2.Laplacian(gy,cv2.CV_32F,ksize=3).var());hy=float(np.square(gy-cv2.GaussianBlur(gy,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean());eyenergy=float(np.square(ey).mean())
     prox.append(dict(base,theta=theta,L1=float((y-target).abs().mean()),MSE=float((y-target).square().mean()),PSNR=metrics['PSNR'],LPIPS=metrics['LPIPS'],DISTS=metrics['DISTS'],SSIM=metrics['SSIM'],MS_SSIM=metrics['MS_SSIM'],Sobel_edge_L1=float(np.abs(ey-ex).mean()),edge_relative_energy_change=(eyenergy-ee)/ee if ee>0 else None,laplacian_ratio=ly/lx if lx>0 else None,HF_ratio=hy/hx if hx>0 else None,edge_energy_x=ee,edge_energy_proxy=eyenergy,laplacian_x=lx,laplacian_proxy=ly,HF_x=hx,HF_proxy=hy))
     if visual and t in (64,160):
      dest=ROOT/'visualizations'/v['name']/f'a{t:03d}_s{s}';e.eco.save_png(y,dest/f'{theta}_P_target.png');e.eco.save_png((y-target).abs(),dest/f'{theta}_P_abs_error.png')
    if visual and t in (64,160):
     dest=ROOT/'visualizations'/v['name']/f'a{t:03d}_s{s}';e.eco.save_png(ref,dest/'GT_reference.png');e.eco.save_png(target,dest/'GT_target.png')
   factorial=[]
   for method,runtime in runtimes.items():
    original_wrapper=runtime.wrapper
    if original_wrapper is not None:
     calls=[0]
     def target_only(x,model=original_wrapper,calls=calls):
      calls[0]+=1
      return x if calls[0]==1 else model(x)
     runtime.wrapper=target_only
    sub=Path(v['name'])/f'a{t:03d}_s{s}'/method
    stream=ROOT/'bitstreams'/sub.with_suffix('.bin');feature=ROOT/'features'/sub.with_suffix('.npz')
    recon=ROOT/'visualizations'/v['name']/f'a{t:03d}_s{s}'/method if visual and t in (64,160) else None
    try:r,rows=runtime.run([frames[t],frames[t+s]],4,stream,feature,recon_dir=recon)
    finally:runtime.wrapper=original_wrapper
    assert len(rows)==2 and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
    assert r['compression_hash_before']==r['compression_hash_after']==conf['compression_hash']
    assert [x['actual_qp'] for x in rows]==[4,load(V65/'qp_semantics_audit.json')['actual_qps']['4'][1]]
    assert rows[0]['real_bits']+rows[1]['real_bits']==r['real_bytes']*8
    transition=read(feature.with_suffix('.transitions.csv'));assert len(transition)==1
    vals={k:float(rows[1][k]) for k in METRICS if k!='FloLPIPS'};vals['FloLPIPS']=float(transition[0]['FloLPIPS'])
    factor=dict(base,method=method,external_qp=4,actual_I_qp=rows[0]['actual_qp'],actual_P_qp=rows[1]['actual_qp'],I_bytes=rows[0]['real_bits']//8,P_bytes=rows[1]['real_bits']//8,total_bytes=r['real_bytes'],P_bits=rows[1]['real_bits'],P_bpp=rows[1]['real_bits']/(1920*1080),equivalent_kbps=r['real_bytes']*8*(120/s)/2/1000,real_bytes=r['real_bytes'],bitstream_sha256=r['bitstream_sha256'],bitstream_path=r['bitstream_path'],real_RANS=True,independent_decode=True,state_sync_pass=True,compression_hash=r['compression_hash_before'],**vals)
    assert all(math.isfinite(z) for z in vals.values())
    factorial.append(factor)
    print('DONE',v['name'],t,s,method,'GPU',a.gpu,flush=True)
   own={r['method']:r for r in factorial};checks=[]
   for theta in ('V62','V64'):
    x,y=own[theta+'_P_only'],own[theta+'_full'];passed=x['real_bytes']==y['real_bytes'] and x['bitstream_sha256']==y['bitstream_sha256']
    checks.append(dict(base,theta=theta,method_a=x['method'],method_b=y['method'],bytes_a=x['real_bytes'],bytes_b=y['real_bytes'],sha_a=x['bitstream_sha256'],sha_b=y['bitstream_sha256'],status='PASS' if passed else 'FAIL'))
   assert all(x['status']=='PASS' for x in checks)
   dump(path,dict(status='PASS',difficulty=difficulty,proxy=prox,factorial=factorial,bitstream_checks=checks))
 dump(ROOT/'parts'/f'{v["name"]}_done.json',dict(status='PASS',pairs=24,methods=120))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'logs'/f'worker_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()))
  raise
