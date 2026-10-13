"""One old-stream UID: independent decode, float colors, fixed oracle correction."""
from audit31 import *
import argparse,fcntl,resource,traceback
def main(t,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from adapter31 import backend,runtime,frames_for,decode
 from color31 import color,residual,fit_stats,fit_finish,transform
 settings=backend();start=time.time();torch.cuda.reset_peak_memory_stats()
 lock=receipt(t).with_suffix('.lock');lock.parent.mkdir(exist_ok=True);lf=lock.open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if valid(t):return
 v=video(t['dataset'],t['video']);oldpath=oldpoint(t['dataset'],t['method'],t['video'],t['qp']);old=load(oldpath)
 assert sha(old['bitstream_path'])==old['bitstream_sha256'] and sha(old['feature_path'])==old['feature_sha256']
 eng,rt=runtime(t['method']);frozen=[tensor_hash(m.state_dict()) for m in (*rt.quality,*rt.metric_models)];wh=None if rt.wrapper is None else eng.core.module_hash(rt.wrapper)
 frames=frames_for(v);rows=[];pclip=[];corr=[];trans=[];fit=[];art={};colprev={};previous={}
 sums=torch.zeros((4,3),dtype=torch.float64,device='cuda');bias=torch.zeros(3,dtype=torch.float64,device='cuda');count=0;params=None
 fs={typ:[] for typ in ('RAW','BIAS','AFFINE')};real=[]
 # Raw FID is sliced from verified exact float feature rows, never old pooled FID.
 with np.load(old['feature_path']) as z:raw_real=z['real'].copy();raw_rec=z['reconstruction'].copy()
 def addcolor(kind,x,target,i):
  c=color(x,target);means={n:c[n+'_mean'] for n in ('R','G','B','Y','Cb','Cr')}
  delta={n+'_frame_mean_change':None if kind not in colprev else means[n]-colprev[kind][n] for n in means};colprev[kind]=means
  rows.append(dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],method=t['method'],QP=t['qp'],frame=i,source_frame_index=v['source_frame_indices'][i],UID=f'{v["name"]}:{v["source_frame_indices"][i]}',kind=kind,frame_group=('I0' if i==0 else 'P1-8' if i<=8 else 'P9-16' if i<=16 else 'P17-32' if i<=32 else 'P33-48' if i<=48 else 'P49-63'),**c,**delta))
 for i,x,bits in decode(eng,rt,old['bitstream_path'],old):
  target=frames[i].cuda();addcolor('reconstruction',x,target,i)
  if t['method']=='original' and t['qp']==0:addcolor('GT',target,target,i)
  if t['method']=='B1_initial' and t['qp']==0:
   px,ps=residual(rt.wrapper,target);addcolor('P',px,target,i)
   pclip.append(dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],frame=i,QP_independent=True,used_for_encoding=i>0,annotation='unused diagnostic P(I0)' if i==0 else 'P-frame input',**ps))
  if 1<=i<=31:
   s,n=fit_stats(x,target);sums+=s;count+=n;bias+=(target.double()-x.double()).mean((0,2,3))/31
   if i==31:
    params=fit_finish(sums,count,bias)
    for typ,p in params.items():fit.append(dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],type=typ,fit_positions=list(range(1,32)),eval_positions=list(range(32,64)),oracle=True,**p))
  if i>=32:
   assert params is not None
   real.append(raw_real[i:i+1]);fs['RAW'].append(raw_rec[i:i+1])
   for typ in ('RAW','BIAS','AFFINE'):
    out=transform(x,typ,params);lp=rt.quality[0](out,target,normalize=True).mean();ds=rt.quality[1](out,target).mean()
    corr.append(dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],method=t['method'],QP=t['qp'],type=typ,frame=i,UID=f'{v["name"]}:{v["source_frame_indices"][i]}',oracle=True,LPIPS=float(lp),DISTS=float(ds),**color(out,target)))
    if typ!='RAW':fs[typ].append(rt.metric_models[2](out).cpu().numpy())
    if typ in previous:
     pg,po=previous[typ];fd=rt.metric_models[0](pg,target)-rt.metric_models[0](po,out)
     assert torch.isfinite(fd).all() and fd.square().sum(1).sqrt().sum()>0
     value=float(rt.metric_models[1](pg,po,fd,normalize=True).item());assert np.isfinite(value)
     trans.append(dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],type=typ,from_frame=i-1,to_frame=i,FloLPIPS=value,oracle=True))
    previous[typ]=(target,out)
  # Fixed preview IDs only. PNG values never feed numeric computations.
  if t['video']==0 and t['qp']==5 and i in (16,32,63):
   import torch.nn.functional as F
   images=dict(GT=target,RAW=x)
   if rt.wrapper is not None:images['P']=rt.wrapper(target)
   if i>=32:images.update({typ:transform(x,typ,params) for typ in ('BIAS','AFFINE')})
   for typ,out in images.items():
    p=ROOT/'previews'/t['dataset']/t['method']/f'frame{i:02d}_{typ}.png';eng.eco.save_png(F.interpolate(out,size=(270,480),mode='area'),p);art[str(p)]=sha(p)
 assert len(corr)==96 and len(trans)==93 and len(fit)==2 and len(rows)>=64
 assert frozen==[tensor_hash(m.state_dict()) for m in (*rt.quality,*rt.metric_models)]
 assert wh==(None if rt.wrapper is None else eng.core.module_hash(rt.wrapper))
 base=ROOT/'raw/color'/uid(t)
 for suffix,data in (('.frames.csv',rows),('.fit.csv',fit),('.metrics.csv',corr),('.transitions.csv',trans)):
  p=base.with_suffix(suffix);write(p,data);art[str(p)]=sha(p)
 if pclip:
  p=base.with_suffix('.P.csv');write(p,pclip);art[str(p)]=sha(p)
 p=ROOT/'features/correction'/(uid(t)+'.npz');save_npz(p,real=np.concatenate(real),**{k:np.concatenate(a) for k,a in fs.items()});art[str(p)]=sha(p)
 assert all(a.shape==(32,2048) for a in [np.concatenate(real),*[np.concatenate(a) for a in fs.values()]])
 r=dict(status='PASS',task=t,signature=signature(t),artifacts=art,old_record_path=str(oldpath),old_record_sha256=sha(oldpath),stream_sha256=old['bitstream_sha256'],input_rgb_sha256=v['rgb_sha256'],decode=rt.decode_audit,correction_feature_path=str(p),frames=64,eval_frames=32,transitions_per_correction=31,wrapper_hash=wh,frozen_metric_hashes=frozen,backend=settings,physical_GPU=gpu,seconds=time.time()-start,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 dump(receipt(t),r);print('DONE',uid(t),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',required=True,type=int,choices=(4,5,6,7));a=ap.parse_args()
 try:main(json.loads(a.task),a.gpu)
 except Exception:error(dict(task=json.loads(a.task),traceback=traceback.format_exc()));raise
