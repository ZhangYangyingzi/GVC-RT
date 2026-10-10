"""Historical V6.20 real RANS path; sequence length extends without a state reset."""
from io21 import *
import argparse,traceback

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--video',type=int,required=True);a=ap.parse_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 import torch,numpy as np
 from adapter import engine,frames_for
 torch.set_num_threads(2);torch.manual_seed(20261010);np.random.seed(20261010);cfg=frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
 v=next(v for v in sources(a.dataset)if v['video_index']==a.video);pending=[]
 for q in range(10):
  p=point(a.dataset,a.method,a.video,q)
  if p.exists():
   r=load(p);assert sha(r['feature_path'])==r['feature_sha256'] and sha(r['bitstream_path'])==r['bitstream_sha256'] and r['cache_key']==key(v,q,a.method)
  else:pending.append(q)
 if not pending:return
 eng=engine();cp={}if a.method=='original'else{a.method:cfg['checkpoints'][a.method]};runtime=eng.Runtime(dict(experiment='B',methods={a.method:[None,None]if a.method=='original'else[a.method,a.method]},checkpoints=cp,force_zero_thres=.12),a.method,torch.device('cuda:0'));frames=frames_for(v)
 for q in pending:
  torch.cuda.reset_peak_memory_stats();t=time.time();rel=Path(a.dataset)/a.method/v['name']/f'qp{q}';stream=(ROOT/'bitstreams'/rel).with_suffix('.bin');feature=(ROOT/'features'/rel).with_suffix('.npz');r,rows=runtime.run(frames,q,stream,feature)
  old=load(V20/'parts'/a.dataset/a.method/f'video_{a.video:02d}_qp{q}.json');oldbytes=Path(old['bitstream_path']).read_bytes();assert stream.read_bytes()[:len(oldbytes)]==oldbytes,('first64 bitstream mismatch',a.dataset,a.method,a.video,q)
  for x,y in zip(rows[:64],read(old['frame_metrics_path'])):
   for k in ['actual_qp','real_bits']:assert int(x[k])==int(y[k])
   for k in ['LPIPS','DISTS','PSNR','SSIM','MS_SSIM']:assert abs(float(x[k])-float(y[k]))<=1e-6
  assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['frames']==v['frames'] and r['bytes_consumed']==r['real_bytes']
  ff=point(a.dataset,a.method,a.video,q).with_suffix('.frames.csv');write(ff,rows)
  r.update(dataset=a.dataset,method=a.method,video_index=a.video,sequence=v['name'],QP=q,external_qp=q,source_sha256=v['source_sha256'],source_frame_indices=v['source_frame_indices'],source_rgb_sha256=v['rgb_sha256'],checkpoint_sha256=''if a.method=='original'else cp[a.method]['sha256'],rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,actual_qps=[x['actual_qp']for x in rows],historical_64_prefix_bitstream_verified=True,historical_64_frame_metrics_verified=True,frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),cache_key=key(v,q,a.method),physical_gpu=a.gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.time()-t)
  dump(point(a.dataset,a.method,a.video,q),r);print('DONE',a.dataset,a.method,a.video,q,'frames',v['frames'],'peak',r['peak_memory_MiB'],flush=True)
 frozen()
def key(v,q,m):return hashlib.sha256(json.dumps(dict(checkpoint=''if m=='original'else evalcfg()['checkpoints'][m]['sha256'],I_mode=evalcfg()['i_frame_modes'][m],source_RGB=v['rgb_sha256'],indices=v['source_frame_indices'],QP=q,protocol=sha(ROOT/'protocol.json')),sort_keys=True).encode()).hexdigest()
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'failure_{os.getpid()}.json',dict(error=traceback.format_exc()));raise
