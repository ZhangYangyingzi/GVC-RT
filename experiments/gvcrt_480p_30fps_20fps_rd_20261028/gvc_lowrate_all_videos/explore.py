import argparse,os,sys,time,csv,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(__file__).resolve().parent;B=R.parent;sys.path.insert(0,str(B))
import torch
from worker480 import GVC,metric_setup,evaluate
from common480 import load,save,sha
assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
videos=load(B/'manifest.json')['videos'];videos=[v for i,v in enumerate(videos) if i%4==a.shard];models=metric_setup();thresholds=[.12,.15,.18,.21,.25,.28,.32,.36,.4,.45,.5,.55,.6,.7,.8,1.,2.,4.,8.]
methods=['C_C1_lr2_FD_1000','original'] if not a.smoke else ['C_C1_lr2_FD_1000']; thresholds=[1.] if a.smoke else thresholds
for v in videos:
    for method in methods:
     runtime=GVC(method)
     for th in thresholds:
      try:
       tag=f'{th:g}';p=R/'points'/v['dataset']/v['name']/method/f't{tag}.json';stream=R/'bitstreams'/v['dataset']/v['name']/method/f't{tag}.bin';feat=R/'features'/v['dataset']/v['name']/method/f't{tag}.npy'
       if p.exists() and load(p)['status']=='PASS':continue
       for im,pm in runtime.pairs:
        im.gaussian_encoder.force_zero_thres=th;pm.gaussian_encoder.force_zero_thres=th
       start=time.time();out,audit=runtime.run(v,0,stream);metrics,frames=evaluate(v,out,models,feat);del out
       p.parent.mkdir(parents=True,exist_ok=True);fpath=p.with_suffix('.frames.csv')
       with fpath.open('w',newline='') as f:w=csv.DictWriter(f,list(frames[0]));w.writeheader();w.writerows(frames)
       size=stream.stat().st_size;row=dict(status='PASS',dataset=v['dataset'],sequence=v['name'],method=method,intervention='shared_gaussian_force_zero_threshold_I_and_P',threshold=th,external_qp=0,frames=64,width=854,height=480,rate_accounting_fps=20,source_rgb_sha256=v['rgb_sha256'],real_bytes=size,kbps=size*8/64*20/1000,bitstream_path=str(stream),bitstream_sha256=sha(stream),encoder_decoder_state_hash_verified=True,audit=audit,per_frame_path=str(fpath),elapsed_seconds=time.time()-start,**metrics);save(p,row);print('PASS',v['dataset'],method,th,row['kbps'],row['PSNR'],row['LPIPS'],row['DISTS'],flush=True)
      except Exception:
       print('FAIL',method,th,traceback.format_exc(),flush=True);save(R/'errors'/v['dataset']/method/f't{th:g}.json',dict(error=traceback.format_exc(),threshold=th));break
     runtime.close();del runtime;torch.cuda.empty_cache()
