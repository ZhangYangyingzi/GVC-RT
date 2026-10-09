"""Resumable whole-sequence real encode and independent decode."""
import argparse,fcntl,math,traceback
from io19 import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--dataset',choices=DATASETS,required=True);ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--video',type=int,required=True);ap.add_argument('--qps',default='0,1,2,3,4,5,6,7,8,9');a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np
    from adapter import frames_for,engine,validate,cache_key
    from prepare import selected
    torch.set_num_threads(2);cfg=frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    (ROOT/'parts').mkdir(exist_ok=True);lock=(ROOT/'parts'/f'{a.dataset}_{a.method}_{a.video}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v=next(v for v in sources(a.dataset) if v['video_index']==a.video);pending=[]
    for q in map(int,a.qps.split(',')):
        p=point(a.dataset,a.method,a.video,q)
        if p.exists():validate(load(p),v,q,a.method)
        else:pending.append(q)
    if not pending:return
    eng=engine();cp={} if a.method=='original' else {a.method:cfg['checkpoints'][a.method]};runtime=eng.Runtime(dict(experiment='B',methods={a.method:[None,None] if a.method=='original' else [a.method,a.method]},checkpoints=cp,force_zero_thres=.12),a.method,torch.device('cuda:0'));frames=frames_for(v);torch.cuda.reset_peak_memory_stats()
    for q in pending:
        t=time.time();relative=Path(a.dataset)/a.method/v['name']/f'qp{q}';stream=(ROOT/'bitstreams'/relative).with_suffix('.bin');feature=(ROOT/'features'/relative).with_suffix('.npz');runtime.capture_first=selected(v) and q in (0,4,9)
        r,rows=runtime.run(frames,q,stream,feature);out=point(a.dataset,a.method,a.video,q);ff=out.with_suffix('.frames.csv');write(ff,rows)
        with np.load(feature) as f:fid=float(eng.metrics.fid.low_rank_fid(f['real'],f['reconstruction']));assert math.isfinite(fid)
        trans=feature.with_suffix('.transitions.csv');r.update(dataset=a.dataset,method=a.method,video_index=a.video,sequence=v['name'],external_qp=q,QP=q,checkpoint_sha256='' if a.method=='original' else cp[a.method]['sha256'],checkpoint_path='' if a.method=='original' else cp[a.method]['path'],source_sha256=v['source_sha256'],source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],source_frame_indices_sha256=hashlib.sha256(json.dumps(v['source_frame_indices'],separators=(',',':')).encode()).hexdigest(),rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,total_bits=r['real_bytes']*8,FID=fid,FID_num_samples=64,force_zero_thres=.12,frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),transitions_path=str(trans),transitions_sha256=sha(trans),actual_qps=[x['actual_qp'] for x in rows],physical_gpu=a.gpu,fresh_run=True,temporal_resampling=False,elapsed_seconds=time.time()-t,config_sha256=sha(ROOT/'config.json'),evaluator_sha256=sha(BASE/'evaluate.py'),visualization_dir='',visualization_sha256=[],cache_key=cache_key(v,q,a.method),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20)
        if a.method!='B_i_bypass':
            baseline=load(V18/'parts'/a.dataset/('B_1000' if a.method=='B_standard' else 'original')/out.name)
            for k in ('real_bytes','bitstream_sha256','reconstruction_sha256','actual_qps','LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM'):assert r[k]==baseline[k],('standard reproduction mismatch',k,r[k],baseline[k])
            r['historical_reproduction_verified']=True
        validate(r,v,q,a.method);dump(out,r);print('DONE',a.dataset,a.method,a.video,q,'peak',r['peak_memory_MiB'],flush=True)
    frozen()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'failure_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
