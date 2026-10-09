import argparse,fcntl,sys,time,os
from parallel_utils import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=('A','B'),required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);args=p.parse_args()
    root=exp(args.experiment);config=load(root/'config.json');assert args.gpu in config['gpus']
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from engine import Runtime
    from proxy_stats import compute
    torch.set_num_threads(4);assert torch.cuda.is_available() and torch.cuda.device_count()==1
    device=torch.device('cuda:0');verify_frozen(config)
    datasets=list(dict.fromkeys(v['dataset'] for v in config['videos']))
    tasks=[(d,m) for m in config['methods'] for d in datasets]
    for task,(dataset,method) in enumerate(tasks):
        if task%args.shards!=args.shard:continue
        runtime=None
        for video in [v for v in config['videos'] if v['dataset']==dataset]:
            pending=[]
            for q in config['qps']:
                target=point(root,dataset,method,video['video_index'],q)
                if target.exists():
                    r=load(target);assert r['protocol']==args.experiment and r['source_rgb_sha256']==video['rgb_sha256']
                    for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                else:pending.append(q)
            if not pending:continue
            if runtime is None:runtime=Runtime(config,method,device)
            frames=frames_for(video)
            if args.experiment=='B' and runtime.sender:
                compute(root,video,runtime.sender,runtime.wrapper,frames,device)
            for q in pending:
                target=point(root,dataset,method,video['video_index'],q);target.parent.mkdir(parents=True,exist_ok=True)
                with target.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if target.exists():continue
                    start=time.time();print('START',dataset,method,video['video_index'],q,flush=True)
                    relative=Path(dataset)/method/f"video_{video['video_index']}_qp{q}"
                    stream=(root/'bitstreams'/relative).with_suffix('.bin');feature=(root/'features'/relative).with_suffix('.npz')
                    recon=root/'reconstructions'/relative if args.experiment=='A' and q in (0,3,6,9) else None
                    result,frame_rows=runtime.run(frames,q,stream,feature,recon)
                    framepath=target.with_suffix('.frames.csv');write(framepath,frame_rows)
                    result.update(protocol=args.experiment,dataset=dataset,group=video['group'],video=video['name'],video_index=video['video_index'],method=method,
                        external_qp=q,actual_i_qp=q,actual_p_qp=sorted({r['actual_qp'] for r in frame_rows[1:]}),source_rgb_sha256=video['rgb_sha256'],
                        source_path=video.get('path',video.get('canonical_dir')),P=runtime.sender or 'identity',receiver=runtime.receiver or 'original',
                        physical_gpu=args.gpu,start_unix=start,end_unix=time.time(),elapsed_seconds=time.time()-start,
                        frame_metrics_path=str(framepath),frame_metrics_sha256=sha(framepath),config_sha256=sha(root/'config.json'))
                    if args.experiment=='B' and method in ('ORIGINAL','V41_FULL','CLIP8_FULL'):
                        oldmethod={'ORIGINAL':'original','V41_FULL':'v41_20000','CLIP8_FULL':'clip8'}[method]
                        old=load(V52/'parts'/dataset/oldmethod/f"video_{video['video_index']}_qp{q}.json")
                        assert result['bitstream_sha256']==old['bitstream_sha256'],'Full-method RANS regression mismatch'
                        assert result['reconstruction_sha256']==old['reconstruction_sha256'],'Full-method decode regression mismatch'
                        for metric in ('LPIPS','DISTS','FloLPIPS','SSIM','MS_SSIM'):
                            assert abs(result[metric]-float(old[metric]))<2e-6,(metric,result[metric],old[metric])
                        result['v5a2_regression_pass']=True
                    dump(target,result);print('DONE',dataset,method,video['video_index'],q,result['elapsed_seconds'],flush=True)
            del frames
        if runtime is not None:del runtime;torch.cuda.empty_cache()
    verify_frozen(config)
    dump(root/f'parts/worker_gpu{args.gpu}_done.json',dict(status='PASS',physical_gpu=args.gpu,experiment=args.experiment))
if __name__=='__main__':main()
