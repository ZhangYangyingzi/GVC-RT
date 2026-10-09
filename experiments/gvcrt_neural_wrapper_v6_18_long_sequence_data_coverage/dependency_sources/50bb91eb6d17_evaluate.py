import argparse,fcntl,math
from v61_io import *
def read_frames(v):
    import torch,numpy as np,tempfile
    if v['dataset']=='validation':
        sys.path.insert(0,str(V41));import metric_runtime as metrics
        frames=metrics.eco.video_frames(v['path'],32);h=hashlib.sha256()
        for f in frames:h.update(f.contiguous().numpy().tobytes())
        assert h.hexdigest()==v['tensor_sha256'];return frames
    if v['dataset'] in ('ulong','uvg'):
        loader=module('v61_input',V52/'canonical_loader.py');arrays=list(loader.canonical_arrays(v['canonical_dir']))
    else:
        assert '__720p_1280x720_' in Path(v['path']).name and sha(v['path'])==v['source_sha256']
        command=['ffmpeg','-v','error','-threads','1','-i',v['path'],'-map','0:v:0','-vsync','0','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
        with tempfile.TemporaryFile(dir=ROOT/'parts') as err:
            p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err);arrays=[]
            try:
                while True:
                    raw=p.stdout.read(1280*720*3)
                    if not raw:break
                    assert len(raw)==1280*720*3;arrays.append(np.frombuffer(raw,np.uint8).reshape(720,1280,3).copy())
                assert p.wait()==0
            finally:
                p.stdout.close()
                if p.poll() is None:p.terminate();p.wait()
    h=hashlib.sha256()
    for a in arrays:h.update(a.tobytes())
    assert len(arrays)==v['frames'] and h.hexdigest()==v['rgb_sha256']
    return [torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255 for a in arrays]
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('validation','ulong','uvg','virat720'),required=True);p.add_argument('--method',required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--parity-only',action='store_true');args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    sys.path.insert(0,str(OLD_ENGINE));from engine import Runtime,core,eco
    torch.set_num_threads(2);frozen();cfg=load(ROOT/'config.json');device=torch.device('cuda:0')
    lockpath=ROOT/'parts'/f'eval_{args.split}_{args.method}_{args.shard}.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.method=='original':cp=None;cp_hash='';methodpair=[None,None];cps={}
        else:
            if args.method=='selected':
                selection=load(ROOT/'checkpoint_selection.json');assert selection['used_final_test'] is False and selection['status']=='PASS';cp=Path(selection['checkpoint'])
                assert sha(cp)==selection['checkpoint_sha256']
            else:cp=checkpoint(int(args.method.split('_')[-1]))
            cp_hash=sha(cp);payload=torch.load(cp,map_location='cpu',weights_only=True)
            cps={'candidate':dict(path=str(cp),sha256=cp_hash,module_hashes={k:tensor_hash(payload[k]) for k in ('wrapper','bridge','generator')})};methodpair=['candidate','candidate'];del payload
        engine_cfg=dict(experiment='A' if args.split=='virat720' else 'B',methods={args.method:methodpair},checkpoints=cps,force_zero_thres=.12)
        videos=load(ROOT/('validation_sources.json' if args.split=='validation' else 'final_sources.json'))['videos'];videos=[v for v in videos if v['dataset']==args.split]
        qps=cfg['validation_qps'] if args.split=='validation' else list(range(10))
        if args.parity_only:assert args.method=='original' and args.split in ('ulong','uvg');videos=videos[:1];qps=[0,9]
        runtime=None
        for ordinal,v in enumerate(videos):
            if ordinal%args.shards!=args.shard:continue
            pending=[]
            for q in qps:
                out=point(args.split,args.method,v['video_index'],q)
                if out.exists():
                    r=load(out);assert r['checkpoint_sha256']==cp_hash and r['force_zero_thres']==.12
                    for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                else:pending.append(q)
            if not pending:continue
            if runtime is None:runtime=Runtime(engine_cfg,args.method,device)
            frames=read_frames(v)
            diagnostics=None
            if runtime.wrapper is not None:
                with torch.inference_mode():
                    vals=[]
                    for f in frames:
                        x=f.to(device);y=runtime.wrapper(x);dx=(y[:,:,:,1:]-y[:,:,:,:-1])-(x[:,:,:,1:]-x[:,:,:,:-1]);dy=(y[:,:,1:,:]-y[:,:,:-1,:])-(x[:,:,1:,:]-x[:,:,:-1,:])
                        vals.append((float((y-x).abs().mean()),float((y-x).square().mean()),float((dx.abs().mean()+dy.abs().mean())/2)))
                    diagnostics=dict(proxy_RGB_L1=sum(a for a,_,_ in vals)/len(vals),proxy_RGB_MSE=sum(a for _,a,_ in vals)/len(vals),proxy_edge_difference=sum(a for _,_,a in vals)/len(vals),edge_definition='Mean absolute difference of horizontal/vertical RGB first differences')
            for q in pending:
                out=point(args.split,args.method,v['video_index'],q);out.parent.mkdir(parents=True,exist_ok=True)
                with out.with_suffix('.lock').open('a') as point_lock:
                    fcntl.flock(point_lock,fcntl.LOCK_EX)
                    if out.exists():continue
                    start=time.time();relative=Path(args.split)/args.method/f"video_{v['video_index']}_qp{q}"
                    stream=(ROOT/'bitstreams'/relative).with_suffix('.bin');feature=(ROOT/'features'/relative).with_suffix('.npz')
                    result,rows=runtime.run(frames,q,stream,feature)
                    result['kbps']=result['bits_per_frame']*v['rate_fps']/1000;result['rate_accounting_fps']=v['rate_fps']
                    framefile=out.with_suffix('.frames.csv');write(framefile,rows)
                    result.update(dataset=args.split,method=args.method,video_index=v['video_index'],video=v['name'],external_qp=q,QP=q,
                        checkpoint=str(cp) if cp else '',checkpoint_sha256=cp_hash,force_zero_thres=.12,physical_gpu=args.gpu,source_record=v,
                        frame_metrics_path=str(framefile),frame_metrics_sha256=sha(framefile),config_sha256=sha(ROOT/'config.json'),
                        start_unix=start,end_unix=time.time(),elapsed_seconds=time.time()-start,fresh_real_RANS=True)
                    if diagnostics:result.update(diagnostics)
                    dump(out,result);print('DONE',args.split,args.method,v['video_index'],q,result['real_bytes'],flush=True)
            del frames
        if runtime is not None:del runtime;torch.cuda.empty_cache()
    frozen()
if __name__=='__main__':main()
