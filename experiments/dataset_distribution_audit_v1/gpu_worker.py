"""One heavy worker per GPU; cached PWC/ResNet motion phase or Original codec phase."""
import argparse
import fcntl
import math
import traceback
from audit_io import *
from frames import view_frames,native_frames

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--phase',choices=('motion','codec'),required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    import numpy as np
    import cv2
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(SEED)
    lock=(ROOT/'parts'/f'gpu_{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=load(ROOT/('motion_shards.json' if a.phase=='motion' else 'gpu_sharding_audit.json'))
    tasks=plan['shards'][str(a.gpu)];tasks=tasks[:a.limit] if a.limit else tasks
    videos={v['sample_id']:v for v in load(ROOT/'manifests/all_sources.json')['videos']}
    device=torch.device('cuda:0');assert torch.cuda.device_count()==1
    if a.phase=='motion':
        compat=load(ROOT/'motion_method_audit.json');flowmod=module('distribution_flo',Path(compat['source']));flow,unused=flowmod.load_models(device);del unused
        embedding_audit=load(ROOT/'visual_embedding_audit.json');embed=None
        if embedding_audit['status']=='AVAILABLE':
            from torchvision.models import resnet50
            embed=resnet50(weights=None);embed.load_state_dict(torch.load(embedding_audit['weights'],map_location='cpu',weights_only=True));embed.fc=torch.nn.Identity();embed=embed.to(device).eval().requires_grad_(False)
        for ordinal,task in enumerate(tasks,1):
            sid=task['sample_id'];v=videos[sid];out=ROOT/'parts/motion'/f'{sid}.json'
            if out.exists():assert load(out)['status']=='PASS';continue
            views=[];embedding=None
            with torch.inference_mode():
                for view in ('standardized_content_view','actual_codec_input_view'):
                    frames,meta=view_frames(v,view);values=[];prev=None
                    for frame in frames:
                        x=torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(device).float()/255
                        if prev is not None:
                            field=flow(prev,x);mag=field.square().sum(1).sqrt();diag=math.hypot(meta['width'],meta['height'])
                            values.append(dict(flow_magnitude_mean=float(mag.mean()),flow_magnitude_p90=float(torch.quantile(mag.flatten(),.9)),
                                flow_magnitude_p95=float(torch.quantile(mag.flatten(),.95)),flow_magnitude_std=float(mag.std(correction=0)),
                                flow_spatial_variance=float(field.var(dim=(2,3),correction=0).sum()),flow_magnitude_normalized=float(mag.mean())/diag,
                                flow_variance_normalized=float(field.var(dim=(2,3),correction=0).sum())/(diag*diag)))
                        prev=x
                    row=dict(dataset=v['dataset'],sample_id=sid,view=view,**meta,**{k:float(np.mean([r[k] for r in values])) for k in values[0]})
                    assert all(math.isfinite(row[k]) for k in values[0]);views.append(row)
                    if view=='standardized_content_view' and embed is not None:
                        inputs=[]
                        for index in sorted(set((0,len(frames)//2,len(frames)-1))):
                            x=torch.from_numpy(frames[index]).permute(2,0,1).unsqueeze(0).to(device).float()/255
                            x=torch.nn.functional.interpolate(x,(224,224),mode='bilinear',align_corners=False)
                            inputs.append((x-torch.tensor([.485,.456,.406],device=device).view(1,3,1,1))/torch.tensor([.229,.224,.225],device=device).view(1,3,1,1))
                        embedding=embed(torch.cat(inputs)).mean(0).cpu().numpy();assert np.isfinite(embedding).all()
                        ep=ROOT/'parts/embeddings'/f'{sid}.npy';ep.parent.mkdir(parents=True,exist_ok=True)
                        with ep.with_suffix('.tmp').open('wb') as f:np.save(f,embedding)
                        ep.with_suffix('.tmp').replace(ep)
            dump(out,dict(status='PASS',sample_id=sid,views=views,embedding_present=embedding is not None,gpu=a.gpu))
            if ordinal%10==0:print('MOTION',a.gpu,ordinal,'/',len(tasks),flush=True)
    else:
        engine=ROOT.parent/'gvcrt_parallel_v5a3_v60';sys.path.insert(0,str(engine));from engine import Runtime
        runtime=Runtime(dict(experiment='B',methods={'original':[None,None]},checkpoints={},force_zero_thres=.12),'original',device)
        loaded_sid=None;frames=None
        qpaudit=load(ROOT/'codec_protocol_audit.json')
        for ordinal,task in enumerate(tasks,1):
            sid,q=task['sample_id'],task['QP'];v=videos[sid];out=ROOT/'parts/codec'/sid/f'qp{q}.json'
            if out.exists():assert load(out)['status']=='PASS';continue
            if sid!=loaded_sid:
                raw,indices,fps=native_frames(v,False);frames=[torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()/255 for x in raw]
                rgb_hash=hashlib.sha256(b''.join(x.tobytes() for x in raw)).hexdigest();loaded_sid=sid
            n=len(frames);expected=7 if v['kind']=='vimeo' else 32;assert n==expected
            t=time.time();bit=ROOT/'bitstreams'/sid/f'qp{q}.bin';feature=ROOT/'features'/sid/f'qp{q}.npz'
            r,rows=runtime.run(frames,q,bit,feature)
            assert [x['actual_qp'] for x in rows]==qpaudit['actual_qps'][str(q)][:n]
            framefile=out.with_suffix('.frames.csv');write(framefile,rows)
            r.update(status='PASS',dataset=v['dataset'],sample_id=sid,external_qp=q,QP=q,method='Original GVC-RT',force_zero_thres=.12,
                total_bits=r['real_bytes']*8,comparison_fps=30.0,source_effective_fps=v['fps'],fps_assumed_for_kbps=v['kind']=='vimeo',
                width=v['width'],height=v['height'],source_rgb_sha256=rgb_hash,frame_metrics_path=str(framefile),frame_metrics_sha256=sha(framefile),
                reused=False,original_only=True,gpu=a.gpu,elapsed_seconds=time.time()-t)
            assert r['real_RANS'] and r['independent_decode_pass'] and r['real_bytes']==bit.stat().st_size
            assert all(math.isfinite(r[k]) for k in ('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM','bpp'))
            dump(out,r);print('CODEC',a.gpu,ordinal,'/',len(tasks),sid,q,flush=True)
    dump(ROOT/'parts'/f'{a.phase}_gpu{a.gpu}{"_smoke" if a.limit else ""}_done.json',dict(status='PASS',tasks=len(tasks),gpu=a.gpu,phase=a.phase,updated_unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu');p.add_argument('--phase');a,_=p.parse_known_args()
        dump(ROOT/'logs'/f'{a.phase}_gpu{a.gpu}_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
