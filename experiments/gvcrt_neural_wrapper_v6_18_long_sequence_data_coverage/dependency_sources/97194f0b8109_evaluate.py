"""Fresh real RANS evaluation of the frozen methods on canonical RGB PNGs."""
import argparse
import fcntl
import hashlib
import math
import os
import sys
import time
from audit_utils import *
from canonical_loader import load_canonical_frames, rgb_hash
from stream_audit import structure

def valid_point(path):
    if not path.exists(): return False
    r=load(path)
    assert r['protocol']=='V5-A.2-30fps-RGB' and r['fps']==30.0 and r['num_frames']==64,path
    assert r['fresh_real_RANS'] and r['independent_decode_pass'] and r['metric_decode_pass'],path
    assert r['canonical_audit_sha256']==sha(ROOT/'canonical_source_audit.json'),path
    for key in ('bitstream','feature','frame_metrics'):
        assert sha(r[key+'_path'])==r[key+'_sha256'],path
    assert r['real_bytes']==Path(r['bitstream_path']).stat().st_size==r['bytes_consumed']
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--dataset',choices=('ulong','uvg'),required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--video-index',type=int)
    p.add_argument('--qps',type=int,nargs='+',default=list(range(10)))
    args=p.parse_args();assert set(args.qps)<=set(range(10))
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    assert load('canonical_source_audit.json')['status']=='PASS'
    loader=load('canonical_loader_audit.json')
    assert loader['status']=='PASS' and sha(loader['loader_path'])==loader['loader_sha256']
    sys.path.insert(0,str(V41))
    import metric_runtime as metrics
    eco,core=metrics.eco,metrics.core
    config=load('config.json');cp=checkpoint(args.method);digest=sha(cp) if cp else ''
    assert digest==config['checkpoints'][args.method]['sha256']
    for mapping in (config['source_hashes'],config['additional_source_hashes'],load('metric_implementation_audit.json')['source_and_weight_sha256']):
        for path,h in mapping.items():assert sha(path)==h,path
    device=torch.device('cuda:0');assert torch.cuda.is_available() and torch.cuda.device_count()==1
    joint=eco.load_joint(cp,device) if cp else None
    i_model,p_model=eco.build_models(device,None if joint is None else joint[0])
    lengths={key:int(getattr(p_model,key).shape[0]) for key in ('q_scale_enc','q_scale_dec','q_scale_feature','q_scale_recon')}
    assert i_model.get_qp_num()==p_model.get_qp_num()==10 and all(v==12 for v in lengths.values())
    assert all(not p.requires_grad for p in i_model.parameters()) and all(not p.requires_grad for p in p_model.parameters())
    dump(f'parts/model_audit_{args.dataset}_{args.method}.json',dict(status='PASS',external_qp_num=10,internal_qp_tensor_length=lengths,
        checkpoint_sha256=digest,physical_gpu=args.gpu,evaluation_only=True,compression_frozen=True))
    del i_model,p_model;torch.cuda.empty_cache()
    quality=core.quality_models(device);metric_models=metrics.load_metrics(device)
    audit_hash=sha(ROOT/'canonical_source_audit.json')
    for index,video in enumerate(videos(args.dataset)):
        if args.video_index is not None and index!=args.video_index:continue
        pending=[q for q in args.qps if not valid_point(point_path(args.dataset,args.method,index,q))]
        if not pending:continue
        name=video.get('name',video.get('sequence_name'))
        directory=ROOT/'canonical_sources'/args.dataset/name
        source=load(f'parts/canonical_{args.dataset}_{index}.json')
        assert rgb_hash(directory)==source['whole_sequence_rgb_hash']
        frames=load_canonical_frames(directory)
        tensor_hash=hashlib.sha256(''.join(eco.tensor_sha(f) for f in frames).encode()).hexdigest()
        dump(f'parts/source_{args.dataset}_{args.method}_{index}.json',dict(source_hash=source['whole_sequence_rgb_hash'],tensor_hash=tensor_hash,
            canonical_dir=str(directory),loader_sha256=loader['loader_sha256'],num_frames=64,fps=30.0,metric_crop=[1920,1080],processing_canvas=[1920,1088]))
        for q in pending:
            path=point_path(args.dataset,args.method,index,q);path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if valid_point(path):continue
                started=time.time()
                print('START',args.dataset,args.method,index,q,'frames',64,flush=True)
                stream=ROOT/'bitstreams'/args.dataset/args.method/f'video_{index}_qp{q}.bin'
                row,frame=eco.run_stream(frames,q,joint,device,30.0,stream,quality=quality)
                feature=ROOT/'features'/args.dataset/args.method/f'video_{index}_qp{q}.npz'
                row.update(metrics.measure(frames,stream,joint,device,metric_models,feature,row))
                header=structure(stream,q,64)
                for a,b in zip(frame,header):
                    assert a['actual_qp']==b['actual_qp']
                    a.update(b,dataset=args.dataset,method=args.method,video_index=index)
                framepath=path.with_suffix('.frames.csv');write(framepath,frame)
                row.update(dataset=args.dataset,method=args.method,video_index=index,sequence_name=name,
                    external_qp=q,qp=q,actual_i_qp=q,actual_p_qp=sorted({r['actual_qp'] for r in header[1:]}),
                    num_frames=64,num_transitions=63,fps=30.0,duration_seconds=64/30.0,
                    checkpoint=str(cp) if cp else '',checkpoint_sha256=digest,real_bytes=row['bytes'],bits_per_frame=row['bytes']*8/64,
                    metric_audit_sha256=config['metric_audit_sha256'],manifest_sha256=config['ulong_manifest_sha256' if args.dataset=='ulong' else 'uvg_manifest_sha256'],
                    frame_metrics_path=str(framepath),frame_metrics_sha256=sha(framepath),position_decode_pass=True,
                    canonical_audit_sha256=audit_hash,canonical_loader_sha256=loader['loader_sha256'],canonical_dir=str(directory),
                    evaluation_only=True,physical_gpu=args.gpu,source_frame_hash=source['whole_sequence_rgb_hash'],tensor_hash=tensor_hash,
                    protocol='V5-A.2-30fps-RGB',fresh_real_RANS=True,elapsed_seconds=time.time()-started,
                    provenance='unchanged eval_core.run_stream and metric_runtime.measure; fresh RANS, independent decode, canonical PNG loader')
                assert math.isclose(row['kbps'],row['bits_per_frame']*30/1000,rel_tol=1e-12)
                assert all(math.isfinite(float(row[k])) for k in NUMERIC)
                assert row['independent_decode_pass'] and row['metric_decode_pass'] and row['state_sync_pass']
                dump(path,row)
                print('DONE',args.dataset,args.method,index,q,'bytes',row['real_bytes'],'seconds',round(row['elapsed_seconds'],2),flush=True)
                torch.cuda.empty_cache()
        del frames
    assert (sha(cp) if cp else '')==digest
    if all(valid_point(point_path(args.dataset,args.method,i,q)) for i in range(len(videos(args.dataset))) for q in range(10)):
        dump(f'parts/{args.dataset}_{args.method}_done.json',dict(status='PASS',evaluation_only=True,checkpoint_unchanged=True,physical_gpu=args.gpu))
    print('EVALUATION COMMAND PASS',args.dataset,args.method,flush=True)

if __name__=='__main__':main()

