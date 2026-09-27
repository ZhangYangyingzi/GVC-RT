"""Evaluation-only extension of the unchanged V5-A real RANS path."""
import argparse
import fcntl
import hashlib
import math
import os
import subprocess
import sys
from fractions import Fraction
from audit_utils import *
from stream_audit import structure

def uvg_frames(video,torch,np):
    count=video['frames_evaluated'];size=1920*1080*3
    command=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','yuv420p','-video_size','1920x1080',
             '-framerate',str(video['fps']),'-i',video['path'],'-frames:v',str(count),'-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    # Same ffmpeg raw-YUV -> RGB convention as the corrected V2 UVG source audit.
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=[];digest=hashlib.sha256()
    try:
        for _ in range(count):
            raw=process.stdout.read(size);assert len(raw)==size,'incomplete UVG frame'
            digest.update(raw);a=np.frombuffer(raw,np.uint8).reshape(1080,1920,3).copy()
            frames.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255)
    finally:
        process.stdout.close();error=process.stderr.read();code=process.wait()
        if code:raise RuntimeError(error.decode())
    return frames,digest.hexdigest(),command

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--dataset',choices=('ulong','uvg'),required=True);p.add_argument('--method',choices=METHODS,required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    assert load('qp0_3_backward_compatibility_audit.json')['status']=='PASS','STOP: backward compatibility has not passed'
    lock=open(ROOT/'parts'/f'{args.dataset}_{args.method}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    import torch
    sys.path.insert(0,str(V41))
    import metric_runtime as metrics
    eco,core=metrics.eco,metrics.core
    config=load('config.json');cp=checkpoint(args.method);digest=sha(cp) if cp else ''
    assert digest==config['checkpoints'][args.method]['sha256']
    for path,h in config['source_hashes'].items():assert sha(path)==h,path
    assert sha(ROOT/'uvg_available_manifest.json')==config['uvg_manifest_sha256']
    for path,h in load('metric_implementation_audit.json')['source_and_weight_sha256'].items():assert sha(path)==h,path
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
    final=module('v51_final_source',V4/'final_evaluate.py')
    for index,video in enumerate(videos(args.dataset)):
        count=64 if args.dataset=='ulong' else video['frames_evaluated']
        pending=[q for q in range(10) if not point_path(args.dataset,args.method,index,q).exists()]
        if not pending:continue
        if args.dataset=='ulong':
            frames=final.load_frames(final.tag(video));frame_hash=hashlib.sha256(''.join(eco.tensor_sha(f) for f in frames).encode()).hexdigest()
            conversion='unchanged V5-A frozen RGB PNG sources';command=[]
        else:
            assert sha(video['path'])==video['sha256']
            frames,frame_hash,command=uvg_frames(video,torch,np)
            conversion='unchanged corrected V2 UVG convention: ffmpeg yuv420p -> rgb24; default BT.601 limited -> full RGB'
        assert len(frames)==count and all(tuple(f.shape)==(1,3,1080,1920) for f in frames)
        source_audit=dict(source_hash=frame_hash,hash_representation='concatenated RGB24 raw bytes' if args.dataset=='uvg' else 'concatenated float32 frame tensor hashes',
            conversion=conversion,ffmpeg_command=command,num_frames=count,metric_crop=[1920,1080],processing_canvas=[1920,1088])
        dump(f'parts/source_{args.dataset}_{args.method}_{index}.json',source_audit)
        for q in pending:
            assert not (args.dataset=='ulong' and q<4),'backward compatibility cache must exist'
            print('START',args.dataset,args.method,index,q,'frames',count,flush=True)
            stream=ROOT/'bitstreams'/args.dataset/args.method/f'video_{index}_qp{q}.bin'
            row,frame=eco.run_stream(frames,q,joint,device,float(Fraction(str(video['fps']))),stream,quality=quality)
            feature=ROOT/'features'/args.dataset/args.method/f'video_{index}_qp{q}.npz'
            row.update(metrics.measure(frames,stream,joint,device,metric_models,feature,row))
            header=structure(stream,q,count)
            for a,b in zip(frame,header):
                assert a['actual_qp']==b['actual_qp'];a.update(b,dataset='Fresh_U_Long' if args.dataset=='ulong' else 'UVG',method=args.method,video_index=index)
            path=point_path(args.dataset,args.method,index,q);framepath=path.with_suffix('.frames.csv');write(framepath,frame)
            row.update(dataset='Fresh_U_Long' if args.dataset=='ulong' else 'UVG',method=args.method,video_index=index,
                sequence_name=video.get('sequence_name',video.get('name')),external_qp=q,qp=q,actual_i_qp=q,
                actual_p_qp=sorted({r['actual_qp'] for r in header[1:]}),num_frames=count,num_transitions=count-1,
                checkpoint=str(cp) if cp else '',checkpoint_sha256=digest,real_bytes=row['bytes'],
                metric_audit_sha256=config['metric_audit_sha256'],manifest_sha256=config['ulong_manifest_sha256' if args.dataset=='ulong' else 'uvg_manifest_sha256'],
                frame_metrics_path=str(framepath),frame_metrics_sha256=sha(framepath),position_decode_pass=True,
                evaluation_only=True,physical_gpu=args.gpu,source_frame_hash=frame_hash,
                provenance='unchanged V4 eval_core.run_stream and V4.1 metric_runtime.measure; fresh full RANS and independent decoders')
            assert all(math.isfinite(float(row[k])) for k in NUMERIC)
            dump(path,row)
            print('DONE',args.dataset,args.method,index,q,row['real_bytes'],row['FloLPIPS'],flush=True)
            torch.cuda.empty_cache()
        del frames
    assert (sha(cp) if cp else '')==digest
    dump(f'parts/{args.dataset}_{args.method}_done.json',dict(status='PASS',evaluation_only=True,checkpoint_unchanged=True,physical_gpu=args.gpu))
    print('EVALUATION PASS',args.dataset,args.method,flush=True)

if __name__=='__main__':main()
