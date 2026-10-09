import argparse
import fcntl
import importlib.util
import json
import os
from fractions import Fraction
from pathlib import Path
import torch
from experiment_utils import ROOT,V4,V3,read,dump,sha,write
from metric_runtime import eco,core,load_metrics,measure

def checkpoint(method):
    if method=='original': return None
    if method=='v3': return V3/'checkpoints/stage_b/step_20000.pt'
    if method=='selected': return Path(json.loads((ROOT/'selected_checkpoint.json').read_text())['checkpoint'])
    return ROOT/'checkpoints/beta_high'/f'step_{int(method.split("_")[-1])}.pt'

def cached(split,method,index,qp):
    if method=='selected' and int(json.loads((ROOT/'selected_checkpoint.json').read_text())['step'])==10000:
        method='v4_10000'
    if method not in ('original','v3','v4_10000'): return None
    if split=='validation':
        stage,step={'original':('original',0),'v3':('v3',20000),'v4_10000':('beta_high',10000)}[method]
        rows=read(V4/'checkpoint_validation.csv')
        result=[r for r in rows if r['stage']==stage and int(r['step'])==step and int(r['video_index'])==index and int(r['qp'])==qp]
    else:
        key={'original':'original','v3':'v3_beta_low','v4_10000':'v4_selected'}[method]
        videos=json.loads((ROOT/'test_manifest.json').read_text())['videos']
        video=videos[index]; tag=f"{video['dataset']}_{int(video['video_id']):02d}"
        result=[r for r in read(V4/'final_rd_points.csv') if r['method']==key and r['video_tag']==tag and int(r['qp'])==qp]
    assert len(result)==1
    return result[0]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    parser.add_argument('--split',choices=('validation','final'),required=True)
    parser.add_argument('--method',choices=('original','v3','v4_10000','v4_15000','v4_20000','selected'),required=True)
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    lock=open(ROOT/'parts'/f'eval_{args.split}_{args.method}.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.split=='final':
        sel=json.loads((ROOT/'selected_checkpoint.json').read_text())
        assert sel['used_final_test'] is False and sel['status']=='PASS'
        assert sel['validation_sha256']==sha(ROOT/'checkpoint_validation.csv')
    videos=json.loads((ROOT/('validation_manifest.json' if args.split=='validation' else 'test_manifest.json')).read_text())['videos']
    count=32 if args.split=='validation' else 64
    path=checkpoint(args.method); digest=sha(path) if path else ''
    step=0 if path is None else int(path.stem.split('_')[-1])
    device=torch.device('cuda:0'); joint=eco.load_joint(path,device) if path else None
    models=load_metrics(device); quality=core.quality_models(device)
    if args.split=='final':
        spec=importlib.util.spec_from_file_location('v4_final',V4/'final_evaluate.py')
        final=importlib.util.module_from_spec(spec); spec.loader.exec_module(final)
    for index,video in enumerate(videos):
        point_dir=ROOT/'parts'/args.split/args.method; point_dir.mkdir(parents=True,exist_ok=True)
        pending=[q for q in range(4) if not (point_dir/f'video_{index}_qp{q}.json').exists()]
        if not pending: continue
        if args.split=='validation': frames=eco.video_frames(video['path'],count); video_name=video['filename']
        else:
            video_name=final.tag(video); frames=final.load_frames(video_name)
        fps=float(Fraction(str(video['fps'])))
        for qp in pending:
            row=cached(args.split,args.method,index,qp)
            if row is not None:
                row=dict(row); stream=Path(row['bitstream_path'])
                assert row['decode_status']=='PASS' and str(row['state_sync_pass']).lower()=='true'
                assert row['checkpoint_sha256']==digest
                assert stream.stat().st_size==int(row.get('bytes',row.get('real_bytes')))==int(row['bytes_consumed'])
                assert sha(stream)==row['bitstream_sha256']
                assert int(row['num_frames'])==count
                provenance='reused verified V4 spatial/RANS results; new independent metric decode'
            else:
                stream=ROOT/'bitstreams'/args.split/args.method/f'video_{index}_qp{qp}.bin'
                row,perframe=eco.run_stream(frames,qp,joint,device,fps,stream,quality=quality)
                write(point_dir/f'video_{index}_qp{qp}.frames.csv',perframe)
                provenance='new real RANS encode and independent decode'
            feature=ROOT/'features'/args.split/args.method/f'video_{index}_qp{qp}.npz'
            extra=measure(frames,stream,joint,device,models,feature,row)
            row.update(extra)
            row.update(split=args.split,method=args.method,step=step,video_index=index,video=video_name,qp=qp,num_frames=count,
                       checkpoint=str(path) if path else '',checkpoint_sha256=digest,
                       real_bytes=int(row.get('bytes',row.get('real_bytes'))),provenance=provenance,
                       metric_audit_sha256=sha(ROOT/'metric_implementation_audit.json'))
            dump(point_dir/f'video_{index}_qp{qp}.json',row)
            print(args.split,args.method,index,qp,row['real_bytes'],row['FloLPIPS'],flush=True)

if __name__=='__main__': main()
