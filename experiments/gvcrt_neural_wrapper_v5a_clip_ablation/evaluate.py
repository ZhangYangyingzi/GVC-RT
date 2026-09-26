import argparse
import fcntl
import io
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
import torch
from v5_utils import ROOT,V41,V4,METHODS,checkpoint,sha,load,dump,read,write,module,truth
sys.path.insert(0,str(V41))
import metric_runtime as metrics
eco,core=metrics.eco,metrics.core

def structure(path):
    data=Path(path).read_bytes();buffer=io.BytesIO(data);helper=eco.SPSHelper();rows=[]
    while buffer.tell()<len(data):
        start=buffer.tell();header=eco.read_header(buffer)
        while header['nal_type']==eco.NalType.NAL_SPS:
            helper.add_sps_by_id(eco.read_sps_remaining(buffer,header['sps_id']));header=eco.read_header(buffer)
        sps=helper.get_sps_by_id(header['sps_id']);qp,bits=eco.read_ip_remaining(buffer)
        index=len(rows);is_i=header['nal_type']==eco.NalType.NAL_I
        assert is_i==(index==0),'unexpected sequence reset'
        rows.append(dict(frame=index,actual_qp=qp,is_I=is_i,real_bits=(buffer.tell()-start)*8,
                         payload_bits=len(bits)*8,sps=sps,bit_stream=bits))
    assert sum(r['real_bits'] for r in rows)==len(data)*8
    return rows

@torch.inference_mode()
def decode_positions(frames,stream,joint,device,quality,expected):
    i_model,p_model=eco.build_models(device,None if joint is None else joint[0])
    p_model.clear_dpb();p_model.set_curr_poc(0);records=structure(stream);hashes=[];rows=[]
    assert len(records)==len(frames)
    for record,reference in zip(records,frames):
        if record['is_I']:
            out=i_model.decompress(record['bit_stream'],record['sps'],record['actual_qp'])
            p_model.clear_dpb();p_model.add_ref_frame(None,out['x_hat'])
        else:out=p_model.decompress(record['bit_stream'],record['sps'],record['actual_qp'])
        hashes.append(eco.tensor_sha(out['x_hat']))
        value=core.unit(out['x_hat'],1080,1920);measurements=eco.frame_metrics(value,reference.to(device),quality)
        assert all(math.isfinite(v) for v in measurements.values())
        rows.append({**{k:v for k,v in record.items() if k not in ('sps','bit_stream')},**measurements})
    assert eco.sha256_bytes(''.join(hashes).encode())==expected['reconstruction_sha256']
    assert core.compression_hash(i_model,p_model)==expected['compression_hash_after']
    for m in ('LPIPS','DISTS','SSIM','MS_SSIM'):
        assert abs(sum(r[m] for r in rows)/len(rows)-float(expected[m]))<2e-6,(m,'cached spatial metric mismatch')
    return rows

def reused(split,method,index,qp):
    if method not in ('original','v41_20000'):return None,None
    old='original' if method=='original' else ('v4_20000' if split=='validation' else 'selected')
    path=V41/'parts'/split/old/f'video_{index}_qp{qp}.json'
    try:
        row=load(path);cp=checkpoint(method)
        manifest='validation_manifest.json' if split=='validation' else 'test_manifest.json'
        assert sha(ROOT/manifest)==sha(V41/manifest)
        assert sha(ROOT/'metric_implementation_audit.json')==sha(V41/'metric_implementation_audit.json')==row['metric_audit_sha256']
        assert sha(V41/'metric_runtime.py')==load('metric_protocol_reuse_audit.json')['source_metric_runtime_sha256']
        assert row['checkpoint_sha256']==(sha(cp) if cp else '')
        assert sha(row['bitstream_path'])==row['bitstream_sha256']
        assert Path(row['bitstream_path']).stat().st_size==int(row['real_bytes'])==int(row['bytes_consumed'])
        assert sha(row['feature_path'])==row['feature_sha256']
        assert truth(row['independent_decode_pass']) and truth(row['metric_decode_pass']) and row['decode_status']=='PASS'
        assert int(row['num_frames'])==(32 if split=='validation' else 64)
        assert len(read(Path(row['feature_path']).with_suffix('.transitions.csv')))==int(row['num_frames'])-1
        return row,None
    except (OSError,AssertionError,KeyError,ValueError) as exc:return None,repr(exc)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--split',choices=('validation','final'),required=True);p.add_argument('--method',choices=METHODS,required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    lock=open(ROOT/'parts'/f'eval_{args.split}_{args.method}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.split=='final':
        ready=load('parts/validation_done.json');assert ready['status']=='PASS'
        assert ready['validation_sha256']==sha(ROOT/'checkpoint_validation.csv')
    videos=load('validation_manifest.json' if args.split=='validation' else 'test_manifest.json')['videos']
    count=32 if args.split=='validation' else 64
    device=torch.device('cuda:0');cp=checkpoint(args.method);digest=sha(cp) if cp else ''
    joint=eco.load_joint(cp,device) if cp else None;quality=core.quality_models(device);metric_models=None
    final=module('v5_final_source',V4/'final_evaluate.py') if args.split=='final' else None
    for index,video in enumerate(videos):
        outdir=ROOT/'parts'/args.split/args.method;outdir.mkdir(parents=True,exist_ok=True)
        pending=[q for q in range(4) if not (outdir/f'video_{index}_qp{q}.json').exists()]
        if not pending:continue
        frames=eco.video_frames(video['path'],count) if args.split=='validation' else final.load_frames(final.tag(video))
        fps=float(Fraction(str(video['fps'])))
        for qp in pending:
            row,reason=reused(args.split,args.method,index,qp)
            if row is not None:
                frame_rows=decode_positions(frames,row['bitstream_path'],joint,device,quality,row)
                provenance='reuse V4.1 verified manifest/checkpoint/protocol/stream/features; new position decode'
            else:
                stream=ROOT/'bitstreams'/args.split/args.method/f'video_{index}_qp{qp}.bin'
                row,frame_rows=eco.run_stream(frames,qp,joint,device,fps,stream,quality=quality)
                if metric_models is None:metric_models=metrics.load_metrics(device)
                feature=ROOT/'features'/args.split/args.method/f'video_{index}_qp{qp}.npz'
                row.update(metrics.measure(frames,stream,joint,device,metric_models,feature,row))
                records=structure(stream);assert len(records)==count
                for r,s in zip(frame_rows,records):r.update({k:s[k] for k in ('is_I','real_bits','payload_bits')})
                provenance='new full RANS encode, independent decode and unchanged V4.1 metric pipeline'
            row.update(method=args.method,split=args.split,video_index=index,qp=qp,num_frames=count,
                checkpoint=str(cp) if cp else '',checkpoint_sha256=digest,step=0 if cp is None else int(cp.stem.split('_')[-1]),
                real_bytes=int(row.get('bytes',row.get('real_bytes'))),provenance=provenance,reuse_failure_reason=reason or '',
                metric_audit_sha256=sha(ROOT/'metric_implementation_audit.json'),
                manifest_sha256=sha(ROOT/('validation_manifest.json' if args.split=='validation' else 'test_manifest.json')))
            frame_path=outdir/f'video_{index}_qp{qp}.frames.csv';write(frame_path,frame_rows)
            assert sum(int(r['real_bits']) for r in frame_rows)==row['real_bytes']*8
            row.update(frame_metrics_path=str(frame_path),frame_metrics_sha256=sha(frame_path),position_decode_pass=True)
            dump(outdir/f'video_{index}_qp{qp}.json',row)
            print(args.split,args.method,index,qp,row['real_bytes'],row['FloLPIPS'],flush=True)
    print('EVALUATION PASS',args.split,args.method,flush=True)

if __name__=='__main__':main()
