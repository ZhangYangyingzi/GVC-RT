"""Fresh original-code RANS encoding, then fresh independent decoding and scoring."""
from common import *
import argparse
import fcntl
import io
import math
import sys
import time
from data import read_frames
from runtime import initialize,load_models,load_quality,tensor_sha,model_sha,verify_codec_imports


def key_for(video,qp):
    return f'GVC_RT_original_{video["domain"]}_{video["sequence"]}_qp{qp}'


def run_point(rt,quality,flow,perceptual,arrays,v,qp):
    torch=rt['torch'];stream_helpers=rt['stream'];device=rt['device']
    key=key_for(v,qp);begun=time.time();n=64;h,w=v['height'],v['width'];ch,cw=(h+63)//64*64,(w+63)//64*64
    protocol_sha=sha(ROOT/'protocol.json')
    sps=dict(sps_id=0,height=ch,width=cw,ec_part=1,use_ada_i=0)
    def gt_frame(i):
        # Match the prior readers: normalize in FP32 on CPU, then transfer.
        return (torch.from_numpy(arrays[i]).permute(2,0,1).unsqueeze(0).float()/255).to(device)
    def reset(pm):
        pm.clear_dpb();pm.set_curr_poc(0)
        assert not pm.dpb
    def progress(stage,frame):
        dump(ROOT/f'progress/gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.json',dict(point=key,stage=stage,frame=frame,time=time.time()))
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        ie,pe=load_models(rt);idec,pdec=load_models(rt)
        before=[model_sha(m) for m in (ie,pe,idec,pdec)]
        reset(pe);reset(pdec)
        buffer=io.BytesIO();stream_helpers.write_sps(buffer,sps)
        sync_hashes=[];actual=[]
        for i in range(n):
            x=rt['replicate_pad'](gt_frame(i).half(),ch-h,cw-w)*2.-1.
            q=qp if i==0 else pe.shift_qp(qp,QP_PATTERN[i%8]);actual.append(q)
            if i==0:
                encoded=ie.compress(x,q);decoded=idec.decompress(encoded['bit_stream'],sps,q)
                assert torch.equal(encoded['x_hat'],decoded['x_hat'])
                reset(pe);reset(pdec)
                pe.add_ref_frame(None,decoded['x_hat']);pdec.add_ref_frame(None,decoded['x_hat'])
            else:
                encoded=pe.compress(x,q);decoded=pdec.decompress(encoded['bit_stream'],sps,q)
                assert torch.equal(pe.dpb[0].feature,pdec.dpb[0].feature)
            stream_helpers.write_ip(buffer,i==0,0,q,encoded['bit_stream'])
            sync_hashes.append(tensor_sha(decoded['x_hat']))
            if i%16==0:progress('fresh_encode',i)
        assert before==[model_sha(m) for m in (ie,pe,idec,pdec)]
        data=buffer.getvalue();path=ROOT/f'bitstreams/{key}.bin';path.parent.mkdir(exist_ok=True)
        temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
        del ie,pe,idec,pdec,encoded,decoded,x
        torch.cuda.empty_cache()
        # New decoder instances read the persisted stream, never an older experiment stream.
        ia,pa=load_models(rt);reset(pa);decoder_before=[model_sha(m) for m in (ia,pa)]
        reader=io.BytesIO(path.read_bytes());helper=stream_helpers.SPSHelper()
        rows=[];transitions=[];recon_hashes=[];previous_gt=previous_out=None
        for i in range(n):
            pos=reader.tell();header=stream_helpers.read_header(reader)
            while header['nal_type']==stream_helpers.NalType.NAL_SPS:
                helper.add_sps_by_id(stream_helpers.read_sps_remaining(reader,header['sps_id']))
                header=stream_helpers.read_header(reader)
            current=helper.get_sps_by_id(header['sps_id']);q,payload=stream_helpers.read_ip_remaining(reader)
            assert current==sps and q==actual[i]
            assert (header['nal_type']==stream_helpers.NalType.NAL_I)==(i==0)
            if i==0:
                decoded=ia.decompress(payload,current,q);reset(pa);pa.add_ref_frame(None,decoded['x_hat'])
            else:decoded=pa.decompress(payload,current,q)
            rec_hash=tensor_sha(decoded['x_hat']);recon_hashes.append(rec_hash)
            assert rec_hash==sync_hashes[i],('INDEPENDENT_DECODE_MISMATCH',key,i)
            # Preserve the established half-precision range conversion before FP32 scoring.
            output=((decoded['x_hat'].clamp(-1,1)+1)/2)[:,:,:h,:w].float();gt=gt_frame(i)
            spatial=rt['frame_metrics'](output,gt,quality)
            assert all(math.isfinite(float(x)) for x in spatial.values())
            rows.append(dict(frame=i,external_qp=qp,actual_qp=q,real_bits=(reader.tell()-pos)*8,payload_bits=len(payload)*8,
                             source_frame_RGB_sha256=v['frame_rgb_sha256'][i],reconstruction_tensor_sha256=rec_hash,
                             decoded_RGB_tensor_sha256=tensor_sha(output),**spatial))
            if previous_gt is not None:
                difference=flow(previous_gt,gt)-flow(previous_out,output)
                magnitude=difference.square().sum(1,keepdim=True).sqrt().sum()
                assert torch.isfinite(magnitude) and magnitude>0
                flo=float(perceptual(previous_gt,previous_out,difference,normalize=True).item())
                backwarp=sys.modules['pwcnet'].backwarp
                rec_warp=backwarp(previous_out,flow(output,previous_out))
                rec_error=float((gt-rec_warp).abs().mean())
                gt_flow=flow(gt,previous_gt)
                gt_error=float((gt-backwarp(previous_out,gt_flow)).abs().mean())
                transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=flo,WarpingError_ReconFlow_PWC=rec_error,
                                        WarpingError_GTFlow_PWC=gt_error,GT_flow_sha256=tensor_sha(gt_flow)))
                assert all(math.isfinite(x) for x in (flo,rec_error,gt_error))
            previous_gt,previous_out=gt,output
            if i%16==0:progress('independent_decode_metrics',i)
        assert reader.tell()==len(data) and sum(r['real_bits'] for r in rows)==len(data)*8
        assert len(rows)==64 and len(transitions)==63
        assert decoder_before==[model_sha(m) for m in (ia,pa)]
        mse=sum(r['pixel_SSE'] for r in rows)/(n*h*w*3)
        result=dict(domain=v['domain'],sequence=v['sequence'],QP=qp,model_state='GVC_RT_original',split=v['split'],frames=n,
                    real_bytes=len(data),bpp=len(data)*8/(n*h*w),kbps=len(data)*8*v['fps']/n/1000,PSNR=-10*math.log10(max(mse,1e-15)),
                    **{k:sum(r[k] for r in rows)/n for k in ('MS_SSIM','LPIPS','DISTS')},
                    **{k:sum(r[k] for r in transitions)/63 for k in ('FloLPIPS','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')},
                    bitstream_file=path.name,bitstream_sha256=sha(path),source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],
                    original_commit=COMMIT,protocol_sha256=protocol_sha)
        assert all(math.isfinite(result[k]) for k in METRICS)
        write_csv(ROOT/f'frames/{key}.csv',rows);write_csv(ROOT/f'transitions/{key}.csv',transitions)
        point=dict(status='PASS',result=result,fresh_encode=True,input_stream_reused=False,independent_decode=True,real_RANS=True,
                   original_codec_modules_only=True,TCG_modules_present=False,adapted_checkpoint_loaded=False,
                   actual_qps=actual,source_frame_indices=v['source_frame_indices'],source_rgb_sha256=v['rgb_sha256'],
                   bytes_consumed=reader.tell(),model_hashes_unchanged=True,decoder_module_hashes=decoder_before,
                   reconstruction_sha256=hashlib.sha256(''.join(recon_hashes).encode()).hexdigest(),
                   source_imports=verify_codec_imports(),protocol_sha256=protocol_sha,
                   seconds=time.time()-begun,physical_GPU=int(os.environ['CUDA_VISIBLE_DEVICES']),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        dump(ROOT/f'points/{key}.json',point)
        del ia,pa
        torch.cuda.empty_cache()
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--limit-videos',type=int)
    args=ap.parse_args();assert args.rank in range(4)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(4+args.rank)
    verify_protocol()
    with (ROOT/f'worker{args.rank}.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt=initialize('cuda:0');rt['torch'].cuda.set_device(0)
        quality,flow,perceptual=load_quality(rt)
        metric_hashes=dict(quality=[model_sha(m) for m in quality],flow=model_sha(flow),FloLPIPS=model_sha(perceptual))
        dump(ROOT/f'metric_runtime_rank{args.rank}.json',dict(status='PASS',hashes=metric_hashes,custom_cuda_kernels=rt['custom_cuda_kernels']))
        selected=[v for i,v in enumerate(load(ROOT/'videos.json')) if i%4==args.rank]
        if args.limit_videos is not None:selected=selected[:args.limit_videos]
        completed=0
        for v in selected:
            missing=[q for q in (0,5,9) if not (ROOT/f'points/{key_for(v,q)}.json').exists()]
            arrays=read_frames(v) if missing else None
            for qp in (0,5,9):
                destination=ROOT/f'points/{key_for(v,qp)}.json'
                if destination.exists():
                    point=load(destination)
                    assert point['status']=='PASS' and point['protocol_sha256']==sha(ROOT/'protocol.json')
                    assert sha(ROOT/'bitstreams'/point['result']['bitstream_file'])==point['result']['bitstream_sha256']
                else:run_point(rt,quality,flow,perceptual,arrays,v,qp)
                completed+=1
                print(json.dumps(dict(event='POINT_COMPLETE',sequence=v['sequence'],QP=qp,completed=completed,rank=args.rank)),flush=True)
            del arrays
        assert metric_hashes==dict(quality=[model_sha(m) for m in quality],flow=model_sha(flow),FloLPIPS=model_sha(perceptual))
        dump(ROOT/f'worker{args.rank}_status.json',dict(status='PASS',completed_points=completed,limited=args.limit_videos is not None))


if __name__=='__main__':
    main()
