"""Fresh real RANS and a new independent decoder; matched RGB FP32-first scoring."""
from common_s2 import *
import argparse,io,traceback
from fractions import Fraction
from model_context import build,backend,assert_reset
METRICS=('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC')
PATTERN=(0,1,0,2,0,2,0,2)

def videos(split):
    if split=='final':return load(ROOT/'manifests/final_videos.json')
    result=[]
    for r in load(ROOT/'vendor/sealed/manifests/evaluation_rows.json'):
        if r['split']!='development':continue
        result.append(dict(domain='ulong_new',sequence=Path(r['video']).stem,split='development',source_path=r['source_path'],source_sha256=r['source_sha256'],
            source_frame_indices=r['source_frame_indices'],height=r['height'],width=r['width'],frame_rgb_sha256=r['full_RGB_sha256'],
            rgb_sha256=r['rgb_sha256'],source_group=r['source_group'],fps=float(Fraction(r['source_fps'])),source_row=r))
    assert len(result)==32 and len({v['sequence'] for v in result})==32
    return result

def arrays_for(v):
    if 'source_row' in v:
        from data_reader import read_arrays
        return read_arrays(v['source_row'],full=True)
    from eval_data import read_frames
    return read_frames(v)

def metrics_setup(rank):
    torch,core,eco,_,_,_=backend(False,rank)
    flo_path=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
    sys.path.insert(0,str(flo_path));import flolpips_compat
    quality=core.quality_models(torch.device('cuda',rank));flow,perceptual=flolpips_compat.load_models(torch.device('cuda',rank))
    hashes=dict(quality=[core.module_hash(m) for m in quality],flow=core.module_hash(flow),FloLPIPS=core.module_hash(perceptual))
    expected=load(ORIGINAL_EVAL/'metric_runtime_rank0.json')['hashes'];assert hashes==expected,('METRIC_IDENTITY_MISMATCH',hashes,expected)
    return quality,flow,perceptual,hashes

def mask_stats(c):
    r=c['router']
    if r is None or r.last_mask is None:return {}
    m=r.last_mask.float();q=c['torch'].quantile(m.flatten(),c['torch'].tensor([.01,.05,.5,.95,.99],device=m.device))
    result=dict(M_mean=float(m.mean()),M_std=float(m.std(unbiased=False)),M_lt_0_1=float((m<.1).float().mean()),M_gt_0_9=float((m>.9).float().mean()))
    result.update({f'M_q{n}':float(a) for n,a in zip(['01','05','50','95','99'],q)})
    for name in ['tcg_low','tcg_mid','tcg_out']:
        stats=getattr(c['models']['generator'],name).last_residual_stats
        result.update({name+'_'+k:v for k,v in stats.items()})
    return result

def point(v,qp,checkpoint,batch,rank=0,scoring=True,reference=False,mode='enabled'):
    import torch
    from src.utils import stream_helper as sh
    from src.layers.cuda_inference import replicate_pad
    root=ROOT/'evaluation'/batch;large=ROOT/'large_artifacts/evaluation'/batch
    for name in ['points','frames','transitions','codec_trace','masks']:(root/name).mkdir(parents=True,exist_ok=True)
    (large/'bitstreams').mkdir(parents=True,exist_ok=True)
    key=f'{v["domain"]}_{v["sequence"]}_qp{qp}';dest=root/f'points/{key}.json'
    cp_sha=sha(checkpoint) if checkpoint else None
    if dest.exists():
        p=load(dest);assert p['status']=='PASS' and p['Router_checkpoint_sha256']==cp_sha and p['mode']==mode
        assert sha(large/p['result']['bitstream_file'])==p['result']['bitstream_sha256'];return p
    arrays=arrays_for(v);h,w=v['height'],v['width'];ch,cw=(h+63)//64*64,(w+63)//64*64
    router_state=None if checkpoint is None else torch.load(checkpoint,map_location='cpu',weights_only=True)['router_state']
    def make():return build(rank,train=False,mode=mode,reference=reference,losses=False,router_state=router_state)
    def gt(i):return (torch.from_numpy(arrays[i]).permute(2,0,1).unsqueeze(0).float()/255).to(rank)
    sps=dict(sps_id=0,height=ch,width=cw,ec_part=1,use_ada_i=0)
    begun=time.time();torch.cuda.reset_peak_memory_stats(rank);sync=[];features=[];actual=[]
    with torch.inference_mode():
        ce=make();cd=make();ie,pe=ce['im'],ce['pm'];idec,pdec=cd['im'],cd['pm'];assert_reset(ce);assert_reset(cd)
        hashes=[tensor_hash(m.state_dict().items()) for m in [ie,pe,idec,pdec]]
        data=io.BytesIO();sh.write_sps(data,sps)
        for i in range(64):
            x=replicate_pad(gt(i).half(),ch-h,cw-w)*2-1
            q=qp if i==0 else pe.shift_qp(qp,PATTERN[i%8]);actual.append(q)
            if i==0:
                enc=ie.compress(x,q);dec=idec.decompress(enc['bit_stream'],sps,q);assert torch.equal(enc['x_hat'],dec['x_hat'])
                assert_reset(ce);assert_reset(cd);pe.add_ref_frame(None,dec['x_hat']);pdec.add_ref_frame(None,dec['x_hat'])
            else:
                enc=pe.compress(x,q);dec=pdec.decompress(enc['bit_stream'],sps,q)
                assert torch.equal(pe.dpb[0].feature,pdec.dpb[0].feature),'CODEC_ENCODER_DECODER_FEATURE_MISMATCH'
            sh.write_ip(data,i==0,0,q,enc['bit_stream']);sync.append(tensor_sha(dec['x_hat']))
            features.append(None if i==0 else tensor_sha(pdec.dpb[0].feature))
        assert hashes==[tensor_hash(m.state_dict().items()) for m in [ie,pe,idec,pdec]]
        raw=data.getvalue();stream=large/f'bitstreams/{key}.bin';temp=stream.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(stream)
        assert_reset(ce);assert_reset(cd);del ce,cd,ie,pe,idec,pdec,enc,dec,x;torch.cuda.empty_cache()
        ci=make();ia,pa=ci['im'],ci['pm'];assert_reset(ci);before=[tensor_hash(m.state_dict().items()) for m in [ia,pa]]
        reader=io.BytesIO(stream.read_bytes());helper=sh.SPSHelper();frames=[];transitions=[];codec=[];masks=[];previous_gt=previous_out=None;recon_hash=[]
        if scoring:quality,flow,perceptual,metric_hash=metrics_setup(rank)
        selected=load(ROOT/'protocol.json')['diagnostics'];diagnostic=v['sequence'] in selected['development_sequences']+selected['final_sequences'] and qp==5
        for i in range(64):
            pos=reader.tell();header=sh.read_header(reader)
            while header['nal_type']==sh.NalType.NAL_SPS:
                helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
            current=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
            assert current==sps and q==actual[i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
            if i==0:
                dec=ia.decompress(payload,current,q);assert_reset(ci);pa.add_ref_frame(None,dec['x_hat'])
            else:dec=pa.decompress(payload,current,q)
            rh=tensor_sha(dec['x_hat']);assert rh==sync[i],('INDEPENDENT_RECONSTRUCTION_MISMATCH',key,i);recon_hash.append(rh)
            feature_hash=None if i==0 else tensor_sha(pa.dpb[0].feature);assert feature_hash==features[i]
            # FP32 BEFORE clamp/range conversion. Exact inherited unit(), native crop.
            out=ci['core'].unit(dec['x_hat'],h,w).float();target=gt(i)
            spatial=ci['eco'].frame_metrics(out,target,quality) if scoring else {}
            frames.append(dict(frame=i,external_qp=qp,actual_qp=q,real_bits=(reader.tell()-pos)*8,payload_bits=len(payload)*8,
                source_frame_RGB_sha256=v['frame_rgb_sha256'][i],reconstruction_tensor_sha256=rh,decoded_RGB_tensor_sha256=tensor_sha(out),**spatial))
            codec.append(dict(frame=i,codec_feature_sha256=feature_hash,payload_sha256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload)))
            if i>0:
                masks.append(dict(frame=i,**mask_stats(ci)))
                if scoring:
                    difference=flow(previous_gt,target)-flow(previous_out,out)
                    magnitude=difference.square().sum(1,keepdim=True).sqrt().sum();assert torch.isfinite(magnitude) and magnitude>0
                    flo=float(perceptual(previous_gt,previous_out,difference,normalize=True).item())
                    backwarp=sys.modules['pwcnet'].backwarp
                    flow_gt=flow(target,previous_gt)
                    wr=float((target-backwarp(previous_out,flow(out,previous_out))).abs().mean())
                    wg=float((target-backwarp(previous_out,flow_gt)).abs().mean())
                    transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=flo,WarpingError=wr,WarpingError_ReconFlow_PWC=wr,WarpingError_GTFlow_PWC=wg,GT_flow_sha256=tensor_sha(flow_gt)))
            if diagnostic and i in selected['P_frames']:
                from PIL import Image
                import numpy as np
                d=ROOT/f'diagnostics/{batch}/{key}';d.mkdir(parents=True,exist_ok=True)
                for label,t in [('GT',target),('Router',out)]:Image.fromarray((t[0].permute(1,2,0).cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)).save(d/f'{i:02d}_{label}.png')
                if ci['router'] is not None and ci['router'].last_mask is not None:
                    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
                    plt.imsave(d/f'{i:02d}_M.png',ci['router'].last_mask[0,0].cpu().numpy(),cmap='magma',vmin=0,vmax=1)
            previous_gt,previous_out=target,out
            dump(root/f'progress_rank{rank}.json',dict(point=key,frame=i,time=time.time()))
        assert reader.tell()==len(raw) and sum(r['real_bits'] for r in frames)==len(raw)*8
        assert before==[tensor_hash(m.state_dict().items()) for m in [ia,pa]]
        result=dict(domain=v['domain'],sequence=v['sequence'],split=v['split'],QP=qp,model_state=batch,frames=64,real_bytes=len(raw),
            bpp=len(raw)*8/(64*h*w),kbps=len(raw)*8*v['fps']/64/1000,bitstream_file=str(stream.relative_to(large)),bitstream_sha256=sha(stream),
            source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'])
        if scoring:
            mse=sum(r['pixel_SSE'] for r in frames)/(64*h*w*3)
            result.update(PSNR=-10*math.log10(max(mse,1e-15)),**{k:sum(r[k] for r in frames)/64 for k in ['MS_SSIM','LPIPS','DISTS']},
                **{k:sum(r[k] for r in transitions)/63 for k in ['FloLPIPS','WarpingError','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']})
            assert all(math.isfinite(float(result[k])) for k in METRICS)
        assert_reset(ci)
        p=dict(status='PASS',result=result,Router_checkpoint_sha256=cp_sha,baseline_sha256=BASELINE_SHA,mode=mode,reference=reference,
            protocol_sha256=sha(ROOT/'protocol.json'),fresh_encode=True,input_stream_reused=False,independent_decode=True,real_RANS=True,
            model_hashes_unchanged=True,TCG_and_Router_reset=True,codec_encoder_sync_independent_features_equal=True,
            reconstruction_sha256=hashlib.sha256(''.join(recon_hash).encode()).hexdigest(),actual_qps=actual,source_frame_indices=v['source_frame_indices'],
            metric_hashes=metric_hash if scoring else None,seconds=time.time()-begun,physical_GPU=rank+4,peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),bytes_consumed=reader.tell())
        for name,records in [('frames',frames),('codec_trace',codec),('masks',masks),('transitions',transitions)]:
            if records:write_csv(root/f'{name}/{key}.csv',records)
        dump(dest,p);del ci,ia,pa
        if scoring:del quality,flow,perceptual
        torch.cuda.empty_cache();disk()
        assert sum(p.stat().st_size for p in (ROOT/'large_artifacts').rglob('*') if p.is_file())<=20*2**30,'LARGE_ARTIFACT_BUDGET_EXCEEDED'
        return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--split',choices=['development','final'],default='development');ap.add_argument('--checkpoint');ap.add_argument('--worker',type=int,required=True);args=ap.parse_args()
    env();verify_sources();backend(False,args.worker)
    with lock(f'evaluation/{args.batch}/worker{args.worker}.lock'):
        completed=0
        for index,v in enumerate(videos(args.split)):
            if index%4!=args.worker:continue
            for q in [0,5,9]:
                p=point(v,q,args.checkpoint,args.batch,args.worker);completed+=1
                event('EVALUATION_POINT_COMPLETE',batch=args.batch,rank=args.worker,sequence=v['sequence'],QP=q,completed=completed)
        dump(ROOT/f'evaluation/{args.batch}/worker{args.worker}_status.json',dict(status='PASS',completed_points=completed))
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
