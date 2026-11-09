"""Real shared streams, fresh independent candidate decoder, native metrics."""
from common_s2 import *
from experiment_model import construct,restore
from model_context import assert_reset,component_hashes
from cache_codec import cache_paths
from temporal import temporal_errors,sobel,VERSION
import io,argparse

SPATIAL=('LPIPS','DISTS','MS_SSIM')
TEMPORAL=('FloLPIPS','MCResidual_RGB_RAFT_pixelcenter_v1','MCResidual_Sobel_RAFT_pixelcenter_v1','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC')

def metrics_setup(rank):
    from evaluate import metrics_setup
    return metrics_setup(rank)

def point(profile,qp,group,step,rank,training_probe=False,export=False,engineering_output=False):
    import torch
    row=profile['row'] if 'row' in profile else profile
    if export:
        row=dict(row);row.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)))
    key,path,cache_receipt=cache_paths(row,qp,training_probe)
    proof=load(cache_receipt);assert proof['status']=='PASS' and sha(path)==proof['cache_sha256']
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);info=data['info'];n=info['frames'];h,w=info['native_shape'];seq=Path(row['video']).stem
    root=ROOT/('long_evaluation' if export else 'engineering_evaluation' if training_probe or engineering_output else 'evaluation')/group/f'step_{step:04d}'
    cp=None if group in ['Original','A2000'] or step==0 else ROOT/group/f'training/checkpoints/step_{step:04d}.pt'
    checkpoint_hash=INIT_SHA if cp is None and group!='Original' else None if cp is None else sha(cp)
    identity=dict(group=group,step=step,checkpoint_SHA256=checkpoint_hash,initial_A2000_SHA256=INIT_SHA,source_cache_identity=key,external_QP=qp,s=.5,
        source_row_SHA256=digest(row),protocol_SHA256=sha(ROOT/'protocol.json'),implementation_SHA256=sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None)
    dest=root/'points'/f'{seq}_qp{qp}.json'
    if dest.exists():
        saved=load(dest);assert saved['status']=='PASS' and saved['identity']==identity
        for p,d in saved['record_SHA256'].items():assert sha(ROOT/p)==d
        return saved
    c=None
    if group!='Original':
        c=construct('A' if group=='A2000' else group,False,rank)
        if cp is not None:restore(c,torch.load(cp,map_location='cpu',weights_only=True))
        assert_reset(c);before=component_hashes(c);full_before=tensor_hash(c['pm'].state_dict().items())
    quality,flow,perceptual,network_hashes=metrics_setup(rank)
    from src.utils import stream_helper as sh
    reader=io.BytesIO(Path(info['stream_path']).read_bytes());helper=sh.SPSHelper();assert sha(info['stream_path'])==info['stream_SHA256']
    device=torch.device('cuda',rank);previous=previous_gt=None;frames=[];transitions=[];traces=[];decoder_position=[0];i_holder=[0]
    if c is not None:
        original_add=c['pm'].add_ref_frame
        def immutable_reference(feature=None,frame=None,increase_poc=True):
            # Keep the original codec reference trajectory, including RGB.
            base_frame=data['original'][i_holder[0]].to(device)
            return original_add(feature,base_frame,increase_poc)
        c['pm'].add_ref_frame=immutable_reference
    writer=None;video_path=None
    if export:
        from videos import open_lossless
        video_path=ROOT/'videos'/f'{seq}_QP{qp}_{group}_step{step}_bytes{info["real_bytes"]}.mkv'
        writer=open_lossless(video_path,w,h,row.get('source_fps','30/1'))
    start=time.time();torch.cuda.reset_peak_memory_stats(rank)
    with torch.no_grad():
        for i in range(n):
            i_holder[0]=i;pos=reader.tell();header=sh.read_header(reader)
            while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
            sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
            assert q==info['actual_qps'][i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
            if c is not None:
                if i==0:
                    raw=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,raw)
                    assert tensor_sha(raw)==tensor_sha(data['original'][0]),'I_FRAME_MISMATCH'
                else:
                    gen=c['models']['generator'];gen.set_temporal_cascade_grad(False)
                    for name in ['tcg_low','tcg_mid','tcg_out']:
                        getattr(gen,name).alignment=__import__('temporal').alignment_field(data,i,device) if group in 'CD' else None
                    raw=c['pm'].decompress(payload,sps,q)['x_hat']
                    assert tensor_sha(c['pm'].dpb[0].feature)==info['codec_trace'][i]['codec_feature_SHA256'],'CANDIDATE_CODEC_FEATURE_CHANGED'
                    assert tensor_sha(c['pm'].dpb[0].frame)==tensor_sha(data['original'][i]),'CODEC_RGB_REFERENCE_NOT_FROZEN'
                out=c['core'].unit(raw,h,w)
            else:
                out=(data['original'][i].to(device).float().clamp(-1,1)+1)/2;out=out[...,:h,:w];raw=data['original'][i]
            target=(data['GT'][i].permute(2,0,1)[None].float()/255).to(device)
            # The sampler used here for old PWC metrics is unchanged.
            from evaluate import backend
            eco=backend(False,rank)[2] if i==0 else eco
            spatial=eco.frame_metrics(out,target,quality)
            record=dict(frame=i,source_frame=row['source_frame_indices'][i],actual_QP=q,source_RGB_SHA256=info['codec_trace'][i]['source_RGB_SHA256'],GT_RGB_FP32_tensor_SHA256=tensor_sha(target),reconstruction_SHA256=tensor_sha(raw),RGB_tensor_SHA256=tensor_sha(out),real_bits=(reader.tell()-pos)*8,**spatial)
            frames.append(record);traces.append(info['codec_trace'][i])
            if i:
                difference=flow(previous_gt,target)-flow(previous,out)
                flo=float(perceptual(previous_gt,previous,difference,normalize=True).item())
                old_sampler=sys.modules['pwcnet'].backwarp
                gt_flow=flow(target,previous_gt)
                oldgt=float((target-old_sampler(previous,gt_flow)).abs().mean());oldrecon=float((target-old_sampler(previous,flow(out,previous))).abs().mean())
                rr,re,valid=temporal_errors(out,previous,target,previous_gt,data['GT_backward'][i].to(device),data['GT_valid'][i].to(device))
                valid_pixels=int(data['GT_valid'][i].sum())
                transitions.append(dict(from_frame=i-1,to_frame=i,transition_type='I_to_P' if i==1 else 'P_to_P',FloLPIPS=flo,MCResidual_RGB_RAFT_pixelcenter_v1=None if rr is None else float(rr),MCResidual_Sobel_RAFT_pixelcenter_v1=None if re is None else float(re),valid_fraction=valid,
                    temporal_status='VALID' if valid_pixels else 'EMPTY_GT_VALID_REGION',valid_pixels=valid_pixels,RGB_valid_channel_denominator=valid_pixels*3,Sobel_valid_channel_denominator=valid_pixels*6,GT_Sobel_strength=float(sobel(target).abs().mean()),reconstruction_Sobel_strength=float(sobel(out).abs().mean()),WarpingError_GTFlow_PWC=oldgt,WarpingError_ReconFlow_PWC=oldrecon,
                    GT_backward_flow_SHA256=tensor_sha(data['GT_backward'][i]),GT_valid_mask_SHA256=tensor_sha(data['GT_valid'][i])))
            if writer is not None:writer.stdin.write(out[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy().tobytes())
            previous,previous_gt=out,target
            dump(root/f'progress_qp{qp}.json',dict(sequence=seq,frame=i,time=time.time(),group=group,step=step))
    if writer is not None:writer.stdin.close();assert writer.wait()==0
    assert reader.tell()==info['real_bytes'] and sum(r['real_bits'] for r in frames)==info['real_bytes']*8
    if c is not None:
        assert component_hashes(c)==before and tensor_hash(c['pm'].state_dict().items())==full_before
        assert_reset(c)
    def transition_mean(key,rows):
        values=[r[key] for r in rows if r[key] is not None]
        return sum(values)/len(values) if values else None
    temporal_valid=[r for r in transitions if r['MCResidual_RGB_RAFT_pixelcenter_v1'] is not None]
    ptop=transitions[1:];ptop_valid=[r for r in ptop if r['MCResidual_RGB_RAFT_pixelcenter_v1'] is not None]
    metrics=dict(model=group,step=step,domain='ULong_development_subset_12',split='development',sequence=seq,QP=qp,frames=n,transitions=n-1,
        real_bytes=info['real_bytes'],bpp=info['bpp'],kbps=info['real_bytes']*8*float(__import__('fractions').Fraction(row.get('source_fps','30/1')))/n/1000,
        PSNR=-10*math.log10(max(sum(r['pixel_SSE'] for r in frames)/(n*h*w*3),1e-15)),**{k:sum(r[k] for r in frames)/n for k in SPATIAL},
        **{k:transition_mean(k,temporal_valid if k.startswith('MCResidual_') else transitions) for k in TEMPORAL},
        **{k+'_PtoP':transition_mean(k,ptop_valid if k.startswith('MCResidual_') else ptop) for k in TEMPORAL},
        **{k+'_I_to_P':transitions[0][k] for k in TEMPORAL},valid_temporal_transitions=len(temporal_valid),empty_temporal_transitions=(n-1)-len(temporal_valid),
        valid_temporal_transitions_PtoP=len(ptop_valid),empty_temporal_transitions_PtoP=len(ptop)-len(ptop_valid),valid_fraction_PtoP=sum(r['valid_fraction'] for r in ptop)/(n-2),
        bitstream_SHA256=info['stream_SHA256'],codec_trace_SHA256=digest(traces),source_RGB_SHA256=digest([r['source_RGB_SHA256'] for r in frames]),checkpoint_SHA256=checkpoint_hash)
    assert all(math.isfinite(float(v)) for k,v in metrics.items() if isinstance(v,(float,int)))
    records={}
    for name,rows in [('frames',frames),('transitions',transitions),('codec_trace',traces)]:
        p=root/name/f'{seq}_qp{qp}.csv';write_csv(p,rows);records[str(p.relative_to(ROOT))]=sha(p)
    record=dict(status='PASS',identity=identity,result=metrics,record_SHA256=records,real_RANS=True,fresh_candidate_decoder=c is not None,original_independent_decode_cached=c is None,
        stream_reused=True,fresh_encode=False,bytes_consumed=reader.tell(),codec_features_match_original=True,codec_references_forced_to_original=True,GT_not_used_for_candidate_alignment=True,
        frozen_components_unchanged=True,metric_network_hashes=network_hashes,initial_A2000_SHA256=INIT_SHA,actual_qps=info['actual_qps'],source_cache_receipt=str(cache_receipt),source_cache_receipt_SHA256=sha(cache_receipt),
        sampler_version=VERSION,old_PWC_metric_definition_unchanged=True,seconds=time.time()-start,physical_GPU=physical_gpu(rank),peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
        alignment_cache_generation_wall_seconds_including_RANS_and_GT_RAFT=info['original_base_and_RAFT_seconds'] if group in 'CD' else 0.,alignment_cache_generation_peak_reserved_bytes=info['peak_reserved_bytes'] if group in 'CD' else 0,
        lossless_video=None if video_path is None else dict(path=str(video_path),SHA256=sha(video_path),format='FFV1 RGB quantized to8bit for viewing; metrics use original float32 decoded RGB'))
    if engineering_output and group=='A2000':
        from legacy_baseline_check import check
        check(record,root)
    if engineering_output and group in 'CD':
        del quality,flow,perceptual;torch.cuda.empty_cache()
        from alignment import check_online
        dump(ROOT/f'audits/online_alignment_native_{group}.json',check_online(c,data,frames=(1,2,3)))
        quality=flow=perceptual=None
    dump(dest,record);del data,c,quality,flow,perceptual;torch.cuda.empty_cache();disk()
    return record

def evaluate_group(group,step,rank):
    env()
    for profile in load(ROOT/'manifests/quick_development.json'):
        for q in [0,5,9]:point(profile,q,group,step,rank)
    rows=[load(p)['result'] for p in (ROOT/f'evaluation/{group}/step_{step:04d}/points').glob('*.json')];assert len(rows)==36
    write_csv(ROOT/f'evaluation/{group}/step_{step:04d}/metrics.csv',rows)
    dump(ROOT/f'evaluation/{group}/step_{step:04d}/completion.json',dict(status='PASS',group=group,step=step,points=36,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--step',type=int,default=0);p.add_argument('--rank',type=int,required=True);p.add_argument('--index',type=int);p.add_argument('--export',action='store_true');p.add_argument('--probe-native',action='store_true');a=p.parse_args();env()
    if a.export:
        for profile in load(ROOT/'manifests/long_videos.json'):point(profile,5,a.group,a.step,a.rank,export=True)
    elif a.index is not None:
        profile=load(ROOT/'manifests/quick_development.json')[a.index//3];point(profile,[0,5,9][a.index%3],a.group,a.step,a.rank,engineering_output=a.probe_native)
    else:evaluate_group(a.group,a.step,a.rank)
