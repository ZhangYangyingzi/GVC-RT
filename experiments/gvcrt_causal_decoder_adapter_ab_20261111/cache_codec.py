"""Frozen Original-only codec/GT-flow caches, read-only legacy reuse, atomic identity."""
from common_s2 import *
import argparse,io

def legacy_key(row,qp,training):
    return digest(dict(row=row,QP=qp,training=training,original_commit=ORIGINAL_COMMIT,RAFT_SHA256=RAFT_SHA,precision='original_FP16_RGB_FP32first',version='flicker_cache_v1' if training else 'flicker_native_v2_CPU_GT'))

def cache_key(row,qp,training):return digest(dict(schema='teacher_nolora_fixed_v1',legacy_key=legacy_key(row,qp,training),base_alignment_cached=False))
def cache_paths(row,qp,training):
    key=cache_key(row,qp,training);p=ROOT/'cache/derived'/f'{key}.pt';rp=ROOT/'audits/codec_cache'/f'{key}.json'
    if not p.exists():
        for src in [H_SOURCE,PARENT_EXPERIMENT,EF_SOURCE,G_SOURCE]:
            old=src/'cache/derived'/f'{key}.pt';proof=src/'audits/codec_cache'/f'{key}.json'
            if old.exists() and proof.exists():
                a=load(proof)
                if a['status']=='PASS' and a['cache_identity']==key and old.stat().st_size==a['cache_bytes']:return key,old,proof
    if not p.exists():
        oldkey=legacy_key(row,qp,training)
        for src in [H_SOURCE,PARENT_EXPERIMENT,EF_SOURCE,G_SOURCE]:
            old=src/'cache/derived'/f'{oldkey}.pt';proof=src/'audits/codec_cache'/f'{oldkey}.json'
            if (ROOT/'audits/cache_rejected'/f'{oldkey}.json').exists():continue
            if old.exists() and proof.exists():
                a=load(proof)
                if a['status']=='PASS' and a['cache_identity']==oldkey and old.stat().st_size==a['cache_bytes']:return oldkey,old,proof
    return key,p,rp

def canonical(row,qp):
    seq=Path(row['video']).stem if 'video' in row else row['sequence'];split='development' if row['split']=='development' else 'final'
    if row.get('domain')=='mcljcv':
        p=REFERENCE_EXPERIMENT/f'canonical_MCL/points/mcljcv_{seq}_qp{qp}.json';a=load(p)
        assert a['status']=='PASS' and a['real_RANS'] and a['independent_decode'] and not a['native_RGB_bypass'] and a['I_frame_bits_included']
        r=a['result'];stream=Path(a['stream_path']);h=r['bitstream_sha256'];trace_path=Path(a['canonical_codec_trace_path'])
        source_rgb=r['source_RGB_sha256'];source_sha=r['source_sha256'];size=r['real_bytes'];old_bpp=r['bpp']
    else:
        entries=load(ROOT/'manifests/bitstream_manifest.json')['streams'];entry=next(x for x in entries if (x['split'],x['sequence'],x['QP'])==(split,seq,qp))
        p=Path(entry['point_path']);assert sha(p)==entry['point_sha256'];a=load(p);stream=Path(entry['stream_path']);h=entry['stream_sha256'];size=entry['real_bytes'];old_bpp=entry['bpp'];source_sha=entry['source_sha256'];source_rgb=entry['source_RGB_sha256']
        trace_path=p.parent.parent/'encoder_reference'/p.with_suffix('.csv').name
        assert a['status']=='PASS' and a['real_RANS'] and a['independent_decode']
    assert sha(stream)==h and stream.stat().st_size==size
    assert source_sha==row['source_sha256'] and a['source_frame_indices']==row['source_frame_indices']
    expected_rgb=row.get('rgb_sha256');assert expected_rgb==source_rgb
    with trace_path.open() as f:trace=list(csv.DictReader(f))
    assert len(trace)==64
    expected_features=[None if i==0 else t.get('reference_feature_sha256',t.get('codec_feature_sha256')) for i,t in enumerate(trace)]
    assert all(expected_features[i] for i in range(1,64))
    return dict(path=stream,SHA256=h,bytes=size,bpp=old_bpp,actual_qps=a['actual_qps'],features=expected_features,source_point=str(p),source_point_SHA256=sha(p),source_trace=str(trace_path),source_trace_SHA256=sha(trace_path))

def native_arrays(row,training):
    if training:
        from data_reader import read_arrays
        return read_arrays(row)
    if 'sequence' in row:
        from eval_data import read_frames
        return read_frames(row)
    from inherited_reader import read_arrays,source_file
    if row.get('long_video'):
        import cv2
        out=[];cv2.setNumThreads(1)
        with source_file(row) as p:
            cap=cv2.VideoCapture(str(p));cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_indices'][0])
            try:
                for i in row['source_frame_indices']:
                    ok,bgr=cap.read();assert ok;out.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            finally:cap.release()
        return out
    return read_arrays(row,full=True)

def load_raft(device):
    import torch
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    assert sha(RAFT)==RAFT_SHA;m=raft_large(weights=None,progress=False);m.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
    return m.to(device).eval().requires_grad_(False),Raft_Large_Weights.C_T_SKHT_V2.transforms()

def ensure(row,qp,training=False,rank=0):
    env();key,path,rp=cache_paths(row,qp,training)
    # Tensor and receipt are one generation; consumers and eviction share this lease.
    with lock('locks/lease_'+key+'.lock',blocking=True), lock('locks/derived_'+key+'.lock',blocking=True):
        if path.exists():
            if not rp.exists():
                import torch
                data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert data['info']['cache_identity']==key;assert sha(data['info']['stream_path'])==data['info']['stream_SHA256']
                dump(rp,dict(data['info'],status='PASS',cache_sha256=sha(path),cache_bytes=path.stat().st_size,recovered_atomic_tensor=True));del data
            a=load(rp);assert a['status']=='PASS' and a['cache_identity']==key and a['cache_bytes']==path.stat().st_size and sha(path)==a['cache_sha256']
            if training:assert a['actual_qps']==row['actual_frame_qps']
            return a
        if rp.exists():
            assert rp.is_relative_to(ROOT)
            archived=ROOT/'audits/codec_cache_history'/key/f'{time.time_ns()}.json';archived.parent.mkdir(parents=True,exist_ok=True);rp.replace(archived)
        disk();import torch,numpy as np
        from original_context import runtime
        from original_runtime import load_models
        rt=runtime(rank);device=torch.device('cuda',rank);sh=rt['stream'];arrays=native_arrays(row,training);n=len(arrays);assert n in [64,128]
        h,w=arrays[0].shape[:2];ph,pw=(h+63)//64*64,(w+63)//64*64;sps=dict(sps_id=0,height=ph,width=pw,ec_part=1,use_ada_i=0)
        def gt(i):
            x=torch.from_numpy(arrays[i]).permute(2,0,1)[None]
            return x.to(device).float()/255 if training else (x.float()/255).to(device)
        torch.cuda.reset_peak_memory_stats(rank);begun=time.time();expected=None;encoder_features=None;encoder_rgb=None
        with torch.no_grad():
            if training or row.get('long_video'):
                im,pe=load_models(rt);idec,pd=load_models(rt)
                for pm in [pe,pd]:pm.clear_dpb();pm.set_curr_poc(0)
                before=[tensor_hash(m.state_dict().items()) for m in [im,pe,idec,pd]];stream=io.BytesIO();sh.write_sps(stream,sps);actual=[];encoder_features=[];encoder_rgb=[]
                for i in range(n):
                    q=qp if i==0 else pe.shift_qp(qp,(0,1,0,2,0,2,0,2)[i%8]);actual.append(q);x=rt['replicate_pad'](gt(i).half(),ph-h,pw-w)*2-1
                    enc=im.compress(x,q) if i==0 else pe.compress(x,q);dec=idec.decompress(enc['bit_stream'],sps,q) if i==0 else pd.decompress(enc['bit_stream'],sps,q)
                    if i==0:assert torch.equal(enc['x_hat'],dec['x_hat']);pe.add_ref_frame(None,dec['x_hat']);pd.add_ref_frame(None,dec['x_hat'])
                    else:assert torch.equal(pe.dpb[0].feature,pd.dpb[0].feature)
                    encoder_rgb.append(tensor_sha(dec['x_hat']));encoder_features.append(None if i==0 else tensor_sha(pd.dpb[0].feature));sh.write_ip(stream,i==0,0,q,enc['bit_stream'])
                assert before==[tensor_hash(m.state_dict().items()) for m in [im,pe,idec,pd]]
                if training:assert actual==row['actual_frame_qps']
                raw=stream.getvalue();stream_path=ROOT/'cache/streams'/f'{key}.bin';stream_path.parent.mkdir(exist_ok=True);tmp=stream_path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(stream_path)
                del im,pe,idec,pd,enc,dec,x;torch.cuda.empty_cache()
            else:
                expected=canonical(row,qp);stream_path=expected['path'];raw=stream_path.read_bytes();actual=expected['actual_qps']
                assert len(raw)*8/(n*h*w)==expected['bpp']
            ii,pm=load_models(rt);pm.clear_dpb();pm.set_curr_poc(0);before=[tensor_hash(m.state_dict().items()) for m in [ii,pm]]
            reader=io.BytesIO(raw);helper=sh.SPSHelper();features=[];base=[];trace=[]
            for i in range(n):
                pos=reader.tell();header=sh.read_header(reader)
                while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
                current=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader);assert current==sps and q==actual[i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
                decoded=ii.decompress(payload,current,q) if i==0 else pm.decompress(payload,current,q)
                if i==0:pm.add_ref_frame(None,decoded['x_hat'])
                feature=None if i==0 else pm.dpb[0].feature.detach().cpu().clone();fh=None if i==0 else tensor_sha(feature);rh=tensor_sha(decoded['x_hat'])
                if expected:assert fh==expected['features'][i],('CANONICAL_FEATURE_TRACE_MISMATCH',i)
                if encoder_features is not None:assert fh==encoder_features[i] and rh==encoder_rgb[i]
                features.append(feature);base.append(decoded['x_hat'].detach().cpu().clone());trace.append(dict(frame=i,actual_QP=q,real_bits=(reader.tell()-pos)*8,payload_SHA256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload),codec_feature_SHA256=fh,original_reconstruction_SHA256=rh,source_RGB_SHA256=hashlib.sha256(arrays[i].tobytes()).hexdigest()))
            assert reader.tell()==len(raw) and sum(t['real_bits'] for t in trace)==len(raw)*8
            assert before==[tensor_hash(m.state_dict().items()) for m in [ii,pm]];qscale=pm.q_scale_recon.detach().cpu().clone();del ii,pm,decoded;torch.cuda.empty_cache()
            raft,transform=load_raft(device);back=[None]*n;forward=[None]*n;valid=[None]*n
            from temporal import flow_valid
            def pad(x):return torch.nn.functional.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
            for i in range(49 if training else 1,n):
                a,b=transform(pad(gt(i)),pad(gt(i-1)));bw=raft(a,b)[-1][...,:h,:w];a,b=transform(pad(gt(i-1)),pad(gt(i)));fw=raft(a,b)[-1][...,:h,:w]
                back[i]=bw.cpu();forward[i]=fw.cpu();valid[i]=flow_valid(bw,fw).cpu()
            info=dict(cache_identity=key,source_row=row,external_QP=qp,training=training,frames=n,native_shape=[h,w],padded_shape=[ph,pw],actual_qps=actual,stream_path=str(stream_path),stream_SHA256=sha(stream_path),real_bytes=len(raw),bpp=len(raw)*8/(n*h*w),codec_trace=trace,codec_trace_SHA256=digest(trace),original_commit=ORIGINAL_COMMIT,RAFT_SHA256=RAFT_SHA,base_receiver_no_TCG_LoRA_Router=True,independent_decode=True,real_RANS=True,codec_state_unchanged=True,canonical_stream_reused=expected is not None,canonical=None if expected is None else {k:str(v) if isinstance(v,Path) else v for k,v in expected.items() if k!='features'},seconds=time.time()-begun,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),physical_GPU=physical_gpu(rank),GT_normalization='CUDA FP32 /255' if training else 'CPU FP32 /255 before device transfer',base_alignment_cached=False)
            atomic_torch(path,dict(features=features,original=base,GT=torch.from_numpy(np.stack(arrays)),GT_backward=back,GT_forward=forward,GT_valid=valid,qscale=qscale,info=info))
            proof=dict(info,status='PASS',cache_sha256=sha(path),cache_bytes=path.stat().st_size);dump(rp,proof);event('codec_READY',key=key,training=training,seconds=info['seconds'],physical_GPU=physical_gpu(rank))
            return proof
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--qp',type=int,required=True);p.add_argument('--training',action='store_true');p.add_argument('--rank',type=int,default=0);a=p.parse_args();ensure(load(a.row),a.qp,a.training,a.rank)
