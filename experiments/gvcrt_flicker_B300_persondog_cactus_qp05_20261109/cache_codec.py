"""Original-only producer: real RANS + independent decode + causal RAFT.

Run in its own process namespace. Candidate weights never enter the codec.
"""
from common_s2 import *
import argparse,io

def cache_key(row,qp,training):
    return digest(dict(row=row,QP=qp,training=training,original_commit=ORIGINAL_COMMIT,RAFT_SHA256=RAFT_SHA,precision='original_FP16_RGB_FP32first',version='flicker_cache_v1' if training else 'flicker_native_v2_CPU_GT'))

def cache_paths(row,qp,training):
    key=cache_key(row,qp,training)
    return key,ROOT/'cache/derived'/f'{key}.pt',ROOT/'audits/codec_cache'/f'{key}.json'

def load_raft(device):
    import torch
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    assert sha(RAFT)==RAFT_SHA
    m=raft_large(weights=None,progress=False);m.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
    return m.to(device).eval().requires_grad_(False),Raft_Large_Weights.C_T_SKHT_V2.transforms()

def native_arrays(row,training):
    assert not training,'INFERENCE_ONLY_TASK'
    from data_reader import read_arrays
    return read_arrays(row)


def ensure(row,qp,training=False,rank=0):
    env();key,path,receipt=cache_paths(row,qp,training)
    with lock('locks/derived_'+key+'.lock',blocking=True):
        if path.exists() and not receipt.exists():
            import torch
            saved=torch.load(path,map_location='cpu',weights_only=True)
            info=saved['info'];assert info['cache_identity']==key and info['source_row']==row
            assert sha(info['stream_path'])==info['stream_SHA256']
            assert sum(t['real_bits'] for t in info['codec_trace'])==info['real_bytes']*8
            for i,t in enumerate(info['codec_trace']):
                assert tensor_sha(saved['original'][i])==t['original_reconstruction_SHA256']
                assert (None if i==0 else tensor_sha(saved['features'][i]))==t['codec_feature_SHA256']
                assert hashlib.sha256(saved['GT'][i].numpy().tobytes()).hexdigest()==t['source_RGB_SHA256']
            dump(receipt,dict(info,status='PASS',cache_sha256=sha(path),cache_bytes=path.stat().st_size,receipt_recovered_after_atomic_tensor_save=True))
            del saved
        if path.exists() and receipt.exists():
            a=load(receipt)
            assert a['status']=='PASS' and a['cache_identity']==key and path.stat().st_size==a['cache_bytes']
            if training:assert a['actual_qps']==row['actual_frame_qps']
            assert sha(path)==a['cache_sha256'],'DERIVED_CACHE_HASH_MISMATCH'
            os.utime(path,None);return a
        disk()
        import torch,numpy as np
        from original_context import runtime
        from original_runtime import load_models
        rt=runtime(rank);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
        im,pe=load_models(rt);idec,pd=load_models(rt);sh=rt['stream']
        for pm in [pe,pd]:pm.clear_dpb();pm.set_curr_poc(0)
        before=[tensor_hash(m.state_dict().items()) for m in [im,pe,idec,pd]]
        arrays=native_arrays(row,training);n=len(arrays);assert n==(64 if not row.get('long_video') else 128)
        h,w=arrays[0].shape[:2];ch,cw=(h+63)//64*64,(w+63)//64*64
        sps=dict(sps_id=0,height=ch,width=cw,ec_part=1,use_ada_i=0)
        stream=io.BytesIO();sh.write_sps(stream,sps);encoded_hashes=[];features_hash=[];actual=[]
        def gt(i):
            x=torch.from_numpy(arrays[i]).permute(2,0,1)[None]
            return x.to(device).float()/255 if training else (x.float()/255).to(device)
        start=time.time();torch.cuda.reset_peak_memory_stats(rank)
        with torch.no_grad():
            for i in range(n):
                x=rt['replicate_pad'](gt(i).half(),ch-h,cw-w)*2-1;q=qp if i==0 else pe.shift_qp(qp,(0,1,0,2,0,2,0,2)[i%8]);actual.append(q)
                enc=im.compress(x,q) if i==0 else pe.compress(x,q)
                dec=idec.decompress(enc['bit_stream'],sps,q) if i==0 else pd.decompress(enc['bit_stream'],sps,q)
                if i==0:pe.add_ref_frame(None,dec['x_hat']);pd.add_ref_frame(None,dec['x_hat'])
                else:assert torch.equal(pe.dpb[0].feature,pd.dpb[0].feature),'REAL_RANS_FEATURE_SYNC_MISMATCH'
                if i==0:assert torch.equal(enc['x_hat'],dec['x_hat']),'REAL_RANS_RECON_SYNC_MISMATCH'
                encoded_hashes.append(tensor_sha(dec['x_hat']));features_hash.append(None if i==0 else tensor_sha(pd.dpb[0].feature))
                sh.write_ip(stream,i==0,0,q,enc['bit_stream'])
            raw=stream.getvalue();large=ROOT/'cache/streams'/f'{key}.bin';large.parent.mkdir(parents=True,exist_ok=True);temp=large.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(large)
            assert before==[tensor_hash(m.state_dict().items()) for m in [im,pe,idec,pd]],'ENCODER_OR_SYNC_DECODER_FROZEN_STATE_CHANGED'
            # New independent decoder instance. No encoder or sync decoder cache.
            del im,pe,idec,pd,enc,dec,x
            torch.cuda.empty_cache();ii,pm=load_models(rt);pm.clear_dpb();pm.set_curr_poc(0)
            reader=io.BytesIO(raw);helper=sh.SPSHelper();features=[];base=[];traces=[]
            independent_before=[tensor_hash(m.state_dict().items()) for m in [ii,pm]]
            for i in range(n):
                pos=reader.tell();header=sh.read_header(reader)
                while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
                current=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
                assert current==sps and q==actual[i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
                decoded=ii.decompress(payload,current,q) if i==0 else pm.decompress(payload,current,q)
                if i==0:pm.add_ref_frame(None,decoded['x_hat'])
                f=None if i==0 else pm.dpb[0].feature.detach().cpu().clone()
                assert tensor_sha(decoded['x_hat'])==encoded_hashes[i]
                assert (None if f is None else tensor_sha(f))==features_hash[i]
                features.append(f);base.append(decoded['x_hat'].detach().cpu().clone())
                traces.append(dict(frame=i,actual_QP=q,real_bits=(reader.tell()-pos)*8,payload_SHA256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload),codec_feature_SHA256=features_hash[i],original_reconstruction_SHA256=encoded_hashes[i],source_RGB_SHA256=hashlib.sha256(arrays[i].tobytes()).hexdigest()))
            assert reader.tell()==len(raw) and sum(t['real_bits'] for t in traces)==len(raw)*8
            assert independent_before==[tensor_hash(m.state_dict().items()) for m in [ii,pm]]
            qscale=pm.q_scale_recon.detach().cpu().clone();del ii,pm,decoded;torch.cuda.empty_cache()
            rans_seconds=time.time()-start
            raft,transform=load_raft(device);base_flows=[None];base_valid=[None];gt_back=[None];gt_forward=[None];gt_valid=[None]
            raft_start=time.time()
            def flow(a,b):
                a,b=transform(a,b);return raft(a,b)[-1]
            from temporal import flow_valid
            for i in range(1,n):
                # base is original-only and detached. Both directions use only
                # current and previous frames, never future frames or GT.
                bp=rt['unit'](base[i-1].to(device),h,w);bc=rt['unit'](base[i].to(device),h,w)
                # Replicate pad for RAFT divisibility; crop field to native RGB.
                def pad(z):return torch.nn.functional.pad(z,(0,(-w)%8,0,(-h)%8),mode='replicate')
                backward=flow(pad(bc),pad(bp))[...,:h,:w];forward=flow(pad(bp),pad(bc))[...,:h,:w]
                mask=flow_valid(backward,forward)
                base_flows.append(backward.cpu());base_valid.append(mask.cpu())
                if not training or i>=49:
                    g=gt(i);p=gt(i-1);b=flow(pad(g),pad(p))[...,:h,:w];f=flow(pad(p),pad(g))[...,:h,:w]
                    gt_back.append(b.cpu());gt_forward.append(f.cpu());gt_valid.append(flow_valid(b,f).cpu())
                else:gt_back.append(None);gt_forward.append(None);gt_valid.append(None)
            flow_seconds=time.time()-raft_start
            info=dict(cache_identity=key,source_row=row,external_QP=qp,training=training,frames=n,native_shape=[h,w],padded_shape=[ch,cw],
                actual_qps=actual,stream_path=str(large),stream_SHA256=sha(large),real_bytes=len(raw),bpp=len(raw)*8/(n*h*w),codec_trace=traces,
                original_commit=ORIGINAL_COMMIT,original_weights={name:sha(PROJECT/'checkpoints'/name) for name in ['GVC-RT_I.pt','GVC-RT_P.pt']},RAFT_SHA256=RAFT_SHA,
                base_receiver_no_TCG_LoRA_Router=True,GT_not_used_by_alignment=True,independent_decode=True,real_RANS=True,rans_seconds=rans_seconds,original_base_and_RAFT_seconds=rans_seconds+flow_seconds,RAFT_seconds=flow_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),physical_GPU=physical_gpu(rank),codec_state_unchanged=True)
            tensors=dict(features=features,original=base,GT=torch.from_numpy(np.stack(arrays)),base_backward=base_flows,base_valid=base_valid,GT_backward=gt_back,GT_forward=gt_forward,GT_valid=gt_valid,qscale=qscale,info=info)
            atomic_torch(path,tensors)
            proof=dict(info,status='PASS',cache_sha256=sha(path),cache_bytes=path.stat().st_size)
            dump(receipt,proof);event('original_codec_cache_complete',key=key,frames=n,QP=qp,seconds=time.time()-start)
        return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--qp',type=int,required=True);p.add_argument('--training',action='store_true');p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    ensure(load(a.row),a.qp,a.training,a.rank)
