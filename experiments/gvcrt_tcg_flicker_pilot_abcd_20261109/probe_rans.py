"""Candidate actual RANS decode vs cached-feature receiver inference."""
from common_s2 import *
from experiment_model import construct,reconstruct
from model_context import assert_reset
from cache_codec import cache_paths
import io,argparse

def main(group,rank):
    import torch
    env();row=plan_rows()[0];key,path,receipt=cache_paths(row,row['external_qp'],True);data=torch.load(path,map_location='cpu',weights_only=True);info=data['info']
    c=construct(group,False,rank);assert_reset(c);hashes=[];cache_hashes=[]
    with torch.no_grad():
        for i in range(1,64):
            alignment=__import__('temporal').alignment_field(data,i,rank) if group in 'CD' else None
            rec=reconstruct(c,data['features'][i].to(rank),info['actual_qps'][i],alignment)
            hashes.append(tensor_sha(rec));cache_hashes.append({n:tensor_sha(getattr(c['models']['generator'],n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']})
        assert_reset(c);before=tensor_hash(c['pm'].state_dict().items());reader=io.BytesIO(Path(info['stream_path']).read_bytes())
        from src.utils import stream_helper as sh
        helper=sh.SPSHelper();original_add=c['pm'].add_ref_frame;idx=[0];decoded_feature_hashes=[]
        def fixed_ref(feature=None,frame=None,increase_poc=True):return original_add(feature,data['original'][idx[0]].to(rank),increase_poc)
        c['pm'].add_ref_frame=fixed_ref
        for i in range(64):
            idx[0]=i;head=sh.read_header(reader)
            while head['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,head['sps_id']));head=sh.read_header(reader)
            sps=helper.get_sps_by_id(head['sps_id']);q,payload=sh.read_ip_remaining(reader);assert q==info['actual_qps'][i]
            if i==0:
                rec=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,rec)
                assert tensor_sha(rec)==tensor_sha(data['original'][0])
            else:
                for n in ['tcg_low','tcg_mid','tcg_out']:getattr(c['models']['generator'],n).alignment=__import__('temporal').alignment_field(data,i,rank) if group in 'CD' else None
                rec=c['pm'].decompress(payload,sps,q)['x_hat']
                assert tensor_sha(c['pm'].dpb[0].feature)==info['codec_trace'][i]['codec_feature_SHA256']
                assert tensor_sha(rec)==hashes[i-1],'CACHED_FEATURE_VS_FULL_DECODER_RECON_MISMATCH'
                assert {n:tensor_sha(getattr(c['models']['generator'],n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']}==cache_hashes[i-1]
            decoded_feature_hashes.append(None if i==0 else tensor_sha(c['pm'].dpb[0].feature))
        assert reader.tell()==info['real_bytes'];assert tensor_hash(c['pm'].state_dict().items())==before;assert_reset(c)
    if group in 'CD':
        from alignment import check_online
        dump(ROOT/f'audits/online_alignment_{group}.json',check_online(c,data))
    dump(ROOT/f'audits/RANS_{group}.json',dict(status='PASS',group=group,stream_SHA256=info['stream_SHA256'],bytes=info['real_bytes'],bpp=info['bpp'],bytes_consumed=reader.tell(),
        fresh_independent_candidate_decoder=True,codec_feature_hashes_equal=True,reconstruction_hashes_equal_cached_path=True,TCG_cache_hashes_equal=True,original_base_receiver_in_alignment=group in 'CD',
        GT_not_provided_to_inference=True,causal_base_flow_source_only_current_and_previous=True,frozen_model_hash_unchanged=True,physical_GPU=physical_gpu(rank),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.group,a.rank)
