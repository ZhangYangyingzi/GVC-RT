"""Read-only real decoder probe for channel entropy attribution."""
import os
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='7')
from runtime import *
from channel_rate import attach,finish,cdf_bits
import io

def main():
    import torch
    sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v2_joint'))
    import core
    from src.utils.stream_helper import SPSHelper,read_header,read_sps_remaining,read_ip_remaining,NalType
    im,pm=core.load_models(torch.device('cuda:0'),force_zero_thres=.12)
    r=load(ROOT/'evaluation/points'/f'ulong_0_{BRANCHES[0]}_0.json')['result']
    def decode(observe):
        stream=io.BytesIO(Path(r['bitstream_path']).read_bytes());helper=SPSHelper();pm.clear_dpb();pm.set_curr_poc(0)
        state,detach=attach(pm);hashes=[]
        with torch.inference_mode():
            for idx in range(33):
                state['enabled']=observe and idx==32
                header=read_header(stream)
                while header['nal_type']==NalType.NAL_SPS:
                    helper.add_sps_by_id(read_sps_remaining(stream,header['sps_id']));header=read_header(stream)
                sp=helper.get_sps_by_id(header['sps_id']);qp,bits=read_ip_remaining(stream)
                state['qp']=qp
                if idx==0:
                    out=im.decompress(bits,sp,qp);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,out['x_hat'])
                else:out=pm.decompress(bits,sp,qp)
                hashes.append(core.module_hash(torch.nn.ParameterList([torch.nn.Parameter(out['x_hat'],requires_grad=False)])))
        detach()
        return hashes,finish(state) if observe else None,bits
    before=core.compression_hash(im,pm);a,_,payload=decode(False);b,result,payload2=decode(True)
    assert a==b and before==core.compression_hash(im,pm)
    # Independently sum the exact packed symbols/indexes sent to the native
    # encoder, without using the decoder channel mapping being tested.
    scalar=[];scalar_z=[];enabled=[False];encode=pm.entropy_coder.encode_y;encode_z=pm.entropy_coder.encode_z
    def observe(symbols,group):
        if enabled[0]:
            packed=symbols.detach().cpu().numpy().astype('int32')
            _,summary=cdf_bits(packed>>8,packed&255,pm.gaussian_encoder)
            scalar.append(summary['total_estimated_bits'])
        return encode(symbols,group)
    pm.entropy_coder.encode_y=observe
    def observe_z(symbols,group,start_offset,per_channel_size):
        if enabled[0]:
            values=symbols.detach().cpu().numpy().astype('int64').flatten()
            indexes=np.arange(len(values))//per_channel_size+start_offset
            _,summary=cdf_bits(values,indexes,pm.bit_estimator_z);scalar_z.append(summary['total_estimated_bits'])
        return encode_z(symbols,group,start_offset,per_channel_size)
    import numpy as np
    pm.entropy_coder.encode_z=observe_z
    sys.path.insert(0,str(V32));from adapter32 import frames_for
    v=load(V32/'eval_manifest.json')['ulong'][0];frames=frames_for(v)
    pm.clear_dpb();pm.set_curr_poc(0)
    with torch.inference_mode():
        for idx in range(33):
            enabled[0]=idx==32;q=r['actual_qps'][idx];x=core.codec_input(frames[idx].cuda())
            if idx==0:
                out=im.compress(x,q);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,out['x_hat'])
            else:out=pm.compress(x,q)
    pm.entropy_coder.encode_y=encode;pm.entropy_coder.encode_z=encode_z
    assert out['bit_stream']==payload==payload2
    encoder_error=abs(sum(scalar)-result['estimated_y_total_bits']);assert encoder_error<1e-7,encoder_error
    z_error=abs(sum(scalar_z)-result['hyperlatent']['total_estimated_bits']);assert z_error<1e-7,z_error
    dump('sanity/channel_rate_probe.json',dict(status='PASS',hook_output_hashes_equal_all_33=True,compression_hash_unchanged=True,independent_encoder_payload_matches_persisted_frame32=True,independent_encoder_packed_CDF_estimate=sum(scalar),decoder_channel_vs_encoder_overall_y_estimate_abs_error=encoder_error,decoder_vs_encoder_z_estimate_abs_error=z_error,actual_frame32_payload_bits=len(payload)*8,result=result))
    print('RATE PROBE PASS',result['estimated_y_total_bits'],result['sum_channel_vs_combined_y_estimate_abs_error'],flush=True)

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump('sanity/channel_rate_probe_failure.json',dict(traceback=traceback.format_exc()));raise
