"""Exercise actual entropy decoder -> generator -> adapter -> head/LoRA entry."""
from common_s2 import *
import io


def packets(c,info):
    from src.utils import stream_helper as sh
    reader=io.BytesIO(Path(info['stream_path']).read_bytes());helper=sh.SPSHelper();out=[]
    for i in range(64):
        header=sh.read_header(reader)
        while header['nal_type']==sh.NalType.NAL_SPS:
            helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
        sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
        assert q==info['actual_qps'][i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
        out.append((sps,q,payload))
    assert reader.tell()==info['real_bytes'];return out


def decode(c,item,i):
    sps,q,payload=item
    if i==0:
        result=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,result)
    else:result=c['pm'].decompress(payload,sps,q)['x_hat']
    return result


def main():
    env();import torch
    from cache_codec import cache_paths
    from experiment_model import construct
    from model_context import assert_reset
    v=load(ROOT/'manifests/development_videos.json')[0];_,p,rp=cache_paths(v.get('source_row',v),0,False)
    assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    c=construct('B300',False,0);items=packets(c,d['info']);h,w=d['info']['native_shape'];records=[]
    baseline=load(H_SOURCE/f'evaluation/development/B300/step_0000/points/{v["dataset"]}__{v["sequence"]}_qp0.json')
    frpath=next(H_SOURCE/f for f in baseline['record_SHA256'] if '/frames/' in f)
    expected=list(csv.DictReader(frpath.open()))
    for changed in (False,True):
        assert_reset(c);c['adapter'].forward_calls=0;c['models']['generator'].head.lora.forward_calls=0;different=0
        if changed:
            with torch.no_grad():c['adapter'].B.bias.fill_(.01)
        with torch.no_grad():
            # Two continuous chunks, preserving all codec and adapter histories.
            for first,last in [(0,4),(4,8)]:
                for i in range(first,last):
                    raw=decode(c,items[i],i);out=((raw.float().clamp(-1,1)+1)/2)[...,:h,:w]
                    f=None if i==0 else tensor_sha(c['pm'].dpb[0].feature)
                    assert f==d['info']['codec_trace'][i]['codec_feature_SHA256']
                    same=tensor_sha(out)==expected[i]['RGB_SHA256']
                    if not changed:assert same,('FULL_DECODER_ZERO_BRANCH_MISMATCH',i)
                    different+=not same
        assert c['adapter'].forward_calls==7 and c['models']['generator'].head.lora.forward_calls==7
        if changed:assert different>0
        records.append(dict(nonzero_B=changed,frames=8,adapter_calls=7,head_LoRA_calls=7,codec_feature_hashes_match=True,
            changed_RGB_frames=different,continuous_chunks=[4,4],history_bytes=c['adapter'].history_bytes()))
    assert_reset(c);assert not c['adapter'].history
    dump(ROOT/'audits/full_decoder_deployment.json',dict(status='PASS',records=records,stream_SHA256=d['info']['stream_SHA256'],
        GT_read_by_decoder=False,actual_RANS_decompress_exercised=True,physical_GPU=physical_gpu(0),time=time.time()))

if __name__=='__main__':main()
