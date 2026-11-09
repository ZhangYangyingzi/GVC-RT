"""Six fixed native points: exact B300 inference bytes and E/F parity."""
from common_s2 import *
import argparse

def main(group,rank):
    import torch
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    from pipeline import wait_rows
    env();checks=[]
    for i in [0,4]:
      profile=load(ROOT/'manifests/quick_development.json')[i]
      for qp in [0,5,9]:
        dest=ROOT/f'audits/step0/{group}_{i}_qp{qp}.json'
        if dest.exists():assert load(dest)['status']=='PASS';checks.append(load(dest));continue
        row=profile['row'];wait_rows([row],qp,False)
        key,path,rp=cache_paths(row,qp,False);proof=load(rp);assert sha(path)==proof['cache_sha256'];data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        c=construct(group,False,rank);assert_reset(c);frames=[]
        parent_root=PARENT_EXPERIMENT/'evaluation/B/step_0300';oldpoint=load(parent_root/'points'/f'{profile["sequence"]}_qp{qp}.json')
        assert oldpoint['identity']['checkpoint_SHA256']==B300_SHA
        assert oldpoint['result']['bitstream_SHA256']==proof['stream_SHA256'] and oldpoint['result']['real_bytes']==proof['real_bytes']
        oldframes=list(csv.DictReader((parent_root/'frames'/f'{profile["sequence"]}_qp{qp}.csv').open()))
        # Decode through the entire codec path as parent evaluation did; cached
        # codec features are checked at every P frame; candidate RGB is isolated.
        import io
        from src.utils import stream_helper as sh
        reader=io.BytesIO(Path(proof['stream_path']).read_bytes());helper=sh.SPSHelper();idx=[0]
        add=c['pm'].add_ref_frame
        def fixed(feature=None,frame=None,increase_poc=True):return add(feature,data['original'][idx[0]].to(rank),increase_poc)
        c['pm'].add_ref_frame=fixed
        with torch.no_grad():
          for j in range(64):
            idx[0]=j;header=sh.read_header(reader)
            while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
            sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
            if j==0:raw=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,raw)
            else:
                raw=c['pm'].decompress(payload,sps,q)['x_hat']
                assert tensor_sha(c['pm'].dpb[0].feature)==proof['codec_trace'][j]['codec_feature_SHA256']
            out=c['core'].unit(raw,*proof['native_shape']);hh=tensor_sha(out)
            assert hh==oldframes[j]['RGB_tensor_SHA256'],('STEP0_OUTPUT_DIFFERENCE',group,i,qp,j,hh,oldframes[j]['RGB_tensor_SHA256'])
            frames.append(hh)
        assert_reset(c)
        result=dict(status='PASS',group=group,profile=i,sequence=profile['sequence'],QP=qp,frames=64,exact_RGB_hashes_match_parent_B300=True,RGB_SHA256=digest(frames),codec_stream_match=True,codec_feature_trace_match=True,B300_SHA256=B300_SHA,precision=c['provenance']['precision'],time=time.time())
        dump(dest,result);checks.append(result);del c,data;torch.cuda.empty_cache()
    dump(ROOT/f'audits/step0_{group}.json',dict(status='PASS',points=6,checks=checks))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.group,a.rank)
