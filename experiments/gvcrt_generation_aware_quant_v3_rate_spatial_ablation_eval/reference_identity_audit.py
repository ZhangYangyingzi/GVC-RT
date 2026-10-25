"""Compare A and IDENTITY decoded codec state on all 64 fixed frames."""
import hashlib
import io
import json
from pathlib import Path

import task

P=Path(__file__).resolve().parent


def decode_state(mode,ctx):
    train,torch,_,_,_,_,_,INDEX_MAP,stream=ctx
    key=task.key(mode,'UVal0',2,2)
    record=json.loads((P/'server_artifacts/evaluation'/f'{key}.json').read_text())
    path=Path(record['stream_path'])
    assert task.sha(path)==record['stream_sha256']
    header_mode,alpha,native=task.read_header(path.read_bytes())
    assert (header_mode,alpha)==(mode,2)
    im,pm,checkpoint_sha=task.load_models(ctx,header_mode,alpha)
    assert checkpoint_sha==record['checkpoint_sha256']
    source=task.source_record('UVal0')
    h,w=source['height'],source['width']
    sps={'sps_id':0,'height':(h+63)//64*64,'width':(w+63)//64*64,
         'ec_part':int(((h+63)//64*64)*((w+63)//64*64)>1280*720),'use_ada_i':0}
    im.set_use_two_entropy_coders(bool(sps['ec_part']));pm.set_use_two_entropy_coders(bool(sps['ec_part']))
    pm.clear_dpb();pm.set_curr_poc(0)
    stream_file=io.BytesIO(native);helper=stream.SPSHelper();states=[]
    with torch.inference_mode():
        for i in range(64):
            head=stream.read_header(stream_file)
            while head['nal_type']==stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(stream_file,head['sps_id']))
                head=stream.read_header(stream_file)
            codec_sps=helper.get_sps_by_id(head['sps_id'])
            assert codec_sps==sps
            qp,bits=stream.read_ip_remaining(stream_file)
            assert qp==(2 if i==0 else pm.shift_qp(2,INDEX_MAP[i%8]))
            if i==0:
                result=im.decompress(bits,codec_sps,qp)
                pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,result['x_hat'])
                ref=result['x_hat'];latent=qmap=None
            else:
                pm.decompress(bits,codec_sps,qp)
                ref=pm.dpb[0].feature
                latent=pm.decoding_trace['y_hat'];qmap=pm.q_actual
            def digest(x):return None if x is None else hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
            states.append({'reference':digest(ref),'latent':digest(latent),'qmap':digest(qmap)})
    assert stream_file.tell()==len(native)
    del im,pm
    torch.cuda.empty_cache()
    return states


def main(gpu=6):
    ctx=task.initialize(gpu)
    a=decode_state('A',ctx);identity=decode_state('IDENTITY',ctx)
    mismatches=[i for i,(x,y) in enumerate(zip(a,identity)) if x!=y]
    result={'status':'PASS' if not mismatches else 'FAIL','frames':64,
            'mismatched_frames':mismatches,'reference_sha_equal_all':all(x['reference']==y['reference'] for x,y in zip(a,identity)),
            'P_latent_sha_equal_all':all(x['latent']==y['latent'] for x,y in zip(a[1:],identity[1:])),
            'P_qmap_sha_equal_all':all(x['qmap']==y['qmap'] for x,y in zip(a[1:],identity[1:]))}
    (P/'reference_identity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='PASS'
    print('IDENTITY_REFERENCE_PASS')


if __name__=='__main__':main()
