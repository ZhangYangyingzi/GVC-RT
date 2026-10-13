"""Normalize historical I0 evidence; independently decode missing legacy I hashes."""
from io21 import *
import argparse,io

def key(d,i,m,q):return f'{d}|{i}|{m}|{q}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from adapter import engine,validate
    torch.set_num_threads(2);eng=engine();cfg=frozen();command([PYTHON,'-B',str(Path(__file__).resolve()),'--gpu',str(a.gpu)])
    runtime=None;ia=None;pa=None;records={};decoded=0
    for d in DATASETS:
        for v in sources(d):
            for q in range(10):
                identities=[]
                for m in METHODS:
                    r=load(point(d,m,v['video_index'],q));validate(r,v,q,m)
                    assert sha(r['bitstream_path'])==r['bitstream_sha256']
                    stream=io.BytesIO(Path(r['bitstream_path']).read_bytes());helper=eng.eco.SPSHelper();header=eng.eco.read_header(stream)
                    while header['nal_type']==eng.eco.NalType.NAL_SPS:
                        helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
                    assert header['nal_type']==eng.eco.NalType.NAL_I
                    sps=helper.get_sps_by_id(header['sps_id']);actual_qp,bits=eng.eco.read_ip_remaining(stream)
                    assert actual_qp==q
                    payload=eng.eco.sha256_bytes(bits)
                    if r.get('I_refresh_audits'):
                        first=r['I_refresh_audits'][0];assert first['frame']==0 and first['payload_sha256']==payload
                        hashes=[first[k] for k in ('native_I_sha256','encoder_reference_sha256','decoder_reference_sha256','independent_reference_sha256')]
                        assert len(set(hashes))==1;native=hashes[0];evidence='verified historical/new I_refresh_audits'
                    elif r.get('first_frame_audit'):
                        first=r['first_frame_audit'];assert first['payload_sha256']==payload
                        hashes=[first[k] for k in ('decoded_tensor_sha256','encoder_initial_reference_sha256','decoder_initial_reference_sha256','independent_decoded_tensor_sha256','independent_initial_reference_sha256')]
                        assert len(set(hashes))==1;native=hashes[0];evidence='verified legacy first_frame_audit'
                    else:
                        assert m=='original' and r.get('reused'),'Missing I evidence on a non-legacy result'
                        if runtime is None:
                            runtime=eng.Runtime(dict(experiment='B',IP=-1,methods={'original':[None,None]},checkpoints={},force_zero_thres=.12),'original',torch.device('cuda'))
                            ia,pa=runtime.models(1088,1920)
                        with torch.inference_mode():native=eng.eco.tensor_sha(ia.decompress(bits,sps,actual_qp)['x_hat'])
                        decoded+=1;evidence='fresh native-I-only independent decode of verified legacy bitstream; no re-encoding'
                    item=dict(dataset=d,video_index=v['video_index'],method=m,QP=q,actual_I_QP=actual_qp,
                        native_I_sha256=native,payload_sha256=payload,evidence=evidence,
                        bitstream_path=r['bitstream_path'],bitstream_sha256=r['bitstream_sha256'],
                        inherited_independent_full_decode=r['independent_decode_pass'],I_model_config_hash=cfg['compression_hash'])
                    records[key(d,v['video_index'],m,q)]=item;identities.append(item)
                assert len({x['payload_sha256'] for x in identities})==1
                assert len({x['native_I_sha256'] for x in identities})==1
            print('NATIVE I AUDIT',d,v['name'],flush=True)
    if runtime is not None:runtime.assert_frozen(ia,pa)
    assert len(records)==1840
    dump(ROOT/'audits/native_I_payload_reconstruction.json',dict(status='PASS',points=1840,records=records,
         legacy_missing_hashes_independently_decoded=decoded,historical_files_unchanged=True,no_reencoding=True,physical_GPU=a.gpu,
         native_I_payload_and_reconstruction_equal=True))
    print('NATIVE I AUDIT PASS',len(records),'legacy independent I decodes',decoded,flush=True)

if __name__=='__main__':main()
