"""Fixed first GT-selected development point, strict historical A2000 check."""
from common_s2 import *

def check(point,root):
    result=point['result'];seq=result['sequence'];q=result['QP'];key=f'ulong_new_{seq}_qp{q}'
    previous=GAIN_SOURCE/'evaluation/s05';old_path=previous/'points'/f'{key}.json';old=load(old_path)
    assert old['status']=='PASS' and old['lora_checkpoint_sha256']==INIT_SHA and old['inference_gain']==.5
    assert old['metric_hashes']==point['metric_network_hashes']
    for name,h in old['record_file_sha256'].items():assert sha(previous/name/f'{key}.csv')==h
    def rows(path):
        with path.open() as f:return list(csv.DictReader(f))
    a=rows(previous/'frames'/f'{key}.csv');b=rows(root/'frames'/f'{seq}_qp{q}.csv');assert len(a)==len(b)==64
    for x,y in zip(a,b):
        assert x['reconstruction_tensor_sha256']==y['reconstruction_SHA256'],'HISTORICAL_A2000_RECONSTRUCTION_MISMATCH'
        assert x['decoded_RGB_tensor_sha256']==y['RGB_tensor_SHA256'],'HISTORICAL_A2000_RGB_MISMATCH'
        assert int(x['actual_qp'])==int(y['actual_QP']) and x['source_frame_RGB_sha256']==y['source_RGB_SHA256']
    for name in ['bpp','real_bytes','bitstream_sha256']:
        newkey='bitstream_SHA256' if name=='bitstream_sha256' else name
        assert old['result'][name]==result[newkey],('HISTORICAL_STREAM_MISMATCH',name)
    errors={k:abs(old['result'][k]-result[k]) for k in ['PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC','kbps']}
    assert all(e<(1e-9 if k in ['PSNR','kbps'] else 1e-10) for k,e in errors.items()),('HISTORICAL_METRIC_MISMATCH',errors)
    dump(ROOT/'audits/historical_A2000_s05_native_endpoint.json',dict(status='PASS',sequence=seq,QP=q,source=str(old_path),source_SHA256=sha(old_path),
        reconstruction_and_RGB_all64_SHA256_exact=True,same_stream_bytes_SHA_bpp=True,metric_absolute_errors=errors,tolerance='unchanged inherited: PSNR/kbps<1e-9; other metrics<1e-10',fixed_by_GT_rule=True,time=time.time()))
