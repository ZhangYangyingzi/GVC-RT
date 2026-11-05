"""Predeclared QP5 frames only; lossless PNG previews, never metric inputs."""
from common_s2 import *
from model_context import build,backend,assert_reset
from evaluate import videos
from reuse import read_csv
import io

def main():
    import numpy as np
    from PIL import Image
    torch,*_=backend(False,0)
    from src.utils import stream_helper as sh
    selection=load(ROOT/'selection.json');step=selection['best_trained_router']['step'];settings=load(ROOT/'protocol.json')['diagnostics']
    for split,source,label in [('development','development_c640d16978902780663b','B_expanded_ulong_add1500'),('final','final_d2b5efc4d5756f3604d7','B_common')]:
        seqs=settings['development_sequences'] if split=='development' else settings['final_sequences']
        for v in videos(split):
            if v['sequence'] not in seqs:continue
            key=f'{label}_{v["domain"]}_{v["sequence"]}_qp5';old=load(HIST/f'evaluation_v3/{source}/points/{key}.json')
            path=HIST/f'large_artifacts/evaluation_v3/{source}/bitstreams/{key}.bin';assert sha(path)==old['result']['bitstream_sha256']
            frames=read_csv(HIST/f'evaluation_v3/{source}/frames/{key}.csv')
            c=build(0,train=False,mode='bypass',losses=False);assert_reset(c);reader=io.BytesIO(path.read_bytes());helper=sh.SPSHelper()
            dest=ROOT/f'diagnostics/{split}_router_{step:04d}/{v["domain"]}_{v["sequence"]}_qp5';dest.mkdir(parents=True,exist_ok=True)
            with torch.inference_mode():
                for i in range(64):
                    header=sh.read_header(reader)
                    while header['nal_type']==sh.NalType.NAL_SPS:
                        helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
                    sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
                    if i==0:
                        dec=c['im'].decompress(payload,sps,q);assert_reset(c);c['pm'].add_ref_frame(None,dec['x_hat'])
                    else:dec=c['pm'].decompress(payload,sps,q)
                    assert tensor_sha(dec['x_hat'])==frames[i]['reconstruction_tensor_sha256']
                    if i in settings['P_frames']:
                        out=c['core'].unit(dec['x_hat'],v['height'],v['width'])
                        a=(out[0].permute(1,2,0).cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
                        Image.fromarray(a).save(dest/f'{i:02d}_B1500.png')
                        gt=np.asarray(Image.open(dest/f'{i:02d}_GT.png'));router=np.asarray(Image.open(dest/f'{i:02d}_Router.png'))
                        Image.fromarray(np.concatenate([gt,a,router],axis=1)).save(dest/f'{i:02d}_GT_B1500_Router.png')
            assert reader.tell()==path.stat().st_size;assert_reset(c);del c;torch.cuda.empty_cache()
    dump(ROOT/'audits/diagnostics.json',dict(status='PASS',predeclared_sequences=settings['development_sequences']+settings['final_sequences'],frames=settings['P_frames'],QP=5,
        order='GT | B+1500 | best trained Router',PNG='8-bit RGB lossless container after rounding; viewing only; neural metrics use original unquantized decoded tensors'))
if __name__=='__main__':env();main()
