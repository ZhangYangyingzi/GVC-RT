"""Optional independent clip4 export; not part of the six-panel main montage."""
import argparse
import io
import os
from diag_utils import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--sample',type=int,choices=range(5),required=True);p.add_argument('--qp',type=int,choices=QPS,required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(old.V41));import metric_runtime as metrics
    from media import writer,finish,probe,panel
    from stream_audit import structure
    eco,core=metrics.eco,metrics.core;s=samples()[args.sample];q=args.qp;m='clip4_control';r=point(s,m,q);checkpoint=cp(m)
    assert sha(checkpoint)==r['checkpoint_sha256'] and sha(r['bitstream_path'])==r['bitstream_sha256']
    frames=structure(Path(r['bitstream_path']),q,count(s));device=torch.device('cuda:0');joint=eco.load_joint(checkpoint,device)
    i_model,p_model=eco.build_models(device,joint[0]);p_model.clear_dpb();p_model.set_curr_poc(0)
    data=Path(r['bitstream_path']).read_bytes();buf=io.BytesIO(data);helper=eco.SPSHelper();hashes=[]
    output=ROOT/'optional_clip4'/s['dataset']/s['name']/f'qp_{q}_clip4.mp4'
    proc,cmd=writer(output,1920,1140,s['video']['fps'])
    rawpath_=output.with_suffix('.npy');raw=np.lib.format.open_memmap(rawpath_,mode='w+',dtype=np.uint8,shape=(count(s),1080,1920,3))
    with torch.inference_mode():
        for index in range(count(s)):
            h=eco.read_header(buf)
            while h['nal_type']==eco.NalType.NAL_SPS:
                helper.add_sps_by_id(eco.read_sps_remaining(buf,h['sps_id']));h=eco.read_header(buf)
            sps=helper.get_sps_by_id(h['sps_id']);qp,bits=eco.read_ip_remaining(buf);assert qp==frames[index]['actual_qp']
            if index==0:
                out=i_model.decompress(bits,sps,qp);p_model.clear_dpb();p_model.add_ref_frame(None,out['x_hat'])
            else:out=p_model.decompress(bits,sps,qp)
            hashes.append(eco.tensor_sha(out['x_hat']))
            value=core.unit(out['x_hat'],1080,1920)[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
            raw[index]=value
            proc.stdin.write(panel(value,f'clip4_control reconstruction\nQP={q} | {float(r["kbps"]):.1f} kbps',1920).tobytes())
            if index in keyframes(s):Image.fromarray(value,'RGB').save(output.with_name(f'qp_{q}_frame_{index:03d}.png'))
    raw.flush();finish(proc)
    assert buf.tell()==len(data) and eco.sha256_bytes(''.join(hashes).encode())==r['reconstruction_sha256']
    assert sha(checkpoint)==r['checkpoint_sha256']
    info=probe(output);assert int(info['nb_read_frames'])==count(s)
    dump(output.with_suffix('.audit.json'),dict(status='PASS',external_qp=q,method=m,video=str(output),video_command=cmd,
        probe=info,bitstream_sha256=r['bitstream_sha256'],checkpoint_sha256=r['checkpoint_sha256'],reencoded=False))
    print('OPTIONAL CLIP4 EXPORT PASS',output)

if __name__=='__main__':main()
