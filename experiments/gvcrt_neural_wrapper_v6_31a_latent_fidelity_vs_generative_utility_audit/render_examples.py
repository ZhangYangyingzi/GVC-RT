"""Decode the automatically selected pairs and show fixed frame32."""
import gc
import io
import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
from audit_io import *


def main():
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from worker import key
    from proxy_runtime import backend,runtime,frames_for
    backend()
    from types import SimpleNamespace
    selected=list(csv.DictReader((ROOT/'figures/selected_counterexamples/selection.csv').open()))
    videos=load(ROOT/'dataset_manifest.json')['datasets']
    for index,record in enumerate(selected):
        for k,vv in record.items():
            try:record[k]=float(vv)
            except (ValueError,TypeError):pass
        pair=SimpleNamespace(**record)
        v=next(v for v in videos[pair.dataset] if v['name']==pair.video)
        pictures=[frames_for(v)[32]]
        for method,qp in [('Original',pair.original_QP),(pair.method,pair.method_QP)]:
            eng,rt=runtime(method,[]);ia,pa=rt.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0)
            t=dict(dataset=pair.dataset,video=v['video_index'],method=method,qp=int(qp))
            data=io.BytesIO((ROOT/'parts'/key(t)/'stream.bin').read_bytes());helper=eng.eco.SPSHelper()
            with torch.inference_mode():
                for i in range(33):
                    header=eng.eco.read_header(data)
                    while header['nal_type']==eng.eco.NalType.NAL_SPS:
                        helper.add_sps_by_id(eng.eco.read_sps_remaining(data,header['sps_id']));header=eng.eco.read_header(data)
                    sp=helper.get_sps_by_id(header['sps_id']);q,bits=eng.eco.read_ip_remaining(data)
                    dec=(ia if i==0 else pa).decompress(bits,sp,q)
                    if i==0:pa.add_ref_frame(None,dec['x_hat'])
                pictures.append(eng.core.unit(dec['x_hat'],1080,1920).cpu())
            rt.assert_frozen(ia,pa)
            del rt,eng,ia,pa,dec;gc.collect();torch.cuda.empty_cache()
        canvas=Image.new('RGB',(1440,365),'white');draw=ImageDraw.Draw(canvas)
        labels=['Source',f"Original | bpp={pair.original_bpp:.6g}",f"{pair.method} | bpp={pair.method_bpp:.6g}"]
        for j,x in enumerate(pictures):
            image=Image.fromarray((x[0].clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)).resize((480,270))
            canvas.paste(image,(480*j,25));draw.text((480*j+5,5),labels[j],fill='black')
            if j:
                prefix='original' if j==1 else 'method'
                text=' | '.join(f'{m}={getattr(pair,prefix+"_P_"+m):.6g}' for m in ('cosine','LPIPS','DISTS','FloLPIPS'))
                draw.text((480*j+5,300),text,fill='black')
        draw.text((5,340),f'{pair.dataset} / {pair.video} / frame32 | relative rate gap={pair.rate_gap:.4%} | proxy teacher',fill='black')
        canvas.save(ROOT/'figures/selected_counterexamples'/f'pair_{index:02d}.png')
    rec=load(ROOT/'counterexample_selection.json');rec['render_pending']=False;rec['rendered']=len(selected)
    dump('counterexample_selection.json',rec)


if __name__=='__main__':main()
