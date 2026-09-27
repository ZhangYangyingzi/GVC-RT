import argparse
import gc
import io
import os
from diag_utils import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--sample',type=int,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import numpy as np
    import cv2
    sys.path.insert(0,str(old.V41));import metric_runtime as metrics
    from sources import ensure_source
    from media import render
    from stream_audit import structure
    eco,core=metrics.eco,metrics.core;device=torch.device('cuda:0')
    s=samples()[args.sample];n=count(s);qs=(0,3) if args.smoke else QPS
    assert not args.smoke or s==samples()[0] and s['dataset']=='uvg'
    frames=ensure_source(s,torch,eco,args.smoke)
    rows=[];audits=[]
    class Stats:
        def __init__(self):
            self.total=np.zeros(3);self.square=np.zeros(3);self.error=np.zeros(3);self.absolute=np.zeros(3)
            self.saturation=0.;self.value=0.;self.pixels=0
        def update(self,x,y):
            a=x[0].permute(1,2,0).float().cpu().numpy();b=y[0].permute(1,2,0).float().cpu().numpy()
            assert a.shape==b.shape==(1080,1920,3) and np.isfinite(a).all()
            flat=a.reshape(-1,3);delta=a-b
            self.total+=flat.sum(0,dtype=np.float64);self.square+=np.square(flat,dtype=np.float64).sum(0)
            self.error+=delta.sum((0,1),dtype=np.float64);self.absolute+=np.abs(delta).sum((0,1),dtype=np.float64)
            hsv=cv2.cvtColor(np.clip(a,0,1),cv2.COLOR_RGB2HSV)
            self.saturation+=hsv[:,:,1].sum(dtype=np.float64);self.value+=hsv[:,:,2].sum(dtype=np.float64);self.pixels+=len(flat)
        def result(self,name,q):
            mean=self.total/self.pixels
            return dict(dataset=s['dataset'],video_index=s['video_index'],name=s['name'],external_qp=q,method=name,
                source_conversion='BT709_EXPLICIT' if args.smoke else 'CURRENT',
                **{f'mean_{c}':float(mean[i]) for i,c in enumerate('RGB')},
                **{f'std_{c}':float(np.sqrt(max(0,self.square[i]/self.pixels-mean[i]**2))) for i,c in enumerate('RGB')},
                **{f'signed_error_{c}':float(self.error[i]/self.pixels) for i,c in enumerate('RGB')},
                **{f'MAE_{c}':float(self.absolute[i]/self.pixels) for i,c in enumerate('RGB')},
                RGB_global_mean_bias=float(self.error.mean()/self.pixels),mean_saturation=float(self.saturation/self.pixels),
                mean_value=float(self.value/self.pixels),num_frames=n,statistics_units='original float RGB [0,1], population std; OpenCV float HSV S/V [0,1]')
    source_stats=Stats()
    for f in frames:source_stats.update(f,f)
    rows.extend(source_stats.result('Source',q) for q in qs)
    quality=core.quality_models(device) if args.smoke else None
    models=metrics.load_metrics(device) if args.smoke else None
    for m in METHODS:
        checkpoint=cp(m);digest=sha(checkpoint) if checkpoint else ''
        assert digest==old.load('config.json')['checkpoints'][m]['sha256']
        joint=eco.load_joint(checkpoint,device) if checkpoint else None
        if joint:
            outpath=rawpath(s,m+'_proxy',smoke=args.smoke);outpath.parent.mkdir(parents=True,exist_ok=True)
            raw=np.lib.format.open_memmap(outpath,mode='w+',dtype=np.uint8,shape=(n,1080,1920,3));stats=Stats()
            with torch.inference_mode():
                for i,f in enumerate(frames):
                    proxy=joint[1](f.to(device));stats.update(proxy,f)
                    raw[i]=proxy[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
            raw.flush();del raw
            rows.extend(stats.result(m+'_proxy',q) for q in qs)
        for q in qs:
            if args.smoke:
                r,fr=eco.run_stream(frames,q,joint,device,float(s['video']['fps']),
                    ROOT/'bt709_smoke/bitstreams'/f'{m}_qp{q}.bin',quality=quality)
                r.update(metrics.measure(frames,r['bitstream_path'],joint,device,models,ROOT/'bt709_smoke/features'/f'{m}_qp{q}.npz',r))
                r.update(method=m,external_qp=q,checkpoint=str(checkpoint) if checkpoint else '',checkpoint_sha256=digest,
                    num_frames=n,real_bytes=r['bytes'],source_conversion='BT709_EXPLICIT')
                dump(f'bt709_smoke/records/{m}_qp{q}.json',r)
            else:r=point(s,m,q)
            assert r['checkpoint_sha256']==digest and int(r['external_qp'])==q and int(r['num_frames'])==n
            path=Path(r['bitstream_path']);assert sha(path)==r['bitstream_sha256']
            header_rows=structure(path,q,n);assert int(r['real_bytes'])==path.stat().st_size
            i_model,p_model=eco.build_models(device,None if joint is None else joint[0])
            p_model.clear_dpb();p_model.set_curr_poc(0);data=path.read_bytes();buf=io.BytesIO(data);helper=eco.SPSHelper()
            outpath=rawpath(s,m,q,args.smoke);outpath.parent.mkdir(parents=True,exist_ok=True)
            raw=np.lib.format.open_memmap(outpath,mode='w+',dtype=np.uint8,shape=(n,1080,1920,3));hashes=[];stats=Stats()
            with torch.inference_mode():
                for i,f in enumerate(frames):
                    h=eco.read_header(buf)
                    while h['nal_type']==eco.NalType.NAL_SPS:
                        helper.add_sps_by_id(eco.read_sps_remaining(buf,h['sps_id']));h=eco.read_header(buf)
                    sps=helper.get_sps_by_id(h['sps_id']);aq,bits=eco.read_ip_remaining(buf);assert aq==header_rows[i]['actual_qp']
                    if h['nal_type']==eco.NalType.NAL_I:
                        decoded=i_model.decompress(bits,sps,aq);p_model.clear_dpb();p_model.add_ref_frame(None,decoded['x_hat'])
                    else:decoded=p_model.decompress(bits,sps,aq)
                    hashes.append(eco.tensor_sha(decoded['x_hat']));output=core.unit(decoded['x_hat'],1080,1920)
                    stats.update(output,f);raw[i]=output[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
            raw.flush();del raw
            actual=eco.sha256_bytes(''.join(hashes).encode())
            assert actual==r['reconstruction_sha256'] and buf.tell()==len(data)
            assert core.compression_hash(i_model,p_model)==r['compression_hash_after']
            rows.append(stats.result(m+'_reconstruction',q))
            framefile=ROOT/'parts'/f'headers_{tag(s)}_{m}_qp{q}_{"709" if args.smoke else "current"}.csv'
            write(framefile,header_rows)
            audits.append(dict(dataset=s['dataset'],video_index=s['video_index'],method=m,external_qp=q,
                status='PASS',checkpoint_sha256=digest,bitstream_path=str(path),bitstream_sha256=sha(path),
                expected_reconstruction_sha256=r['reconstruction_sha256'],decoded_reconstruction_sha256=actual,
                frame_count=n,frame_QP_csv=str(framefile),reencoded=args.smoke,raw_RGB_path=str(outpath),
                raw_RGB_sha256=sha(outpath),visualization_quantization='round(clamp(RGB,0,1)*255); hashes and statistics computed before visualization quantization',
                physical_gpu=args.gpu))
            del i_model,p_model;gc.collect();torch.cuda.empty_cache()
            print('DECODE PASS',tag(s),m,q,'BT709' if args.smoke else 'CURRENT',flush=True)
        del joint;gc.collect();torch.cuda.empty_cache()
    suffix='709' if args.smoke else 'current'
    write(f'parts/color_{tag(s)}_{suffix}.csv',rows)
    dump(f'parts/reuse_{tag(s)}_{suffix}.json',dict(status='PASS',records=audits))
    del models,quality;gc.collect();torch.cuda.empty_cache()
    exported=[render(s,q,args.smoke) for q in qs]
    dump(f'parts/comparison_{tag(s)}_{suffix}.json',dict(status='PASS',videos=exported))
    print('WORKER PASS',tag(s),suffix,flush=True)

if __name__=='__main__':main()
