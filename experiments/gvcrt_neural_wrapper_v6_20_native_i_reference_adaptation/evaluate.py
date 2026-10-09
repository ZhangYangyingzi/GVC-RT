"""Resumable whole-sequence real encode and independent decode."""
import argparse,fcntl,math,traceback
from io20 import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--dataset',choices=DATASETS,required=True);ap.add_argument('--method',choices=EVAL_METHODS,required=True);ap.add_argument('--video',type=int,required=True);ap.add_argument('--qps',default='0,1,2,3,4,5,6,7,8,9');a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np
    from adapter import frames_for,engine,validate,cache_key
    from prepare import selected
    torch.set_num_threads(2);cfg=frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    (ROOT/'parts').mkdir(exist_ok=True);lock=(ROOT/'parts'/f'{a.dataset}_{a.method}_{a.video}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v=next(v for v in sources(a.dataset) if v['video_index']==a.video);pending=[]
    for q in map(int,a.qps.split(',')):
        p=point(a.dataset,a.method,a.video,q)
        if p.exists():validate(load(p),v,q,a.method)
        else:pending.append(q)
    if not pending:return
    eng=engine();cp={} if a.method=='original' else {a.method:cfg['checkpoints'][a.method]};runtime=eng.Runtime(dict(experiment='B',methods={a.method:[None,None] if a.method=='original' else [a.method,a.method]},checkpoints=cp,force_zero_thres=.12),a.method,torch.device('cuda:0'));frames=frames_for(v);torch.cuda.reset_peak_memory_stats()
    for q in pending:
        t=time.time();relative=Path(a.dataset)/a.method/v['name']/f'qp{q}';stream=(ROOT/'bitstreams'/relative).with_suffix('.bin');feature=(ROOT/'features'/relative).with_suffix('.npz');runtime.capture_first=selected(v) and q in (0,4,9)
        r,rows=runtime.run(frames,q,stream,feature);out=point(a.dataset,a.method,a.video,q);ff=out.with_suffix('.frames.csv');write(ff,rows)
        with np.load(feature) as f:fid=float(eng.metrics.fid.low_rank_fid(f['real'],f['reconstruction']));assert math.isfinite(fid)
        trans=feature.with_suffix('.transitions.csv');r.update(dataset=a.dataset,method=a.method,video_index=a.video,sequence=v['name'],external_qp=q,QP=q,checkpoint_sha256='' if a.method=='original' else cp[a.method]['sha256'],checkpoint_path='' if a.method=='original' else cp[a.method]['path'],source_sha256=v['source_sha256'],source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],source_frame_indices_sha256=hashlib.sha256(json.dumps(v['source_frame_indices'],separators=(',',':')).encode()).hexdigest(),rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,total_bits=r['real_bytes']*8,FID=fid,FID_num_samples=64,force_zero_thres=.12,frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),transitions_path=str(trans),transitions_sha256=sha(trans),actual_qps=[x['actual_qp'] for x in rows],physical_gpu=a.gpu,fresh_run=True,temporal_resampling=False,elapsed_seconds=time.time()-t,config_sha256=sha(ROOT/'config.json'),evaluator_sha256=sha(BASE/'evaluate.py'),visualization_dir='',visualization_sha256=[],cache_key=cache_key(v,q,a.method),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20)
        if a.method.endswith('_0') or a.method in METHODS[:3]:
            oldm='B_i_bypass' if a.method=='native_i_adapt_0' else 'B_standard' if a.method=='wrapped_i_control_0' else a.method
            baseline=load(V19/'parts'/a.dataset/oldm/out.name)
            for k in ('real_bytes','bitstream_sha256','reconstruction_sha256','actual_qps','LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM'):assert r[k]==baseline[k],('standard reproduction mismatch',k,r[k],baseline[k])
            r['historical_reproduction_verified']=True
        if a.method.endswith('_0'):
            ref_method='B_i_bypass' if a.method=='native_i_adapt_0' else 'B_standard'
            ref_runtime=eng.Runtime(dict(experiment='B',methods={ref_method:[ref_method,ref_method]},checkpoints={ref_method:cfg['checkpoints'][ref_method]},force_zero_thres=.12),ref_method,torch.device('cuda:0'))
            rs,rrows=ref_runtime.run(frames,q,ROOT/'verification'/a.dataset/ref_method/v['name']/f'qp{q}.bin',ROOT/'features/verification'/a.dataset/ref_method/v['name']/f'qp{q}.npz')
            for k in ('real_bytes','bitstream_sha256','reconstruction_sha256','first_frame_audit','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'):
                if k=='first_frame_audit':
                    for key in ('payload_sha256','decoded_tensor_sha256','encoder_initial_reference_sha256','decoder_initial_reference_sha256'):assert r[k][key]==rs[k][key]
                else:assert r[k]==rs[k],('same GPU step0 mismatch',k)
            r['same_GPU_step0_verification']=dict(status='PASS',physical_gpu=a.gpu,baseline=ref_method,baseline_bitstream_sha256=rs['bitstream_sha256'],baseline_reconstruction_sha256=rs['reconstruction_sha256']);del ref_runtime;torch.cuda.empty_cache()
        if cfg['i_frame_modes'][a.method]=='native_RGB_bypass':
            import io
            orig=load(point(a.dataset,'original',v['video_index'],q));buf=io.BytesIO(Path(orig['bitstream_path']).read_bytes());header=eng.eco.read_header(buf);assert header['nal_type']==eng.eco.NalType.NAL_SPS;sps=eng.eco.read_sps_remaining(buf,header['sps_id']);header=eng.eco.read_header(buf);assert header['nal_type']==eng.eco.NalType.NAL_I;iq,payload=eng.eco.read_ip_remaining(buf);assert iq==q and eng.eco.sha256_bytes(payload)==r['first_frame_audit']['payload_sha256']
            oi,op=eng.core.load_models(torch.device('cuda:0'),force_zero_thres=.12);oi.set_use_two_entropy_coders(bool(sps['ec_part']))
            with torch.inference_mode():dec=oi.decompress(payload,sps,iq);h=eng.eco.tensor_sha(dec['x_hat'])
            assert h==r['first_frame_audit']['decoded_tensor_sha256'];r['native_I_original_match']=dict(status='PASS',payload_sha256=eng.eco.sha256_bytes(payload),decoded_tensor_sha256=h,source_original_point=str(point(a.dataset,'original',v['video_index'],q)),same_GPU=a.gpu);del oi,op,dec;torch.cuda.empty_cache()
        validate(r,v,q,a.method);dump(out,r);print('DONE',a.dataset,a.method,a.video,q,'peak',r['peak_memory_MiB'],flush=True)
    frozen()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'failure_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
