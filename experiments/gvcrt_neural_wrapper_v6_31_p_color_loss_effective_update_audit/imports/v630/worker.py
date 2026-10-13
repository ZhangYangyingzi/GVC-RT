"""Read-only old-stream closure and fixed new-stream evaluation."""
from audit_io import *
import argparse, fcntl, io, resource, traceback
def official_model(torch):
    sys.path.insert(0,str(OLD/'vendor/pytorch-fid/src'))
    import pytorch_fid.inception as src
    assert sha(WEIGHT)==WEIGHT_SHA
    src.load_state_dict_from_url=lambda *a,**kw:torch.load(WEIGHT,map_location='cpu',weights_only=True)
    net=src.InceptionV3([3],resize_input=True,normalize_input=True,requires_grad=False,use_fid_inception=True).to('cuda').eval().requires_grad_(False)
    class Flat(torch.nn.Module):
        def __init__(self):super().__init__();self.net=net
        def forward(self,x):return self.net(x.float())[0].flatten(1)
    return Flat().eval().requires_grad_(False)
def gt_path(d,i):return ROOT/'features/GT'/d/f'video_{i:02d}.npz'
def features_path(t):return ROOT/'features'/t['kind']/t['dataset']/t['method']/f'video_{t["video"]:02d}_qp{t["qp"]}.npz'
def main(t,gpu):
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch,numpy as np
    torch.set_num_threads(2)
    sys.path.insert(0,str(REPO))
    from src.utils.common import set_torch_env
    set_torch_env()  # Same deterministic backend setup called by historical codec load_models.
    from adapter import engine,frames_for
    from io21 import tensor_hash
    lock=ROOT/'tasks'/(task_id(t)+'.lock');lock.parent.mkdir(exist_ok=True)
    lf=lock.open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if valid(t):return
    if output(t).exists():
        previous=load(output(t));dump(ROOT/'audits/invalidated_receipts'/f'{task_id(t)}_{time.time_ns()}.json',dict(previous=previous,reason='input or code signature changed; only this task is recomputed'))
    for p,h in load(ROOT/'source_hashes.json')['hashes'].items():
        # Actual checkpoints and code must match, including independent deployment SHA.
        if Path(p).suffix=='.py':assert sha(p)==h,('frozen dependency changed',p)
    if t['method'] in METHODS[1:]:
        cp=load(ROOT/'evaluation/config.json')['checkpoints'][t['method']]
        assert sha(cp['path'])==cp['sha256'],('deployment checkpoint changed',t['method'])
    v=next(x for x in sources(t['dataset'],t['kind']=='new') if x['video_index']==t['video'])
    frames=frames_for(v);assert len(frames)==64
    torch.cuda.reset_peak_memory_stats();start=time.time();official=official_model(torch)
    official_before=tensor_hash(official.state_dict());artifacts={};records=[]
    def commit(status,**data):
        assert tensor_hash(official.state_dict())==official_before
        r=dict(task=t,signature=signature(t),execution_complete=True,status=status,artifacts=artifacts,
               physical_GPU=gpu,seconds=time.time()-start,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,
               peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               official_parameter_buffer_hash_before=official_before,official_parameter_buffer_hash_after=tensor_hash(official.state_dict()),
               precision='FP32 official FID, float RGB no PNG reconstruction',
               backend=dict(setup_source=str(REPO/'src/utils/common.py'),setup_sha256=sha(REPO/'src/utils/common.py'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),**data)
        dump(output(t),r);print('DONE',task_id(t),status,r['peak_memory_MiB'],flush=True)
    if t['kind']=='gt':
        real=[];rows=[]
        with torch.inference_mode():
            for i,cpu in enumerate(frames):
                x=cpu.cuda().float();z=official(x).cpu().numpy();real.append(z)
                import torch.nn.functional as F
                rows.append(dict(frame=i,source_frame_index=v['source_frame_indices'][i],UID=f'{v["name"]}:{v["source_frame_indices"][i]}',canonical_RGB_sha256=v['frame_rgb_sha256'][i],float_RGB_stage_sha256=tensor_hash({'RGB':x}),FID_resized_normalized_stage_sha256=tensor_hash({'RGB':F.interpolate(x,size=(299,299),mode='bilinear',align_corners=False)*2-1}),feature_row_sha256=hashlib.sha256(z.tobytes()).hexdigest()))
        real=np.concatenate(real);assert real.shape==(64,2048) and np.isfinite(real).all()
        sealed=OLD.parent/('gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit' if t['dataset']=='mcljcv' else 'gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation')/'features'/f'GT_{t["dataset"]}.npy'
        oldvs=sources(t['dataset']);offset=next(i for i,x in enumerate(oldvs) if x['video_index']==t['video'])*64
        pool=np.load(sealed,mmap_mode='r');assert pool.shape==(len(oldvs)*64,2048)
        checked=compare_features(real,np.array(pool[offset:offset+64]));
        for r,c in zip(rows,checked):r.update(c,kind='GT_vs_sealed_pool')
        p=gt_path(t['dataset'],t['video']);save_npz(p,real=real);artifacts[str(p)]=sha(p)
        p=ROOT/'audits/GT'/(task_id(t)+'.csv');write(p,rows);artifacts[str(p)]=sha(p)
        commit('PASS' if all(x['PASS'] for x in checked) else 'FAIL',records=rows,sealed_GT_path=str(sealed),sealed_GT_sha256=sha(sealed),shape=list(real.shape),dtype=str(real.dtype),finite=True)
        return
    eng=engine();cfg=load(ROOT/'evaluation/config.json');m=t['method'];q=t['qp']
    runtime=eng.Runtime(dict(experiment='B',IP=-1,methods={m:[None,None] if m=='original' else [m,m]},checkpoints={} if m=='original' else {m:cfg['checkpoints'][m]},force_zero_thres=.12),m,torch.device('cuda:0'))
    wrapper_before=None if runtime.wrapper is None else eng.core.module_hash(runtime.wrapper)
    metrics_before=[tensor_hash(x.state_dict()) for x in (*runtime.quality,*runtime.metric_models)]
    if t['kind']=='new':
        # Only the FID extractor is replaced by the independently audited official implementation.
        # LPIPS, DISTS, flow and true RANS adapter are unchanged.
        runtime.metric_models=(*runtime.metric_models[:2],official)
        actual_metrics_before=[tensor_hash(x.state_dict()) for x in (*runtime.quality,*runtime.metric_models)]
        b=ROOT/'bitstreams/new'/t['dataset']/m/f'video_{t["video"]:02d}_qp{q}.bin';f=features_path(t)
        result,rows=runtime.run(frames,q,b,f)
        trans=f.with_suffix('.transitions.csv');assert len(list(csv.DictReader(trans.open())))==63
        from fid_statistics import lowrank_unclamped
        with np.load(f) as z:fid=lowrank_unclamped(z['real'],z['reconstruction'])
        fps=v['rate_accounting_fps'];result.update(dataset=t['dataset'],video_index=t['video'],sequence=v['name'],method=m,QP=q,source_frame_indices=v['source_frame_indices'],source_rgb_sha256=v['rgb_sha256'],FID=fid,native_fps=fps,kbps=result['bits_per_frame']*fps/1000,kbps_20fps=result['bits_per_frame']*20/1000,rate_accounting_fps=fps,actual_qps=[x['actual_qp'] for x in rows],checkpoint_sha256='' if m=='original' else cfg['checkpoints'][m]['sha256'],official_FID_parameter_hash=official_before)
        assert result['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
        assert result['frames']==64 and result['bytes_consumed']==b.stat().st_size and result['independent_decode_pass'] and result['state_sync_pass']
        assert actual_metrics_before==[tensor_hash(x.state_dict()) for x in (*runtime.quality,*runtime.metric_models)]
        p=ROOT/'new_clip_metrics'/t['dataset']/m/f'video_{t["video"]:02d}_qp{q}.json';dump(p,result)
        ff=p.with_suffix('.frames.csv');write(ff,rows)
        for p0 in (b,f,trans,p,ff):artifacts[str(p0)]=sha(p0)
        commit('PASS',result=result,model_hashes_before=actual_metrics_before,model_hashes_after=actual_metrics_before,wrapper_hash=wrapper_before,metric_FID_only_override='official pytorch-fid FP32 with same frozen weights and preprocessing')
        return
    oldr=load(point(t['dataset'],m,t['video'],q));b=Path(oldr['bitstream_path']);cached_path=Path(oldr['feature_path'])
    assert sha(b)==oldr['bitstream_sha256'] and sha(cached_path)==oldr['feature_sha256']
    with np.load(cached_path) as z:cached_real=z['real'].copy();cached_rec=z['reconstruction'].copy()
    with np.load(gt_path(t['dataset'],t['video'])) as z:real=z['real'].copy()
    for c in compare_features(real,cached_real):records.append(dict(**c,kind='GT_vs_historical_real'))
    h,w=1080,1920;ia,pa=runtime.models(1088,1920)
    model_before=[tensor_hash(x.state_dict()) for x in (ia,pa)]
    pa.clear_dpb();pa.set_curr_poc(0);stream=io.BytesIO(b.read_bytes());helper=eng.eco.SPSHelper();rec=[];raw_hashes=[];rgbhash=[];actuals=[]
    with torch.inference_mode():
        for i in range(64):
            header=eng.eco.read_header(stream)
            while header['nal_type']==eng.eco.NalType.NAL_SPS:
                helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
            sps=helper.get_sps_by_id(header['sps_id']);actual,bits=eng.eco.read_ip_remaining(stream)
            assert (header['nal_type']==eng.eco.NalType.NAL_I)==(i==0)
            assert sps['height']==1088 and sps['width']==1920 and actual==oldr['actual_qps'][i];actuals.append(actual)
            if i==0:
                dec=ia.decompress(bits,sps,actual);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,dec['x_hat'])
            else:dec=pa.decompress(bits,sps,actual)
            raw_hashes.append(eng.eco.tensor_sha(dec['x_hat']))
            x=eng.core.unit(dec['x_hat'],h,w).float();assert x.shape==(1,3,1080,1920) and torch.isfinite(x).all()
            import torch.nn.functional as F
            rgbhash.append(dict(raw_decoder_tensor_stage_sha256=raw_hashes[-1],float_RGB_stage_sha256=tensor_hash({'RGB':x}),FID_resized_normalized_stage_sha256=tensor_hash({'RGB':F.interpolate(x,size=(299,299),mode='bilinear',align_corners=False)*2-1})))
            rec.append(official(x).cpu().numpy())
    assert stream.tell()==b.stat().st_size
    reconstruction_hash=eng.eco.sha256_bytes(''.join(raw_hashes).encode())
    hash_consistent=reconstruction_hash==oldr['reconstruction_sha256']
    reconstruction=np.concatenate(rec)
    for c in compare_features(reconstruction,cached_rec):records.append(dict(**c,kind='reconstruction_vs_historical',**rgbhash[c['frame']]))
    for c in records:c.update(dataset=t['dataset'],method=m,QP=q,sequence=v['name'],source_frame_index=v['source_frame_indices'][c['frame']],UID=f'{v["name"]}:{v["source_frame_indices"][c["frame"]]}',cached_dtype=str(cached_rec.dtype),fresh_dtype=str(reconstruction.dtype),finite=True,shape=[2048])
    runtime.assert_frozen(ia,pa);assert model_before==[tensor_hash(x.state_dict()) for x in (ia,pa)]
    assert metrics_before==[tensor_hash(x.state_dict()) for x in (*runtime.quality,*runtime.metric_models)]
    assert wrapper_before==(None if runtime.wrapper is None else eng.core.module_hash(runtime.wrapper))
    f=features_path(t);save_npz(f,real=real,reconstruction=reconstruction);artifacts[str(f)]=sha(f)
    p=ROOT/'audits/closure_frames'/(task_id(t)+'.csv');write(p,records);artifacts[str(p)]=sha(p)
    commit('PASS' if hash_consistent and all(x['PASS'] for x in records) else 'FAIL',records=records,input_record_sha256=sha(point(t['dataset'],m,t['video'],q)),cached_feature_sha256=sha(cached_path),bitstream_sha256=sha(b),bytes_consumed=stream.tell(),actual_qps=actuals,reconstruction_hash_match=hash_consistent,reconstruction_sha256=reconstruction_hash,model_hashes_before=model_before,model_hashes_after=model_before,metric_hashes_before=metrics_before,metric_hashes_after=metrics_before,wrapper_hash=wrapper_before,fresh_feature_sha256=sha(f),reason='NONE' if hash_consistent and all(x['PASS'] for x in records) else 'UNKNOWN; recorded per-frame evidence without altering history')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    try:main(json.loads(a.task),a.gpu)
    except Exception:
        append_error(dict(worker=os.getpid(),task=json.loads(a.task),traceback=traceback.format_exc()));raise
