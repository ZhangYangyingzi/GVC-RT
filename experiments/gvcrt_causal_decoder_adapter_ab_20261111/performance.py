"""Same-GPU/precision/input deployment timing only on a GPU without foreign users."""
from common_s2 import *
import statistics,gc


def foreign_processes():
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid=next(line.split(',')[1].strip() for line in uuids.splitlines() if int(line.split(',')[0])==gpu)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    out=[]
    for line in raw.splitlines():
        if line.split(',')[0].strip()!=uuid:continue
        pid=int(line.split(',')[1]);
        if pid!=os.getpid():out.append(dict(PID=pid,process=','.join(line.split(',')[2:]).strip()))
    return out


def main():
    env();import torch
    from experiment_model import construct,reconstruct,restore
    from cache_codec import cache_paths
    from model_context import assert_reset
    from deployment_check import packets,decode
    foreign=foreign_processes();method='same physical GPU/precision/canonical RANS packets; 8 warmup frames, 3 passes of P frames9..16; synchronized wall clock. Both full RANS+codec+reconstruction decode and reconstruction-only times; excludes file IO, GT and scoring.'
    if foreign:
        dump(ROOT/'performance.json',dict(status='PENDING_UNCONTENDED_GPU',parameters=21696,physical_GPU=physical_gpu(0),
            foreign_GPU_processes=foreign,decode_timing_completed=False,method=method,
            decision='No timing taken while this GPU has another compute process. Task does not terminate foreign processes.',
            resume='CUDA_VISIBLE_DEVICES=<idle physical GPU0..3> python -B performance.py',time=time.time()));return
    v=load(ROOT/'manifests/media_cases.json')['cases'][0]['video'];row=v.get('source_row',v)
    _,p,rp=cache_paths(row,5,False);assert p.exists() and sha(p)==load(rp)['cache_sha256']
    d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);inputs=[None]+[x.to('cuda:0') for x in d['features'][1:17]];results=[]
    for model in ['B300',*GROUPS]:
        gc.collect();torch.cuda.empty_cache()
        c=construct(model,False,0,with_adapter=model in GROUPS)
        if model in GROUPS:restore(c,torch.load(ROOT/model/'training/checkpoints/step_0100.pt',map_location='cpu',weights_only=True))
        passes=[];full_passes=[];torch.cuda.reset_peak_memory_stats();items=packets(c,d['info'])
        with torch.no_grad():
            for repeat in range(3):
                assert_reset(c)
                for i in range(1,9):reconstruct(c,inputs[i],d['info']['actual_qps'][i])
                torch.cuda.synchronize();start=time.perf_counter()
                for i in range(9,17):reconstruct(c,inputs[i],d['info']['actual_qps'][i])
                torch.cuda.synchronize();passes.append((time.perf_counter()-start)/8)
            for repeat in range(3):
                assert_reset(c)
                for i in range(9):decode(c,items[i],i)
                torch.cuda.synchronize();start=time.perf_counter()
                for i in range(9,17):decode(c,items[i],i)
                torch.cuda.synchronize();full_passes.append((time.perf_counter()-start)/8)
                assert tensor_sha(c['pm'].dpb[0].feature)==d['info']['codec_trace'][16]['codec_feature_SHA256']
        assert not foreign_processes(),'GPU_CONTENTION_APPEARED_DURING_BENCHMARK'
        results.append(dict(model=model,reconstruction_seconds_per_P_frame=statistics.median(passes),raw_reconstruction_passes=passes,
            full_decode_seconds_per_P_frame=statistics.median(full_passes),raw_full_decode_passes=full_passes,
            history_bytes=c['adapter'].history_bytes() if model in GROUPS else 0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del c;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'performance.json',dict(status='PASS',parameters=21696,physical_GPU=physical_gpu(0),method=method,
        results=results,precision=DEPLOYMENT,input_stream_SHA256=d['info']['stream_SHA256'],no_foreign_compute_processes=True,time=time.time()))

if __name__=='__main__':main()
