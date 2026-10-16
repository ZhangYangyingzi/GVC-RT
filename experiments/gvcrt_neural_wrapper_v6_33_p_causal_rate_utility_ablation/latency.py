"""Synchronized sender-only P overhead; identical receiver on both paths."""
import os
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='7')
from runtime import *
import time
import subprocess

def main(checkpoint=None,gpu=7):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    import torch,numpy as np
    from evaluate import write
    gpu_before=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    ctx=setup(BRANCHES[3]);v4,im,pm,models,quality=ctx
    cp=load(ROOT/'checkpoint_hashes.json')['source'] if checkpoint is None else dict(path=str(Path(checkpoint).resolve()),sha256=sha(checkpoint))
    s=torch.load(cp['path'],map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(s[k]);m.eval().requires_grad_(False)
    pm.recon_generation_net.half();del s,quality,ctx
    idec,pdec=v4.load_models(torch.device('cuda:0'),force_zero_thres=.12)
    s=torch.load(cp['path'],map_location='cpu',weights_only=True)
    for k,m in [('bridge',pdec.recon_generation_net.mlp),('generator',pdec.recon_generation_net.decoder)]:m.load_state_dict(s[k])
    del s
    v=load(V32/'eval_manifest.json')['ulong'][0]
    sys.path.insert(0,str(V32));from adapter32 import frames_for
    f=frames_for(v);x0=f[0].cuda();x=f[1].cuda();q=5
    si=dict(height=1088,width=1920,ec_part=1)
    for m in (im,pm,idec,pdec):m.set_use_two_entropy_coders(True)
    before=v4.compression_hash(im,pm)
    with torch.inference_mode():
        i=im.compress(v4.codec_input(x0),q);ref=idec.decompress(i['bit_stream'],si,q)['x_hat']
        qa=pm.shift_qp(q,0)
        def reset(m):m.clear_dpb();m.set_curr_poc(0);m.add_ref_frame(None,ref)
        rows=[];raw=[]
        for mode in ('No-P','With-P'):
            models['wrapper'].to('cpu' if mode=='No-P' else 'cuda:0')
            samples={k:[] for k in ('P_standalone','encode','decode','end_to_end')}
            torch.cuda.reset_peak_memory_stats()
            for it in range(120):
                reset(pm);reset(pdec);torch.cuda.synchronize();t=time.perf_counter()
                proxy=x if mode=='No-P' else models['wrapper'](x)
                torch.cuda.synchronize();tP=time.perf_counter()
                encoded=pm.compress(v4.codec_input(proxy),qa)
                torch.cuda.synchronize();te=time.perf_counter()
                decoded=pdec.decompress(encoded['bit_stream'],si,qa)
                torch.cuda.synchronize();td=time.perf_counter()
                if it>=20:
                    vals=dict(P_standalone=(tP-t)*1000,encode=(te-t)*1000,decode=(td-te)*1000,end_to_end=(td-t)*1000)
                    for k,val in vals.items():samples[k].append(val)
                    raw.append(dict(mode=mode,iteration=it-20,**vals,payload_bytes=len(encoded['bit_stream'])))
            peak=torch.cuda.max_memory_allocated()/2**20
            for component,values in samples.items():
                rows.append(dict(mode=mode,component=component,mean_ms=float(np.mean(values)),std_ms=float(np.std(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),FPS=1000/float(np.mean(values)),warmup=20,measured=100,peak_allocated_MiB=peak,gpu=gpu,batch=1,QP=q,actual_P_QP=qa,height=1080,width=1920))
            print('LATENCY',mode,rows[-1],flush=True)
    assert before==v4.compression_hash(im,pm)
    write('latency/latency_summary.csv',rows);write('latency/latency_raw.csv',raw)
    write('latency/latency_delta.csv',[dict(component=k,NoP_mean_ms=next(r['mean_ms'] for r in rows if r['mode']=='No-P' and r['component']==k),WithP_mean_ms=next(r['mean_ms'] for r in rows if r['mode']=='With-P' and r['component']==k),delta_WithP_minus_NoP_ms=next(r['mean_ms'] for r in rows if r['mode']=='With-P' and r['component']==k)-next(r['mean_ms'] for r in rows if r['mode']=='No-P' and r['component']==k)) for k in ('P_standalone','encode','decode','end_to_end')])
    # Conv-only analytical MAC count; all operations in the existing P are Conv2d/ReLU/add/clamp.
    hooks=[];mac=[0]
    for m in models['wrapper'].modules():
        if isinstance(m,torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda m,a,o:mac.__setitem__(0,mac[0]+o.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1])))
    with torch.inference_mode():models['wrapper'](x)
    for h in hooks:h.remove()
    dump('latency/complexity.json',dict(P_parameters=sum(p.numel() for p in models['wrapper'].parameters()),conv_MACs=mac[0],conv_FLOPs_multiply_add_2=2*mac[0],FLOPs_scope='Conv2d multiply-add only; excludes ReLU, residual addition, clamp',resolution=[1080,1920],batch=1,source_checkpoint=cp,receiver_same_for_NoP_and_WithP=True,P_sender_only=True,decode_does_not_execute_P=True,protocol='single P frame with reset native-I reference outside timed segment; includes real RANS, no I0 cost amortization',weights='historical v6.22 B1, measured before new training finishes'))
    complexity=load(ROOT/'latency/complexity.json');complexity['weights']='historical B1 reference' if checkpoint is None else 'new factorial D step1000; same D receiver in both paths';complexity['gpu']=gpu
    complexity.update(quality_models_excluded_from_memory=True,P_parameters_on_CPU_for_NoP_memory_measurement=True,memory_scope='single-process independent encoder+decoder plus active sender P; peak allocated bytes, not reserved cache',decode_P_forward_calls=0)
    dump('latency/complexity.json',complexity)
    gpu_after=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    dump('latency_integrity.json',dict(status='PASS',CUDA_synchronize=True,warmup=20,measured_iterations=100,compression_hash_unchanged=True,scope=complexity['weights'],checkpoint=cp,gpu=gpu,gpu_snapshot_before_model_load=gpu_before,gpu_snapshot_after_measurement=gpu_after,exclusive_GPU_not_assumed=True))

if __name__=='__main__':
    import traceback,argparse
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=7);a=ap.parse_args()
    try:main(a.checkpoint,a.gpu)
    except Exception:
        dump('latency/failure.json',dict(traceback=traceback.format_exc()));raise
