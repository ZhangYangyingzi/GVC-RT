"""Same-device synchronized RANS decode timing, with no metric networks or file I/O in timer."""
from common_s2 import *
from evaluate import videos,historical_input
from model_context import backend,build,assert_reset,component_hashes
import io

def gpu_snapshot():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in gpu.splitlines() if x.split(',')[0].strip()=='4');uuid=row.split(',')[1].strip()
    others=[x for x in apps.splitlines() if x.split(',')[0].strip()==uuid and x.split(',')[1].strip()!=str(os.getpid())]
    return dict(physical_GPU=4,gpu=row,compute_processes=apps.splitlines(),other_processes_on_device=others,time=time.time())

def main():
    env();verify_sources();torch,core,eco,tr,qp,tree=backend(False,0)
    assert physical_gpu(0)==4
    settings=load(ROOT/'protocol.json')['decoding_timing']
    v=next(v for v in videos('development') if v['sequence']==settings['sequence'])
    begin=time.perf_counter();p,stream,trace=historical_input(v,5);raw=stream.read_bytes();io_seconds=time.perf_counter()-begin
    from src.utils import stream_helper as sh
    reader=io.BytesIO(raw);helper=sh.SPSHelper();packets=[]
    for i in range(64):
        header=sh.read_header(reader)
        while header['nal_type']==sh.NalType.NAL_SPS:
            helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
        sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
        assert q==p['actual_qps'][i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
        packets.append((dict(sps),q,payload))
    assert reader.tell()==len(raw)
    records=[];models=[]
    for group in ['baseline',*GROUPS]:
        checkpoint=None if group=='baseline' else Path(load(group_dir(group)/'selection.json')['best_trained']['checkpoint'])
        state=None if checkpoint is None else torch.load(checkpoint,map_location='cpu',weights_only=True)
        load_begin=time.perf_counter();c=build(0,train=False,losses=False,baseline=group=='baseline',group='A_control' if group=='baseline' else group,lora_checkpoint=state)
        load_seconds=time.perf_counter()-load_begin;im,pm=c['im'],c['pm'];frozen=component_hashes(c);warm_hash=None
        model_record=dict(model=group,LoRA_parameters=sum(x.numel() for x in c['params']),full_I_P_parameters=sum(x.numel() for m in [im,pm] for x in m.parameters()),
            model_load_seconds_excluded=load_seconds,checkpoint_sha256=None if checkpoint is None else sha(checkpoint),backend=c['provenance']['backend'])
        for repeat in range(4):
            assert_reset(c);before=gpu_snapshot();torch.cuda.synchronize(0);torch.cuda.reset_peak_memory_stats(0)
            start=time.perf_counter()
            with torch.inference_mode():
                for i,(sps,q,payload) in enumerate(packets):
                    if i==0:
                        decoded=im.decompress(payload,sps,q);assert_reset(c);pm.add_ref_frame(None,decoded['x_hat'])
                    else:decoded=pm.decompress(payload,sps,q)
                torch.cuda.synchronize(0)
            elapsed=time.perf_counter()-start
            last_hash=tensor_sha(decoded['x_hat']);feature_hash=tensor_sha(pm.dpb[0].feature)
            assert feature_hash==trace[-1]['reference_feature_sha256']
            if repeat==0:warm_hash=last_hash
            else:
                assert last_hash==warm_hash
                records.append(dict(model=group,repeat=repeat,physical_GPU=4,sequence=v['sequence'],QP=5,frames=64,
                    native_width=v['width'],native_height=v['height'],decoded_padded_width=packets[0][0]['width'],decoded_padded_height=packets[0][0]['height'],
                    codec_dtype='FP16',TCG_LoRA_internal_dtype='FP32',decode_seconds_excluding_IO_metrics_model_load=elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                    last_reconstruction_sha256=last_hash,stream_sha256=sha(stream),metric_networks_instantiated=False,
                    external_GPU_processes_observed=bool(before['other_processes_on_device']),resource_snapshot=json.dumps(before)))
            assert_reset(c)
        assert component_hashes(c)==frozen
        models.append(model_record);del decoded,im,pm,c,state;torch.cuda.empty_cache()
    assert len(records)==9
    write_csv(ROOT/'decode_timing.csv',records)
    dump(ROOT/'decoding_timing.json',dict(status='PASS',records=records,models=models,source_sequence_preregistered=True,
        pre_read_and_hash_IO_seconds_excluded=io_seconds,warmup_sequences=1,repeats=3,same_GPU_inputs_precision=True,
        includes_RANS_and_inherited_TCG_diagnostics=True,scope='Operational synchronized decode timing, not an exclusive real-time benchmark',
        no_exclusive_GPU_reservation=True,IO_and_model_load_and_metric_network_time_excluded=True))

if __name__=='__main__':main()
