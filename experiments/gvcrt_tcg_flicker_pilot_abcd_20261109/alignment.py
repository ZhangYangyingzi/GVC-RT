"""Independent original-weight receiver: no TCG/LoRA/updated Bridge."""
from common_s2 import *
import torch
from temporal import flow_valid

def receiver(c):
    from src.models.video_model_gvcrt import PretrainedReconWrapper
    m=PretrainedReconWrapper()
    # Remove the optional TCG modules before strict loading. This is an
    # original-only receiver instance, not candidate bypass after training.
    for name in ['tcg_low','tcg_mid','tcg_out']:setattr(m.decoder,name,None)
    raw=torch.load(PROJECT/'checkpoints/GVC-RT_P.pt',map_location='cpu',weights_only=True)
    raw=raw.get('student_ema',raw.get('student',raw.get('state_dict',raw)))
    prefix='recon_generation_net.';sd={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
    assert set(sd)==set(m.state_dict()),'ORIGINAL_ONLY_RECEIVER_KEYS_NOT_EXACT'
    m.load_state_dict(sd,strict=True);m=m.to(c['device']).half().eval().requires_grad_(False)
    assert not any(any(x in n.lower() for x in ['tcg','lora','router']) for n,t in m.state_dict().items())
    return m

def check_online(c,data,frames=(1,2)):
    from cache_codec import load_raft
    m=receiver(c);raft,transform=load_raft(c['device']);before=tensor_hash(m.state_dict().items());records=[];h,w=data['info']['native_shape'];torch.cuda.reset_peak_memory_stats(c['device'])
    def pad(x):return torch.nn.functional.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
    def flow(a,b):
        a,b=transform(pad(a),pad(b));return raft(a,b)[-1][...,:h,:w]
    with torch.no_grad():
        for i in frames:
            previous=data['original'][i-1].to(c['device'])
            if i>1:
                q=data['info']['actual_qps'][i-1]
                previous=m(data['features'][i-1].to(c['device']),data['qscale'][q:q+1].to(c['device']))
                assert torch.equal(previous.cpu(),data['original'][i-1]),'ONLINE_ORIGINAL_RECEIVER_PREVIOUS_MISMATCH'
            torch.cuda.synchronize(c['device']);started=time.perf_counter();q=data['info']['actual_qps'][i]
            current=m(data['features'][i].to(c['device']),data['qscale'][q:q+1].to(c['device']))
            torch.cuda.synchronize(c['device']);receiver_seconds=time.perf_counter()-started
            assert torch.equal(current.cpu(),data['original'][i]),'ONLINE_ORIGINAL_RECEIVER_CURRENT_MISMATCH'
            a=c['core'].unit(current,h,w);b=c['core'].unit(previous,h,w)
            torch.cuda.synchronize(c['device']);started=time.perf_counter();back=flow(a,b);forward=flow(b,a);valid=flow_valid(back,forward);torch.cuda.synchronize(c['device'])
            flow_seconds=time.perf_counter()-started
            assert torch.equal(back.cpu(),data['base_backward'][i]),'ONLINE_VS_CACHE_BACKWARD_FLOW_MISMATCH'
            assert torch.equal(valid.cpu(),data['base_valid'][i]),'ONLINE_VS_CACHE_VALID_MASK_MISMATCH'
            records.append(dict(frame=i,original_receiver_seconds=receiver_seconds,RAFT_two_directions_validity_seconds=flow_seconds,total_extra_seconds=receiver_seconds+flow_seconds,
                flow_SHA256=tensor_sha(back),mask_SHA256=tensor_sha(valid),geometry=[h,w],GT_input_supplied=False))
    assert tensor_hash(m.state_dict().items())==before
    return dict(status='PASS',records=records,original_receiver_exact_cached_original_RANS_reconstruction=True,original_receiver_SHA256=before,
        original_weight_source_SHA256=sha(PROJECT/'checkpoints/GVC-RT_P.pt'),no_TCG_LoRA_Router=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(c['device']),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(c['device']),timing='Measured uncached original receiver + two RAFT directions + fixed validity, CUDA synchronized; excludes loading and I/O',physical_GPU=physical_gpu(c['device'].index))
