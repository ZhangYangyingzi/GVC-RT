"""Physical LoRA removal equality against gain0, with freshly rebuilt histories."""
from common_s2 import *

def main(rank=1):
    import torch,copy
    from experiment_model import construct,reconstruct,trainable_state
    from convlora import set_inference_gain
    from model_context import assert_reset,frozen_items
    from cache_codec import cache_paths
    from pipeline import wait_rows
    env()
    if (ROOT/'initialization_audit.json').exists():assert load(ROOT/'initialization_audit.json')['status']=='PASS';return
    row=plan_rows()[0];wait_rows([row]);_,p,rp=cache_paths(row,row['external_qp'],True);assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    seed(PLAN_SEED);reference=construct('B300_original_reference',True,rank);original=tensor_hash(reference['pm'].recon_generation_net.state_dict().items())
    atomic_torch(ROOT/'checkpoints/B300_original_reference.pt',dict(recon_state={n:t.detach().cpu().clone() for n,t in reference['pm'].recon_generation_net.state_dict().items()},B300_SHA256=B300_SHA,LoRA_gain=.5,provenance=reference['provenance']))
    common=construct(GROUPS[0],True,rank,reference=reference);common_tensor=tensor_hash(common['pm'].recon_generation_net.state_dict().items())
    expected={n:t.detach().cpu().clone() for n,t in common['pm'].recon_generation_net.state_dict().items()}
    atomic_torch(ROOT/'checkpoints/B300_noLoRA_step0.pt',dict(recon_state=expected,trainable_state=trainable_state(common),B300_SHA256=B300_SHA,LoRA_physically_removed=True,LoRA_merged=False,source_reference_recon_SHA256=original,recon_tensor_SHA256=common_tensor,provenance=common['provenance']))
    common_sha=sha(ROOT/'checkpoints/B300_noLoRA_step0.pt')
    zero=construct('B300_original_reference',True,rank,reference=reference);set_inference_gain(zero['models']['generator'],0.)
    for c in [zero,common]:assert_reset(c)
    frames=[]
    with torch.no_grad():
        for i in range(1,64):
            a=reconstruct(zero,d['features'][i],d['info']['actual_qps'][i]);b=reconstruct(common,d['features'][i],d['info']['actual_qps'][i])
            error=float((a-b).abs().max());assert error==0,('PHYSICAL_REMOVAL_VS_GAIN0',i,error)
            frames.append(dict(frame=i,max_abs=error,reconstruction_SHA256=tensor_sha(b)))
    assert original==tensor_hash(reference['pm'].recon_generation_net.state_dict().items())
    assert common_tensor==tensor_hash(common['pm'].recon_generation_net.state_dict().items())
    clone=construct(GROUPS[1],True,rank);assert tensor_hash(clone['pm'].recon_generation_net.state_dict().items())==common_tensor
    for c in [common,clone]:
        assert len(c['optimizer'].state)==0 and sum(p.numel() for p in c['params'])==938884
        assert all(p.requires_grad==(n in c['trainable_parameter_names']) for n,p in c['pm'].named_parameters())
        assert not any('lora' in n.lower() or 'router' in n.lower() for n,m in c['pm'].named_modules())
    names=[dict(name=n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype)) for n,p in common['named']]
    frozen=[dict(name=n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype)) for n,p in common['pm'].named_parameters() if n not in common['trainable_parameter_names']]
    dump(ROOT/'trainable_parameters.json',dict(Bridge=801124,TCG=137760,total=938884,trainables=names,frozen=frozen,LoRA_parameters=0,groups_match=True))
    dump(ROOT/'initialization_audit.json',dict(status='PASS',B300_SHA256=B300_SHA,common_step0_path=str(ROOT/'checkpoints/B300_noLoRA_step0.pt'),common_step0_SHA256=common_sha,common_reconstruction_tensor_SHA256=common_tensor,reference_recon_tensor_SHA256=original,reference_LoRA_gain=.5,comparison_LoRA_gain=0.,comparison_dtype='FP32 both, identical safe dispatch and inputs, histories reset and rebuilt1..63',physical_removal_matches_gain0_exact=True,G0_G1_step0_same=True,frames=frames,source_Bridge_TCG_previously_LoRA_trained=True,base_groups_preserved=True,LoRA_not_merged=True,no_teacher_in_inference=True,time=time.time()))
    print('INITIALIZATION_AND_GAIN0_EQUALITY_PASS',common_sha,flush=True)
if __name__=='__main__':main()
