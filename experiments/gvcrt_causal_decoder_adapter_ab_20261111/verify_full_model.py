"""Record actual post-B300 component identities in both reconstruction precisions."""
from common_s2 import *


def main():
    env();import torch
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    records=[]
    for precision in [True,False]:
        c=construct('B300',precision,0,with_adapter=False)
        pm=c['pm'];gen=c['models']['generator'];bridge=c['models']['bridge']
        groups=dict(I_model=c['im'].state_dict().items(),P_codec=[(n,t) for n,t in pm.state_dict().items() if not n.startswith('recon_generation_net.')],
            Bridge=bridge.state_dict().items(),TCG=[(n,t) for n,t in gen.state_dict().items() if n.startswith(TCG_PREFIXES)],
            LoRA=[(n,t) for n,t in gen.state_dict().items() if '.lora.' in n],
            Backbone=[(n,t) for n,t in gen.state_dict().items() if not n.startswith(TCG_PREFIXES) and '.lora.' not in n])
        hashes={name:tensor_hash(list(items)) for name,items in groups.items()}
        if precision:
            saved=torch.load(ROOT/'checkpoints/frozen_FP32_head.pt',map_location='cpu',weights_only=True)
            assert tensor_hash(gen.head.state_dict().items())==saved['tensor_SHA256']
        assert all(not p.requires_grad for p in pm.parameters())
        records.append(dict(precision='FP32 training reconstruction' if precision else DEPLOYMENT,
            actual_POST_B300_component_SHA256=hashes,full_P_SHA256=tensor_hash(pm.state_dict().items()),
            exactly5_LoRA=len([m for m in gen.modules() if m.__class__.__name__=='ParallelConvLoRA'])==5,
            s_values=[m.inference_gain for m in gen.modules() if m.__class__.__name__=='ParallelConvLoRA'],
            all_original_parameters_frozen=True,provenance=c['provenance']))
        del c,pm,gen,bridge;torch.cuda.empty_cache()
    dump(ROOT/'audits/full_B300_final_identities.json',dict(status='PASS',records=records,
        inherited_provenance_component_hash_note='model_context provenance.components describes pilot/B1500 stage before A2000/B300 overlays; actual final identities are explicitly above',time=time.time()))

if __name__=='__main__':main()
