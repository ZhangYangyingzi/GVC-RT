from util import *
import torch
def restore_b300(fp32=True):
    from model_context import build
    from common_s2 import B300,B300_SHA,INIT_SHA,TCG_PREFIXES
    # Use the audited loading chain but immediately discard its temporary optimizer.
    c=build(0,train=fp32,losses=False,group='A_control',fresh_optimizer=True)
    c['optimizer']=None
    assert sha(B300)==B300_SHA
    b=torch.load(B300,map_location='cpu',weights_only=True)
    assert b['experiment_group']=='B' and b['committed_updates']==300 and b['initial_A2000_sha256']==INIT_SHA
    keys={n for n,p in c['pm'].named_parameters() if '.lora.' in n or n.startswith('recon_generation_net.mlp.3.') or n.startswith(tuple('recon_generation_net.decoder.'+s for s in TCG_PREFIXES))}
    assert set(b['trainable_state'])==keys and len(keys)==52
    state=dict(c['pm'].state_dict());state.update(b['trainable_state']);c['pm'].load_state_dict(state,strict=True)
    for n,t in b['trainable_state'].items():assert torch.equal(c['pm'].state_dict()[n].cpu(),t.to(c['pm'].state_dict()[n].dtype))
    for m in [c['im'],c['pm']]:m.eval().requires_grad_(False)
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(False)
    # The FP32 reference avoids custom fused decoder operators. Codec dispatch remains original.
    if fp32:
        import types
        for m in gen.modules():
            if m.__class__.__name__ in ['DepthConvBlock','StageBlock']:
                m.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),m)
                if hasattr(m,'proxy'):m.proxy=None
    assert all(m.inference_gain==.5 for m in gen.modules() if hasattr(m,'inference_gain'))
    c['provenance'].update(B300_overlay_last=True,B300_keys=sorted(keys),B300_SHA256=B300_SHA,all_frozen=True,receiver_precision='FP32' if fp32 else 'FP16')
    c['named']=[];c['params']=[];c['trainable_parameter_names']=set()
    assert not any(p.requires_grad for p in c['pm'].parameters())
    return c

