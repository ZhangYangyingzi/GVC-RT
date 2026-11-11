"""Independent original GVC-RT weights, FP32 P receiver; no pilot, TCG or LoRA."""
from common_s2 import *
import torch,types,argparse


def main(cache,output):
    from original_context import runtime
    from original_runtime import load_models,verify_codec_imports
    rt=runtime(0);im,pm=load_models(rt);gen=pm.recon_generation_net.decoder
    pm.recon_generation_net.float()
    for m in gen.modules():
        if m.__class__.__name__ in ['DepthConvBlock','StageBlock','SubpelConv2x']:
            m.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),m)
    gen.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),gen)
    assert not any('tcg' in n.lower() or 'lora' in n.lower() for n,p in pm.named_parameters())
    before=tensor_hash(pm.state_dict().items());data=torch.load(cache,map_location='cpu',weights_only=True,mmap=True);values=[data['original'][0]]
    with torch.no_grad():
        for i in range(1,data['info']['frames']):
            f=data['features'][i].to('cuda:0').float();q=data['info']['actual_qps'][i]
            values.append(pm.recon_generation_net(f,pm.q_scale_recon[q:q+1]).cpu())
    assert before==tensor_hash(pm.state_dict().items()) and not pm.dpb
    atomic_torch(Path(output),dict(values=values,stream_SHA256=data['info']['stream_SHA256'],precision=DEPLOYMENT,original_commit=ORIGINAL_COMMIT,no_pilot_TCG_LoRA=True,I_frame='canonical independent original FP16 codec',source_imports=verify_codec_imports(),state_SHA256=before))


if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('output');a=p.parse_args();main(a.cache,a.output)
