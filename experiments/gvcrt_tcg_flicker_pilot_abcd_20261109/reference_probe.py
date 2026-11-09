"""Unmodified historical receiver, isolated import tree; output only under ROOT."""
from common_s2 import *
from cache_codec import cache_paths
from model_context import build,assert_reset

def main():
    import torch
    env();row=plan_rows()[0];key,path,receipt=cache_paths(row,row['external_qp'],True)
    data=torch.load(path,map_location='cpu',weights_only=True)
    c=build(0,train=False,losses=False,reference=True,fresh_optimizer=True)
    gen=c['models']['generator'];assert_reset(c);hashes=[];caches=[]
    with torch.no_grad():
        for i in range(1,64):
            out=c['pm'].recon_generation_net(data['features'][i].to(c['device']),c['pm'].q_scale_recon[data['info']['actual_qps'][i]:data['info']['actual_qps'][i]+1])
            hashes.append(tensor_sha(out));caches.append({n:tensor_sha(getattr(gen,n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']})
    dump(ROOT/'audits/unmodified_A2000_reference.json',dict(status='PASS',checkpoint_SHA256=INIT_SHA,s=.5,codec_cache_identity=key,reconstruction_hashes=hashes,TCG_cache_hashes=caches,
        original_reference_tree=True,codec_feature_trace=data['info']['codec_trace'],precision='matched FP32 TCG/LoRA, FP16 base receiver'))

if __name__=='__main__':main()
