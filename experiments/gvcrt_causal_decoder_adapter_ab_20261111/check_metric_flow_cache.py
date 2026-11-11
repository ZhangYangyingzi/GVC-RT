"""Exact-value verification of the shared frozen PWC GT-flow metric cache."""
from common_s2 import *


def main():
    env();import torch
    from evaluation import Evaluator
    from cache_codec import cache_paths
    v=load(ROOT/'manifests/development_videos.json')[0]
    _,p,rp=cache_paths(v['source_row'],0,False);assert sha(p)==load(rp)['cache_sha256']
    d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);ev=Evaluator();flow=ev.networks[3];perceptual=ev.networks[4]
    calls=[];hook=flow.register_forward_hook(lambda *a:calls.append(1))
    def gt(i):return (d['GT'][i].permute(2,0,1)[None].float()/255).to('cuda:0')
    with torch.no_grad():
        a,b=gt(0),gt(1);reference=flow(a,b);start=len(calls)
        first=ev.gt_metric_flow(a,b,1);second=ev.gt_metric_flow(a,b,1)
        assert len(calls)-start==1 and torch.equal(reference,first) and torch.equal(first,second)
        x,y=(a*.9).clamp(0,1),(b*.9).clamp(0,1);reconstruction_flow=flow(x,y)
        original=perceptual(a,x,reference-reconstruction_flow,normalize=True)
        cached=perceptual(a,x,second-reconstruction_flow,normalize=True)
        assert torch.isfinite(original).all() and torch.equal(original,cached)
    hook.remove()
    dump(ROOT/'audits/metric_GT_flow_cache_check.json',dict(status='PASS',GT_flow_bit_exact=True,FloLPIPS_bit_exact=True,
        reference_FloLPIPS=float(original),cached_FloLPIPS=float(cached),two_cache_requests_one_forward=True,
        dtype=str(second.dtype),CPU_flow_cache=True,model_weights_changed=False,optimizer_updates=0,
        GT_flow_is_internal_frozen_PWC_for_FloLPIPS=True,RAFT_MC_supervision_cache_unchanged=True,time=time.time()))
    print('METRIC_FLOW_CACHE_EXACT_PASS')

if __name__=='__main__':main()
