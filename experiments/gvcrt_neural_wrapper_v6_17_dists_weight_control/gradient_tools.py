"""Component-specific gradients; finite differences only on independent RGB."""
import math
from io17 import *
def gradients(loss,params):
    import torch
    if not loss.requires_grad:return [None for _ in params]
    return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
def stats(gs,params):
    import torch
    defined=[g for g in gs if g is not None];s=sum(float(g.detach().double().square().sum()) for g in defined)
    return dict(norm=math.sqrt(s),finite=all(bool(torch.isfinite(g).all()) for g in defined),total_parameters=len(params),connected_parameters=len(defined),effective_parameters=sum(bool((g!=0).any()) for g in defined),effective_elements=sum(int((g!=0).sum()) for g in defined))
def dists_details(quality,x,y,fixed):
    import torch
    captures=[];ds=quality[1];original=ds.forward_once
    def traced(z):
        fs=original(z);captures.append(fs);return fs
    ds.forward_once=traced
    try:score=ds(x,y,require_grad=True).mean() if fixed else quality[1](x,y).mean()
    finally:ds.forward_once=original
    assert len(captures)==2
    terms=[];ws=ds.alpha.sum()+ds.beta.sum();alpha=torch.split(ds.alpha/ws,ds.chns,dim=1);beta=torch.split(ds.beta/ws,ds.chns,dim=1)
    for k,(a,b) in enumerate(zip(*captures)):
        am=a.mean([2,3],keepdim=True);bm=b.mean([2,3],keepdim=True)
        s1=(2*am*bm+1e-6)/(am.square()+bm.square()+1e-6)
        av=(a-am).square().mean([2,3],keepdim=True);bv=(b-bm).square().mean([2,3],keepdim=True);cov=(a*b).mean([2,3],keepdim=True)-am*bm
        s2=(2*cov+1e-6)/(av+bv+1e-6)
        terms.append(-((alpha[k]*s1).sum(1,keepdim=True)+(beta[k]*s2).sum(1,keepdim=True)).mean())
    assert torch.allclose(1+sum(terms),score,rtol=1e-5,atol=2e-6)
    layers=[]
    for k,(feature,term) in enumerate(zip(captures[0],terms)):
        g=gradients(term,[x]);layers.append(dict(layer=k,kind='RGB' if k==0 else 'deep',shape=list(feature.shape),requires_grad=feature.requires_grad,grad_fn=None if feature.grad_fn is None else type(feature.grad_fn).__name__,connected_to_output=g[0] is not None,output_gradient=stats(g,[x])))
    return score,layers,terms
