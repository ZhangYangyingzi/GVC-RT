"""Single pre-registered FP32 centered covariance target; no extra normalization."""
import torch
LAYERS=(2,3,4)
def covariance(f):
    z=f.float().flatten(2);z=z-z.mean(-1,keepdim=True)
    return torch.bmm(z,z.transpose(1,2))/z.shape[-1]
def feature_loss(rec,gt):
    terms=[]
    for i in LAYERS:
        a=covariance(rec[i]);b=covariance(gt[i]).detach()
        terms.append((a-b).square().mean()/(b.square().mean().detach()+1e-6))
    return torch.stack(terms).mean(),terms
def dists_and_stat(ds,rec,gt):
    captures=[];original=ds.forward_once
    def capture(x):
        f=original(x);captures.append(f);return f
    ds.forward_once=capture
    try:score=ds(rec,gt,require_grad=True).mean()
    finally:ds.forward_once=original
    assert len(captures)==2
    loss,terms=feature_loss(*captures)
    return score,loss,terms,[list(captures[0][i].shape) for i in LAYERS]
