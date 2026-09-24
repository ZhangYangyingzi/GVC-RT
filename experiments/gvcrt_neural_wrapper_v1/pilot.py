#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
import torch
from wrapper_model import NeuralWrapper
from core import ROOT, load_models, load_train_pair, establish_dpb, ste_forward_p, quality_models, perceptual, model_hash


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.float().square().sum()) for g in grads if g is not None))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,default=4); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); torch.manual_seed(20260924); dev=torch.device('cuda:0')
    wrapper=NeuralWrapper().to(dev).float().train(); sample=torch.rand(1,3,256,256,device=dev)
    identity=float((wrapper(sample)-sample).abs().max()); params=list(wrapper.parameters())
    im,pm=load_models(dev); frozen_before=model_hash(im,pm); quality=quality_models(dev)
    (previous,current),plan=load_train_pair(1,dev); qp=int(plan['requested_qp'])
    with torch.no_grad(): previous_proxy=wrapper(previous)
    establish_dpb(im,pm,previous_proxy,qp); proxy=wrapper(current); out,rate,ste=ste_forward_p(pm,proxy,qp)
    distortion,lp,di=perceptual(out,current,quality); rnorm=rate/current.shape[-2]/current.shape[-1]
    gd=grad_norm(distortion,params); gr=grad_norm(rnorm,params)
    if not all(math.isfinite(x) and x>0 for x in (gd,gr,float(distortion),float(rnorm))): raise RuntimeError('pilot finite/nonzero gradient gate failed')
    balance=gd/gr; betas={'beta_low':balance*.3,'beta_mid':balance,'beta_high':balance*3}
    result={'identity_initialization_max_abs_diff':identity,'wrapper_identity_init_pass':identity==0,'distortion':float(distortion),'LPIPS':float(lp),'DISTS':float(di),'R_est_bpp':float(rnorm),'grad_perceptual_norm':gd,'grad_rate_norm':gr,'grad_balance_beta':balance,'betas':betas,'STE_forward_max_abs_diff':ste,'frozen_hash_before':frozen_before,'frozen_hash_after':model_hash(im,pm),'all_finite':True,'status':'PASS'}
    (ROOT/'sanity/pilot_calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    cfg=json.load(open(ROOT/'config.json')); cfg['beta_weights']=betas; cfg['training_enabled']=True; (ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

