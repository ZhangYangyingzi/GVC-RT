import math,random
from v61_io import *
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='4'
    import torch
    import torch.nn.functional as F
    cfg=load(ROOT/'config.json');v4=module('v61_smoke_v4',V4/'train.py');device=torch.device('cuda:0')
    cp=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True)
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    im.requires_grad_(False);pm.requires_grad_(False);bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    for k,m in models.items():m.load_state_dict(cp[k],strict=True);assert all(torch.equal(v.cpu(),cp[k][n]) for n,v in m.state_dict().items())
    before={k:v4.module_hash(m) for k,m in models.items()};compression=v4.compression_hash(im,pm);quality=v4.quality_models(device)
    frames,plan=v4.sample_clip(load(V41/'train_manifest.json')['videos'],random.Random(cfg['seed']),device);results=[]
    for q in (0,9):
        for m in models.values():m.zero_grad(set_to_none=True)
        pm.clear_dpb();pm.set_curr_poc(0)
        with torch.no_grad():encoded=im.compress(v4.codec_input(wrapper(frames[0])),q);pm.add_ref_frame(None,encoded['x_hat'])
        captured=[];hook=pm.dec.register_forward_hook(lambda _m,_a,out:captured.append(out));loss_sum=0.0
        try:
            for x in frames[1:]:
                y=wrapper(x);rec,rate,_=v4.joint_ste_forward(pm,y,q);dist,_,_=v4.perceptual(rec,x,quality)
                loss=(dist+cfg['beta']*rate/(x.shape[-2]*x.shape[-1])+cfg['lambda_proxy']*F.l1_loss(y,x))/3
                assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
                with torch.no_grad():pm.add_ref_frame(captured[-1].detach(),(rec.detach()*2-1).half())
        finally:hook.remove()
        norms={k:v4.grad_norm(m.parameters()) for k,m in models.items()};assert all(math.isfinite(v) and v>0 for v in norms.values())
        assert v4.compression_hash(im,pm)==compression and all(v4.module_hash(m)==before[k] for k,m in models.items())
        results.append(dict(external_qp=q,actual_i_qp=q,actual_training_p_qps=[q,q,q],loss=loss_sum,gradient_norms=norms))
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    dump(ROOT/'initialization_smoke_audit.json',dict(status='PASS',physical_gpu=4,optimizer_updates=0,source_checkpoint_unchanged=True,trainable_weights_unchanged=True,compression_unchanged=True,clip_length=4,crop=[256,256],sample=plan,checks=results))
    print('INITIALIZATION / QP0 QP9 BACKWARD PASS; NO OPTIMIZER UPDATES',flush=True)
if __name__=='__main__':main()
