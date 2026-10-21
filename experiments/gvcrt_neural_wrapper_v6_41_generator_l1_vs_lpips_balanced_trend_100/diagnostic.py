"""Step0 paired objective and gradient audit; deliberately has no optimizer."""
from io32 import *

def verify(ctx, source, allmods, basic_crop=None):
    import torch, training32 as tr
    from frames32 import PlanFrames
    assert basic_crop is not None
    row=load(ROOT/'plans/first100.json')['updates'][0]['source_row']
    v4,im,pm,models,quality,common,params,_=ctx
    before={k:v4.module_hash(m) for k,m in allmods.items()}
    outputs=[]
    for lpips_weight in (1.,0.):
        tr.restore_rng(source['crop_rng']);tr.restore_buffers(allmods,source['buffers'])
        frames=PlanFrames(row,'cuda:0')
        gs,s=basic_crop(ctx,frames,row['external_qp'],audit=True,
                        dists_weight=0.,lpips_weight=lpips_weight)
        frames.close();outputs.append((gs,s))
    basic, l1=outputs
    a,b=basic[1]['losses'],l1[1]['losses']
    common_keys=('rate_bpp','beta_rate','proxy_L1','reconstruction_L1','weighted_reconstruction_L1')
    common_errors={k:abs(a[k]-b[k]) for k in common_keys}
    assert all(v<=1e-8 for v in common_errors.values()),common_errors
    scalar_error=abs((a['total_loss']-b['total_loss'])-a['LPIPS'])
    assert scalar_error<=1e-7,scalar_error
    comps=l1[1]['audit_component_tensors']
    norms={k:tr.norm(v) for k,v in comps.items()}
    assert norms['weighted_reconstruction']>0 and all(torch.isfinite(g).all() for g in comps['weighted_reconstruction'])
    assert norms['rate']==0 and norms['proxy']==0
    diff=[x-y-z for x,y,z in zip(basic[0],l1[0],basic[1]['audit_component_tensors']['LPIPS'])]
    lp_norm=norms_basic=tr.norm(basic[1]['audit_component_tensors']['LPIPS'])
    rel=tr.norm(diff)/max(lp_norm,1e-12)
    assert rel<=1e-3,(rel,lp_norm)
    assert a['proxy_L1']==b['proxy_L1']==0
    assert all(p.grad is None and not p.requires_grad for m in (models['bridge'],im,*quality) for p in m.parameters())
    assert all(p.grad is None and not p.requires_grad for n,p in pm.named_parameters() if not n.startswith('recon_generation_net.decoder.'))
    assert all(p.grad is None for p in models['generator'].parameters())
    tr.restore_rng(source['crop_rng']);tr.restore_buffers(allmods,source['buffers'])
    assert before=={k:v4.module_hash(m) for k,m in allmods.items()}
    dump(ROOT/'audits/objective_gradient_step0.json',dict(status='PASS',same_step0_same_input=True,
         diagnostic_optimizer_step_calls=0,scalar_difference_error=scalar_error,scalar_tolerance=1e-7,
         common_component_errors=common_errors,common_component_tolerance=1e-8,
         LPIPS_gradient_difference_relative_L2_error=rel,gradient_difference_relative_tolerance=1e-3,
         LPIPS_generator_gradient_norm=lp_norm,L1_generator_gradient_norm=norms['weighted_reconstruction'],
         rate_generator_gradient_norm=norms['rate'],identity_proxy_generator_gradient_norm=norms['proxy'],
         identity_proxy_L1=0,bridge_compression_and_quality_parameters_frozen=True,
         frozen_parameter_gradients_absent=True,modules_unchanged=True,RNG_and_buffers_restored=True,
         common_losses=[a,b]))
