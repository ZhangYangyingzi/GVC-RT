"""Native STE forward with only the direct Gaussian-scale backward path added."""
import functools, inspect
import torch
from core import prepare_p, codec_input, unit, round_and_to_int8
from v68_io import ROOT,sha

def gaussian_component(quantized,sigma,active,gaussian_encoder,mode):
    assert mode in ('legacy','scale_ste')
    quantized = quantized.float()[active]
    s = sigma.float()[active].clamp(.11,16)
    index = ((torch.log(s)-gaussian_encoder.log_scale_min)*gaussian_encoder.log_step_recip).long().clamp(0,127)
    stable = gaussian_encoder.scale_table.to(s.device)[index]
    scale_for_rate = stable if mode=='legacy' else stable.detach()+(s-s.detach())
    normal = torch.distributions.Normal(torch.zeros_like(stable),scale_for_rate)
    probability = (normal.cdf(quantized+.5)-normal.cdf(quantized-.5)).clamp_min(1e-9)
    bits = (-torch.log2(probability)).sum()
    return bits,dict(s=s,stable=stable,index=index,scale_for_rate=scale_for_rate,
                     quantized=quantized,active=active,probability=probability)

def bind(v4,mode):
    assert mode in ('legacy','scale_ste')
    v4.rate_trace={}
    v4.joint_ste_forward=functools.partial(joint_ste_forward,mode=mode,trace=v4.rate_trace)
    v4.rate_source=dict(path=inspect.getsourcefile(joint_ste_forward),
        sha256=sha(ROOT/'rate_forward.py'),function='joint_ste_forward',
        mode=mode,bound_before_functional_call=True,
        gaussian_helper='gaussian_component')

def joint_ste_forward(model, proxy, qp, *, mode, trace):
    context, temporal, qe, qd, qr = prepare_p(model, qp)
    y = model.enc(codec_input(proxy), context, qe)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hard = torch.clamp(torch.round(z), -128, 127)
    z_ste = z + (z_hard - z).detach()
    params = model.res_prior_param_decoder(z_ste, temporal)
    qdec, scale, mean = params.chunk(3, 1)
    qdec = torch.clamp_min(qdec, .5)
    y_scaled = y * torch.reciprocal(qdec)
    batch, channels, height, width = y_scaled.shape
    mask0, mask1 = model.get_mask_2x(batch, channels, height, width,
                                     y_scaled.dtype, y_scaled.device)

    def process(value, sigma, mu, mask):
        sigma, mu = sigma * mask, mu * mask
        residual = (value - mu) * mask
        hard = torch.clamp(torch.round(residual), -128, 127)
        active = mask.bool() & (sigma > .12)
        hard = hard * active
        ste = residual + (hard - residual).detach()
        return hard, ste, sigma, mu, active

    hard0, quant0, scale0, mean0, active0 = process(y_scaled, scale, mean, mask0)
    y_hat0 = quant0 + mean0
    scale1, mean1 = model.y_spatial_prior(torch.cat((y_hat0, params), 1)).chunk(2, 1)
    hard1, quant1, scale1, mean1, active1 = process(y_scaled, scale1, mean1, mask1)
    y_hat = (y_hat0 + quant1 + mean1) * qdec
    feature = model.dec(y_hat, context, qd)
    reconstruction = unit(model.recon_generation_net(feature.float(), qr.float()),
                          proxy.shape[-2], proxy.shape[-1])
    upper = model.bit_estimator_z.get_cdf(z_ste.float() + .5, qp)
    lower = model.bit_estimator_z.get_cdf(z_ste.float() - .5, qp)
    rate = (-torch.log2((upper - lower).clamp_min(1e-9))).sum()

    rz = rate
    ry0, g0 = gaussian_component(quant0, scale0, active0, model.gaussian_encoder, mode)
    ry1, g1 = gaussian_component(quant1, scale1, active1, model.gaussian_encoder, mode)
    # Preserve the original left-associated floating-point addition order.
    rate = rate + ry0 + ry1
    trace.clear()
    trace.update(R_y0=ry0, R_y1=ry1, R_z=rz, total_rate=rate, passes=[g0,g1],
                 z_hard=z_hard, hard0=hard0, hard1=hard1, active0=active0,
                 active1=active1, y_hat=y_hat)
    with torch.no_grad():
        z_true, _ = round_and_to_int8(z)
        true_params = model.res_prior_param_decoder(z_true, temporal)
        _, _, _, _, true_y_hat = model.compress_prior_2x(y, true_params, model.y_spatial_prior)
        ste_difference = float((true_y_hat - y_hat.detach()).abs().max())
    return reconstruction, rate, ste_difference
