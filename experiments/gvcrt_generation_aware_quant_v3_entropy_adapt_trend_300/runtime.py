"""Receiver-synchronous experimental modules; no encoder-only side inputs."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.video_model_gvcrt import DMC


class BlockQuantAdjust(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(385, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, common_params, qp):
        p = common_params.float()
        b, _, h, w = p.shape
        pooled = F.avg_pool2d(p, 4, stride=4, ceil_mode=True)
        q = pooled.new_full((b, 1, *pooled.shape[-2:]), qp / 12)
        a = self.net(torch.cat((pooled, q), 1))
        m = torch.exp(math.log(2) * torch.tanh(a))
        return m.repeat_interleave(4, -2).repeat_interleave(4, -1)[..., :h, :w]


class SpatialVisualCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(260, 32, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.stage2 = nn.Conv2d(32, 1024, 1)
        self.stage4 = nn.Conv2d(32, 640, 1)
        for layer in (self.stage2, self.stage4):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, common_params, q_new, qp):
        size = feature.shape[-2:]
        prior = common_params.float()
        pieces = [feature.float(),
                  F.interpolate(q_new.float().mean(1, keepdim=True), size, mode='nearest'),
                  F.interpolate(prior[:, 128:256].mean(1, keepdim=True), size, mode='nearest'),
                  F.interpolate(prior[:, 256:384].mean(1, keepdim=True), size, mode='nearest'),
                  feature.new_full((feature.shape[0], 1, *size), qp / 12).float()]
        x = self.trunk(torch.cat(pieces, 1))
        return self.stage2(F.interpolate(x, scale_factor=.5, mode='bilinear', align_corners=False)), self.stage4(x)


class ExperimentalDMC(DMC):
    """Base codec with a receiver-recomputed block step and optional spatial G input."""
    def install(self, use_quant=False, use_condition=False):
        self.use_quant = bool(use_quant)
        self.use_condition = bool(use_condition)
        self.current_qp = None
        self.prior_params = None
        self.q_actual = None
        if use_quant:
            self.quant_adjust = BlockQuantAdjust().to(next(self.parameters()).device)
        if use_condition:
            self.visual_condition = SpatialVisualCondition().to(next(self.parameters()).device)
            g = self.recon_generation_net.decoder
            self._condition_maps = None
            self._condition_hooks = [g.stage2.register_forward_hook(self._affine_hook(0)),
                                     g.stage4.register_forward_hook(self._affine_hook(1))]
        return self

    def _affine_hook(self, index):
        def hook(module, inputs, output):
            if self._condition_maps is None:
                return output
            affine = self._condition_maps[index]
            if affine.shape[-2:] != output.shape[-2:]:
                affine = F.interpolate(affine, output.shape[-2:], mode='bilinear', align_corners=False)
            gamma, beta = affine.chunk(2, 1)
            return output * (1 + gamma.to(output.dtype)) + beta.to(output.dtype)
        return hook

    def res_prior_param_decoder(self, z_hat, ctx_t):
        params = super().res_prior_param_decoder(z_hat, ctx_t)
        self.prior_params = params
        self.q_actual = self.quant_step(params)
        return params

    def quant_step(self, params):
        base = params[:, :128].clamp_min(.5)
        if self.use_quant:
            assert self.current_qp is not None
            result = (base.float() * self.quant_adjust(params, self.current_qp)).clamp_min(.5)
            return result.to(base.dtype)
        return base

    def separate_prior_for_video_encoding(self, params, y):
        q = self.quant_step(params)
        self.q_actual = q
        return y * torch.reciprocal(q), q, *params[:, 128:].chunk(2, 1)

    def separate_prior_for_video_decoding(self, params):
        q = self.quant_step(params)
        self.q_actual = q
        return q, *params[:, 128:].chunk(2, 1)

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.dec(y_hat, ctx, q_decoder)
        if self.use_condition:
            assert self.prior_params is not None and self.q_actual is not None
            self._condition_maps = self.visual_condition(feature, self.prior_params,
                                                          self.q_actual, self.current_qp)
        try:
            x_hat = self.recon_generation_net(feature.float(), q_recon.float())
        finally:
            if self.use_condition:
                self._condition_maps = None
        return x_hat, feature

    def compress(self, x, qp):
        self.current_qp = qp
        return super().compress(x, qp)

    def decompress(self, bit_stream, sps, qp):
        self.current_qp = qp
        return super().decompress(bit_stream, sps, qp)



class SkipDomainDMC(ExperimentalDMC):
    """Local two-pass coordinate transform, shared by coding and independent decoding."""
    def install_v2(self, branch):
        self.install(branch != 'A', False)
        self.branch = branch
        self.fix_skip = branch in ('H10', 'H11')
        self.domain = branch in ('H01', 'H11')
        return self

    def coordinates(self, params):
        base = params[:, :128].clamp_min(.5)
        q = self.quant_step(params)
        self.q_actual = q
        rho = q / base
        scale, mean = params[:, 128:].chunk(2, 1)
        if self.domain:
            scale, mean = scale / rho, mean / rho
        return q, rho, scale, mean

    def second_prior(self, u0, common, spatial, rho):
        prior_input = u0 * rho if self.domain else u0
        scale, mean = spatial(torch.cat((prior_input, common), 1)).chunk(2, 1)
        if self.domain:
            scale, mean = scale / rho, mean / rho
        return scale, mean

    def compress_prior_2x(self, y, common_params, y_spatial_prior):
        q, rho, scales, means = self.coordinates(common_params)
        y = y * torch.reciprocal(q)
        masks = self.get_mask_2x(*y.shape, y.dtype, y.device)
        _, q0, u0, s0 = self.process_with_mask(y, scales, means, masks[0])
        scales, means = self.second_prior(u0, common_params, y_spatial_prior, rho)
        _, q1, u1, s1 = self.process_with_mask(y, scales, means, masks[1])
        # Match native arithmetic without mutating tensors retained by prior/diagnostics.
        from src.layers.cuda_inference import add_and_multiply
        y_hat = add_and_multiply(u0.clone(), u1, q)
        self.coding_trace = {'q0': q0, 'q1': q1, 'qmap': q, 'rho': rho, 'y_hat': y_hat}
        return tuple(self.single_part_for_writing_2x(t) for t in (q0, q1, s0, s1)) + (y_hat,)

    def decompress_prior_2x_part1(self, common_params):
        from src.layers.cuda_inference import combine_for_reading_2x
        q, rho, scales, means = self.coordinates(common_params)
        m0, m1 = self.get_mask_2x(*means.shape, means.dtype, means.device)
        scales_r = combine_for_reading_2x(scales, m0, inplace=False)
        indexes, skip = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        return dict(q_dec=q, rho=rho, mask_0=m0, mask_1=m1, means=means,
                    scales_r=scales_r, skip_cond=skip, indexes=indexes)

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        from src.layers.cuda_inference import combine_for_reading_2x, restore_y_2x, add_and_multiply
        dtype, device = common_params.dtype, common_params.device
        symbol = self.gaussian_encoder.get_y(infos['scales_r'].shape, infos['scales_r'].numel(),
                                             dtype, device, infos['skip_cond'], infos['indexes'])
        symbol0_read=symbol.clone()
        u0 = restore_y_2x(symbol, infos['means'], infos['mask_0'])
        scales, means = self.second_prior(u0, common_params, y_spatial_prior, infos['rho'])
        sr = combine_for_reading_2x(scales, infos['mask_1'], inplace=False)
        symbol1 = self.gaussian_encoder.decode_and_get_y(sr, dtype, device)
        u1 = restore_y_2x(symbol1, means, infos['mask_1'])
        result = add_and_multiply(u0, u1, infos['q_dec'])
        self.decoding_trace = {'qmap': infos['q_dec'], 'rho': infos['rho'], 'y_hat': result,'q0_read':symbol0_read,'q1_read':symbol1.clone()}
        return result


def upgrade(model, branch):
    assert type(model) is DMC
    model.__class__ = SkipDomainDMC
    return model.install_v2(branch)

# v3 keeps the H11 math above. No V is installed in any condition.
PHI_SCOPE=('hyper_dec','temporal_prior_encoder','y_prior_fusion','y_spatial_prior','bit_estimator_z')

def _half_forward(module):
    """FP32 master weights, native half-weight forward, differentiable cast VJP."""
    import types
    original=module.forward
    module._phi_functional_inside=False
    def forward(self,*args,**kwargs):
        if self._phi_functional_inside:return original(*args,**kwargs)
        self._phi_functional_inside=True
        try:
            values={n:p.to(torch.float16) for n,p in self.named_parameters()}
            values.update({n:b.to(torch.float16) if b.is_floating_point() else b for n,b in self.named_buffers()})
            return torch.func.functional_call(self,values,args,kwargs,strict=True)
        finally:self._phi_functional_inside=False
    module.forward=types.MethodType(forward,module)

def install_phi(pm,original_state):
    from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
    # v2 ran the unfused torch fallback. No cached CUDA proxies may replace live weights.
    assert not CUSTOMIZED_CUDA_INFERENCE,'Requires the same unfused runtime verified in v2'
    for name in PHI_SCOPE:
        m=getattr(pm,name).float()
        sd={k[len(name)+1:]:v for k,v in original_state.items() if k.startswith(name+'.')}
        m.load_state_dict(sd,strict=True);m.requires_grad_(True)
        if name=='bit_estimator_z':
            for child in (m.f1,m.f2,m.f3,m.f4):_half_forward(child)
        else:_half_forward(m)
    return pm

def upgrade(model,branch):
    assert type(model) is DMC
    assert branch in ('N0','N1','Q0','Q1')
    model.__class__=SkipDomainDMC
    model.install_v2('H11' if branch.startswith('Q') else 'A')
    model.fix_skip=True;model.domain=True;model.logical_branch=branch
    return model
