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


def upgrade(model, quant=False, condition=False):
    assert type(model) is DMC
    model.__class__ = ExperimentalDMC
    return model.install(quant, condition)
