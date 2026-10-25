"""Inference-only spatial multiplier ablations on the v2 codec."""
import torch

from runtime import SkipDomainDMC

ALPHAS = (0.8, 0.9, 1.0, 1.1, 1.25)
MODES = ('A', 'MAP', 'MEAN', 'ROLL', 'IDENTITY')


def transform_multiplier(raw, mode, alpha_index):
    assert mode in MODES and 0 <= alpha_index < len(ALPHAS)
    if mode == 'MAP': variant = raw
    elif mode == 'MEAN': variant = raw.mean(dim=(-2, -1), keepdim=True).expand_as(raw)
    elif mode == 'ROLL':
        blocks = (raw.shape[-2] // 4 // 2, raw.shape[-1] // 4 // 2)
        variant = torch.roll(raw, shifts=(4 * blocks[0], 4 * blocks[1]), dims=(-2, -1))
    elif mode == 'IDENTITY': variant = torch.ones_like(raw)
    else: variant = torch.ones_like(raw)
    effective = (ALPHAS[alpha_index] * variant).clamp(.5, 2.0)
    return variant, effective


class AblationDMC(SkipDomainDMC):
    mode = 'MAP'
    alpha_index = 2

    def configure(self, mode, alpha_index=2):
        assert mode in MODES and 0 <= alpha_index < len(ALPHAS)
        assert alpha_index == 2 or mode in ('MEAN', 'ROLL')
        self.mode, self.alpha_index = mode, alpha_index
        self.last_m_raw = self.last_m_variant = self.last_m_effective = None
        self.last_q_clamp_fraction = None
        self.last_roll_blocks = None
        return self

    def quant_step(self, params):
        base = params[:, :128].clamp_min(.5)
        if self.mode == 'A':
            return base
        assert self.use_quant and self.current_qp is not None
        raw = self.quant_adjust(params, self.current_qp)
        variant, effective = transform_multiplier(raw, self.mode, self.alpha_index)
        if self.mode == 'ROLL':
            self.last_roll_blocks = (raw.shape[-2] // 4 // 2, raw.shape[-1] // 4 // 2)
        self.last_m_raw, self.last_m_variant, self.last_m_effective = raw, variant, effective
        self.last_q_clamp_fraction = float((base.float() * effective < .5).float().mean())
        return (base.float() * effective).clamp_min(.5).to(base.dtype)


def upgrade_v3(model, mode, alpha_index=2):
    assert isinstance(model, SkipDomainDMC)
    model.__class__ = AblationDMC
    return model.configure(mode, alpha_index)
