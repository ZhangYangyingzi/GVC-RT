"""Decoder-side visual latent residual; all conditioning is receiver available."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.expand = nn.Conv2d(64, 128, 1)
        self.reduce = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        return x + self.reduce(F.silu(self.expand(self.depthwise(x))))


class Adapter(nn.Module):
    def __init__(self, feature_channels=256, latent_channels=18, quant_channels=128):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(feature_channels * 4, 64, 1), nn.SiLU())
        self.latent = nn.Sequential(nn.Conv2d(latent_channels, 16, 1), nn.SiLU())
        self.quant = nn.Sequential(nn.Conv2d(quant_channels, 16, 1), nn.SiLU())
        self.fuse = nn.Conv2d(96, 64, 1)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(3)))
        self.output = nn.Conv2d(64, latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, f, u, d):
        assert f.shape[-2:] == tuple(2 * k for k in u.shape[-2:])
        assert d.shape[-2:] == u.shape[-2:]
        h = torch.cat((self.feature(F.pixel_unshuffle(f.float(), 2)),
                       self.latent(u.float()), self.quant(d.float())), 1)
        return self.output(self.blocks(self.fuse(h)))


def condition(q_actual, branch):
    if branch == 'R':
        return torch.zeros_like(q_actual, dtype=torch.float32)
    if branch == 'RQ':
        assert bool((q_actual >= .5).all())
        return q_actual.float().log()
    raise ValueError(branch)


def bridge(wrapper, f):
    # Mirrors PretrainedReconWrapper.forward exactly up to its generator call.
    z = F.pixel_unshuffle(f.float(), 2)
    z = wrapper.mlp[0](z)
    z = wrapper.mlp[1](z)
    z = wrapper.mlp[2](z)
    z = z * torch.sigmoid(z)
    return wrapper.mlp[3](z)
