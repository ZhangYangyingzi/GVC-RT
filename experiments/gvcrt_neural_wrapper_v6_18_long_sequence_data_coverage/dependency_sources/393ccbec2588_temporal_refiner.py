import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()

    def forward(self, value):
        return value + self.pointwise(self.activation(self.norm(self.depthwise(value))))


class TemporalRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(54, 64, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(3)])
        self.correction_head = nn.Conv2d(64, 18, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, current, previous):
        inputs = torch.cat((current, previous, current - previous), dim=1)
        delta = self.correction_head(self.blocks(self.input_projection(inputs)))
        return {"codeword": current + delta, "delta": delta}
