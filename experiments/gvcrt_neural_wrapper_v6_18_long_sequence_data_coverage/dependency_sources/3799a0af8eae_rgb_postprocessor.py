import torch
from torch import nn

from temporal_refiner import ResidualBlock


class RGBPostprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(9, 72, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(72) for _ in range(3)])
        self.correction_head = nn.Conv2d(72, 3, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, current, previous):
        inputs = torch.cat((current, previous, current - previous), dim=1)
        delta = self.correction_head(self.blocks(self.input_projection(inputs)))
        return {"image": (current + delta).clamp(-1, 1), "delta": delta}
