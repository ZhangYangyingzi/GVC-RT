import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class NeuralWrapper(nn.Module):
    def __init__(self, channels=64, blocks=5):
        super().__init__()
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return torch.clamp(x + self.output(self.blocks(self.input(x))), 0.0, 1.0)

