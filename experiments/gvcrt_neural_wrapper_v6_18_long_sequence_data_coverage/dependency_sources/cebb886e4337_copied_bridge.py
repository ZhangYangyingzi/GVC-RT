import copy

import torch
import torch.nn.functional as F
from torch import nn


class CopiedBridge(nn.Module):
    def __init__(self, official_bridge, train_bridge=True, adapter=None):
        super().__init__()
        self.bridge = copy.deepcopy(official_bridge).float()
        self.adapter = adapter
        self.bridge.requires_grad_(train_bridge)

    def forward(self, feature):
        adapted = self.adapter(feature) if self.adapter is not None else feature
        value = F.pixel_unshuffle(adapted, 2)
        value = self.bridge[0](value)
        value = self.bridge[1](value)
        value = self.bridge[2](value)
        value = value * torch.sigmoid(value)
        return self.bridge[3](value)

    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
