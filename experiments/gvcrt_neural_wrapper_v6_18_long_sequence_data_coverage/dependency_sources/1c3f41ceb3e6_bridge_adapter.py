from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, rank=37):
        super().__init__()
        self.down = nn.Conv2d(256, rank, 1)
        self.activation = nn.GELU()
        self.up = nn.Conv2d(rank, 256, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        return feature + self.up(self.activation(self.down(feature)))
