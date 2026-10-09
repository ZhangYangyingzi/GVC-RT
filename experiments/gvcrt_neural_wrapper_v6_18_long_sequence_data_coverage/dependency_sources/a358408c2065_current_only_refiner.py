from torch import nn

from temporal_refiner import ResidualBlock


class CurrentOnlyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(18, 69, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(69, groups=3) for _ in range(3)])
        self.correction_head = nn.Conv2d(69, 18, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, current, previous=None):
        del previous
        delta = self.correction_head(self.blocks(self.input_projection(current)))
        return {"codeword": current + delta, "delta": delta}
