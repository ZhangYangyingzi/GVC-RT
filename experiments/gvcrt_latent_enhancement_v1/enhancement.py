"""P-frame display-only latent correction and an isolated factorized entropy model."""
import struct
import zlib
import bisect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.entropy_models import EntropyCoder


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                 nn.SiLU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class Analysis(nn.Module):
    def __init__(self, latent_channels, hidden_channels, enhancement_channels, grid_hw, blocks):
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        h = hidden_channels
        stem = []
        for cin in [3, h, h, h]:
            stem += [nn.Conv2d(cin, h, 3, stride=2, padding=1), nn.SiLU()]
        self.stem = nn.Sequential(*stem)
        self.fuse = nn.Sequential(nn.Conv2d(h + latent_channels, h, 1), nn.SiLU(),
                                  *[Residual(h) for _ in range(blocks)],
                                  nn.Conv2d(h, enhancement_channels, 1))

    def forward(self, x, ell):
        source = F.interpolate(self.stem(x), size=ell.shape[-2:], mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(self.fuse(torch.cat((source, ell), 1)), self.grid_hw)


class Synthesis(nn.Module):
    def __init__(self, latent_channels, hidden_channels, enhancement_channels, grid_hw, blocks):
        super().__init__()
        h = hidden_channels
        self.net = nn.Sequential(nn.Conv2d(latent_channels + enhancement_channels, h, 3, padding=1),
                                 nn.SiLU(), *[Residual(h) for _ in range(blocks)])
        self.head = nn.Conv2d(h, latent_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, u, ell):
        u = F.interpolate(u, size=ell.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.net(torch.cat((u, ell), 1)))


class Enhancement(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.encoder = Analysis(**architecture)
        self.decoder = Synthesis(**architecture)

    def forward(self, x, ell, delta=None):
        u = self.encoder(x, ell)
        symbols = None
        if delta is not None:
            raw = u / delta
            symbols = raw + (raw.round() - raw).detach() if self.training else raw.round()
            u = symbols * delta
        return ell + self.decoder(u, ell), symbols


class BaseOnly(nn.Module):
    """Comparable decoder capacity; receives only shared base latent."""
    def __init__(self, architecture):
        super().__init__()
        a = dict(architecture, enhancement_channels=0)
        self.decoder = Synthesis(**a)

    def forward(self, ell):
        return ell + self.decoder.head(self.decoder.net(ell))


def display_reconstruction(generator, ell_base, q_recon, correction=None):
    # None explicitly bypasses every enhancement operation, including the decoder.
    if correction is None:
        return generator(ell_base, q_recon)
    return generator(ell_base + correction, q_recon)


class FactorizedEntropy(nn.Module):
    """Channel-wise logistic with real 32-bit arithmetic coding, independent state.

    V1 accepts int8. Out-of-range integers raise instead of clipping.
    All 256 integers have positive mass; tails are folded into endpoint bins.
    CDFs are regenerated on CPU from the shared entropy model, never source data.
    """
    HEADER = struct.Struct("<4sBBBBHHfII")

    def __init__(self, channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def bits(self, symbols, delta):
        scale = F.softplus(self.log_scale) + 1e-4
        lo = torch.sigmoid((symbols - .5) * delta / scale)
        hi = torch.sigmoid((symbols + .5) * delta / scale)
        p = torch.where(symbols <= -128, hi, torch.where(symbols >= 127, 1-lo, hi-lo))
        return -p.clamp_min(1e-9).log2().sum()

    def coder(self, delta):
        scale = F.softplus(self.log_scale.detach().cpu().double()).reshape(-1, 1) + 1e-4
        s = torch.arange(-128, 128, dtype=torch.float64)[None]
        lo, hi = torch.sigmoid((s-.5)*delta/scale), torch.sigmoid((s+.5)*delta/scale)
        pmf = hi-lo
        pmf[:, 0], pmf[:, -1] = hi[:, 0], 1-lo[:, -1]
        # Reuse the existing PMF->CDF component. Native rANS allocates only N bytes
        # (rans.cpp:221), unsafe for small or high-entropy N-symbol streams.
        # A bounded-memory reference arithmetic coder avoids modifying that extension.
        cdf = torch.stack([EntropyCoder.pmf_to_quantized_cdf(p) for p in pmf])
        fingerprint = zlib.crc32(cdf.numpy().tobytes())
        return cdf.tolist(), fingerprint

    def encode(self, symbols, delta, level):
        if symbols.ndim != 4 or symbols.shape[0] != 1 or symbols.shape[1] != self.log_scale.shape[1]:
            raise ValueError("Expected one NCHW enhancement grid")
        if not torch.isfinite(symbols).all() or not torch.equal(symbols, symbols.round()):
            raise ValueError("Non-finite or non-integer symbols")
        if symbols.min() < -128 or symbols.max() > 127:
            raise OverflowError(f"Native int8 symbol range exceeded: {symbols.min().item()}, {symbols.max().item()}")
        if not 0 < delta < float("inf"):
            raise ValueError("Invalid delta")
        _, c, h, w = symbols.shape
        cdfs, fingerprint = self.coder(delta)
        payload = arithmetic_encode((symbols.cpu().to(torch.int16).reshape(-1)+128).tolist(), cdfs, h*w)
        return self.HEADER.pack(b"GLE1", 1, c, level, 0, h, w, delta, fingerprint, len(payload)) + payload

    def decode(self, stream, device="cpu"):
        if len(stream) < self.HEADER.size:
            raise ValueError("Truncated header")
        magic, version, c, level, reserved, h, w, delta, fingerprint, length = self.HEADER.unpack_from(stream)
        if magic != b"GLE1" or version != 1 or reserved or c != self.log_scale.shape[1] or not 0 < h*w <= 65536:
            raise ValueError("Invalid enhancement header")
        if len(stream) != self.HEADER.size+length or not 0 < delta < float("inf"):
            raise ValueError("Invalid enhancement payload")
        cdfs, expected = self.coder(delta)
        if fingerprint != expected:
            raise ValueError("Entropy CDF/model mismatch")
        raw = arithmetic_decode(stream[self.HEADER.size:], cdfs, h*w, c*h*w)
        symbols = (torch.tensor(raw, device=device, dtype=torch.int16)-128).reshape(1,c,h,w)
        return symbols, delta, level


def arithmetic_encode(symbols, cdfs, per_channel):
    low, high, pending = 0, (1<<32)-1, 0
    bits = []
    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1-bit]*pending)
        pending = 0
    for index, symbol in enumerate(symbols):
        cdf = cdfs[index//per_channel]
        span = high-low+1
        high = low + span*cdf[symbol+1]//65536-1
        low += span*cdf[symbol]//65536
        while True:
            if high < (1<<31):
                emit(0)
            elif low >= (1<<31):
                emit(1)
                low -= 1<<31
                high -= 1<<31
            elif low >= (1<<30) and high < (3<<30):
                pending += 1
                low -= 1<<30
                high -= 1<<30
            else:
                break
            low *= 2
            high = high*2+1
    pending += 1
    emit(0 if low < (1<<30) else 1)
    bits += [0]*32  # explicit termination lookahead, included in actual rate
    bits += [0]*((-len(bits))%8)
    return bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))


def arithmetic_decode(payload, cdfs, per_channel, count):
    position = 0
    def bit():
        nonlocal position
        if position >= len(payload)*8:
            raise ValueError("Truncated arithmetic stream")
        b = (payload[position//8]>>(7-position%8))&1
        position += 1
        return b
    low, high, value = 0, (1<<32)-1, 0
    for _ in range(32):
        value = value*2+bit()
    symbols = []
    for index in range(count):
        cdf = cdfs[index//per_channel]
        span = high-low+1
        scaled = ((value-low+1)*65536-1)//span
        symbol = bisect.bisect_right(cdf,scaled)-1
        if not 0 <= symbol < 256:
            raise ValueError("Invalid arithmetic symbol")
        high = low + span*cdf[symbol+1]//65536-1
        low += span*cdf[symbol]//65536
        symbols.append(symbol)
        while True:
            if high < (1<<31):
                pass
            elif low >= (1<<31):
                low -= 1<<31
                high -= 1<<31
                value -= 1<<31
            elif low >= (1<<30) and high < (3<<30):
                low -= 1<<30
                high -= 1<<30
                value -= 1<<30
            else:
                break
            low *= 2
            high = high*2+1
            value = value*2+bit()
    return symbols
