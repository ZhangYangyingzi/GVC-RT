"""FP32 causal three-frame residual after ada_final, before the existing head."""
import types
import torch
from torch import nn
import torch.nn.functional as F
from fp32_island import FP32Island


class CausalAdapter(FP32Island):
    def __init__(self, channels=320, rank=32):
        super().__init__()
        self.A = nn.Conv2d(channels, rank, 1, bias=False)
        self.temporal = nn.Conv3d(rank, rank, (3, 3, 3), padding=(0, 1, 1), groups=rank, bias=True)
        self.B = nn.Conv2d(rank, channels, 1, bias=True)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)
        self.history = []
        self.forward_calls = 0
        self.last_shape = None
        assert sum(p.numel() for p in self.parameters()) == 21696

    def reset(self):
        self.history = []

    def from_three(self, history, output_dtype=None):
        assert len(history) == 3 and all(x.shape == history[0].shape for x in history)
        assert all(p.dtype == torch.float32 for p in self.parameters())
        self.forward_calls += 1
        self.last_shape = list(history[-1].shape)
        with torch.autocast(device_type=history[-1].device.type, enabled=False):
            # Project afresh. Only raw, detached pre-adapter h may be cached.
            u = torch.stack([F.silu(self.A(h.detach().float())) for h in history], dim=2)
            v = F.silu(self.temporal(u))
            assert v.shape[2] == 1
            delta = self.B(v.squeeze(2))
            result = history[-1].detach().float() + delta
        return result.to(output_dtype or history[-1].dtype)

    def forward(self, h, update=True):
        past = self.history[-2:] + [h.detach()]
        while len(past) < 3:
            past.insert(0, past[0])
        result = self.from_three(past, h.dtype)
        if update:
            self.history = (self.history + [h.detach()])[-2:]
        return result

    def history_bytes(self):
        return sum(h.numel() * h.element_size() for h in self.history)


def attach(generator, branch):
    assert not hasattr(generator, 'causal_adapter')
    assert generator.head.base.in_channels == 320
    assert hasattr(generator.head, 'lora')
    generator.causal_adapter = branch
    original_forward = generator.forward_torch
    original_reset = generator.reset_temporal_buffer

    def adapt(module, inputs, h):
        return generator.causal_adapter(h, update=generator._adapter_update)

    generator.ada_final.register_forward_hook(adapt)

    def forward(self, x, quant_step=None, use_tcg=True, update_buffer=True):
        self._adapter_update = update_buffer
        return original_forward(x, quant_step, use_tcg=use_tcg, update_buffer=update_buffer)

    def reset(self):
        original_reset()
        self.causal_adapter.reset()

    # Explicitly cover all deployment entry points, including the fused CUDA entry.
    for name in ('forward', 'forward_torch', 'forward_cuda'):
        setattr(generator, name, types.MethodType(forward, generator))
    generator.reset_temporal_buffer = types.MethodType(reset, generator)
    return generator


def decode_head(head, adapted):
    dtype = head.base.weight.dtype
    return F.pixel_shuffle(head(adapted.to(dtype)), 8).clamp(-1, 1)
