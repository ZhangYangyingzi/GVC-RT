"""Preserve FP32 parameters across parent half()/to(), without a lossy round-trip."""
import torch
from torch import nn

class FP32Island(nn.Module):
    def _apply(self,fn,recurse=True):
        saved={n:p.detach() for n,p in self.named_parameters()}
        gradients={n:p.grad.detach() for n,p in self.named_parameters() if p.grad is not None}
        buffers={n:b.detach() for n,b in self.named_buffers() if b.is_floating_point()}
        result=super()._apply(fn,recurse=recurse)
        for name,p in self.named_parameters():
            if p.dtype!=torch.float32:p.data=saved[name].to(device=p.device,dtype=torch.float32)
            if name in gradients and p.grad is not None and p.grad.dtype!=torch.float32:
                p.grad=gradients[name].to(device=p.device,dtype=torch.float32)
        for name,old in buffers.items():
            parent,_,leaf=name.rpartition('.')
            module=self.get_submodule(parent) if parent else self
            now=module._buffers[leaf]
            if now.dtype!=torch.float32:module._buffers[leaf]=old.to(device=now.device,dtype=torch.float32)
        assert all(p.dtype==torch.float32 for p in self.parameters())
        return result
