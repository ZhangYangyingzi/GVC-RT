"""Router reads only current/previous decoded codec features; M suppresses history."""
import math
import torch
from torch import nn

class SpatialTCGRouter(nn.Module):
    def __init__(self,channels,seed=20261105):
        super().__init__()
        # CPU-local seeding preserves the caller's CPU and CUDA RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net=nn.Sequential(nn.Conv2d(3*channels,32,1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1),nn.Sigmoid())
        nn.init.zeros_(self.net[4].weight)
        nn.init.constant_(self.net[4].bias,math.log(.01/.99))
        self.channels=channels
        self.mode='enabled'
        self._previous=None
        self.last_mask=None
        self.last_input=None
        self.last_had_history=False
    def reset_history(self):
        self._previous=None;self.last_mask=None;self.last_input=None;self.last_had_history=False
    def mask(self,feature,update_buffer):
        current=feature.detach().float()
        previous=self._previous
        compatible=previous is not None and previous.shape==current.shape and previous.device==current.device
        reset_needed=previous is not None and not compatible
        if not compatible:previous=torch.zeros_like(current)
        if self.mode=='bypass':mask=None
        elif self.mode=='zero':mask=torch.zeros_like(current[:,:1])
        elif self.mode=='one':mask=torch.ones_like(current[:,:1])
        else:
            assert self.mode=='enabled'
            assert next(self.parameters()).dtype==torch.float32
            mask=self.net(torch.cat((current,previous,(current-previous).abs()),dim=1))
        if update_buffer:
            self.last_mask=None if mask is None else mask.detach()
            self.last_input=current.detach()
            self.last_had_history=compatible
        return mask,reset_needed
    def commit(self,feature):
        self._previous=feature.detach().float().clone()

def attach_router(pm,mode='enabled'):
    assert not hasattr(pm.recon_generation_net,'router')
    channels=pm.dec.conv2.out_channels
    router=SpatialTCGRouter(channels).to(next(pm.parameters()).device).float()
    router.mode=mode
    pm.recon_generation_net.router=router
    pm.recon_generation_net.reset_generator_buffer()
    return router
