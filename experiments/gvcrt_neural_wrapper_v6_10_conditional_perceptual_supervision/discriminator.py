"""FP32 conditional PatchGAN, shared architecture for spatial/temporal supervision."""
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class ConditionalDiscriminator(nn.Module):
    def __init__(self,feature_channels):
        super().__init__()
        assert feature_channels%32==0
        self.project=nn.Sequential(nn.GroupNorm(32,feature_channels),
            spectral_norm(nn.Conv2d(feature_channels,32,1)),nn.LeakyReLU(.2))
        layers=[];channels=39
        for out,stride in zip((64,128,256,512),(2,2,2,1)):
            layers.extend([spectral_norm(nn.Conv2d(channels,out,4,stride,1)),nn.LeakyReLU(.2)])
            channels=out
        layers.append(spectral_norm(nn.Conv2d(channels,1,3,1,1)))
        self.main=nn.Sequential(*layers)
    def forward(self,images,condition,external_qp):
        assert images.dtype==condition.dtype==torch.float32
        assert not condition.requires_grad
        z=F.interpolate(self.project(condition),size=images.shape[-2:],mode='bilinear',align_corners=False)
        q=torch.full_like(images[:,:1],float(external_qp)/9)
        return self.main(torch.cat((images*2-1,z,q),dim=1))

def create(feature_channels,device,seed):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        result=ConditionalDiscriminator(feature_channels).to(device).float()
    return result
