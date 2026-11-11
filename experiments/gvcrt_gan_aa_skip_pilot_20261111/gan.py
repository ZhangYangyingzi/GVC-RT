"""Spatial spectral-normalized PatchGAN, hinge losses and isolated G/D RNG."""
import torch
from torch import nn


class PatchGAN(nn.Sequential):
    def __init__(self):
        layers=[]
        channels=[3,64,128,256,512,1]
        for i,stride in enumerate([2,2,2,1,1]):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(channels[i],channels[i+1],4,stride,padding=1)))
            if i<4:layers.append(nn.LeakyReLU(.2))
        super().__init__(*layers)


def discriminator(seed,device):
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed)
        d=PatchGAN().to(device)
    return d,torch.optim.Adam(d.parameters(),lr=2e-4,betas=(0.,.9))


def g_mode(d):
    d.eval().requires_grad_(False)


def d_mode(d):
    d.train().requires_grad_(True)


def d_loss(d,real,fake):
    a=d(real*2-1);b=d(fake.detach()*2-1)
    return .5*(torch.relu(1-a).mean()+torch.relu(1+b).mean()),a.detach().mean(),b.detach().mean()


def adv(d,fake):return -d(fake*2-1).mean()
