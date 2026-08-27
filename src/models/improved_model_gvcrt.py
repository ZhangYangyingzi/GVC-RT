import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 use_conv_shortcut=False):
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)

        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return x + residual

class ResBlock_Out(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 use_conv_shortcut=False):
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        residual = x
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)

        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
                 resolution, double_z=False):
        super().__init__()
        self.num_res_blocks = 4
        ch = 128
        self.num_blocks = len(ch_mult)

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=(3, 3), padding=1, bias=False)

        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)
            self.down.append(down)

        self.mid_block = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))

        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = x * torch.sigmoid(x)
        x = self.conv_out(x)
        return x

from ..layers.layers import  DepthConvBlock
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, bias_pixel_shuffle_8

import torch
import torch.nn as nn
import torch.nn.functional as F

class StageBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks):
        super().__init__()
        num_norms = num_blocks - 1
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_ch = in_ch
        for i in range(num_blocks):
            block_out_ch = out_ch if i == 0 else out_ch

            self.blocks.append(DepthConvBlock(current_ch, block_out_ch))

            if i < num_norms:
                self.norms.append(nn.GroupNorm(32, block_out_ch, eps=1e-6))

            current_ch = block_out_ch

    def forward(self, x, quant_step=None):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)
        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step=None):
        if quant_step is None:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i < len(self.norms):
                    x = self.norms[i](x)
            return x
        else:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i < len(self.norms):
                    x = self.norms[i](x)
            return x * quant_step

    def forward_cuda(self, x, quant_step=None):
        x = self.blocks[0](x)
        x = self.norms[0](x)
        x = self.blocks[1](x)
        x = self.norms[1](x)
        x = self.blocks[2](x)
        x = self.norms[2](x)
        if quant_step is not None:
            return self.blocks[3](x, quant_step=quant_step)
        return self.blocks[3](x)

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
                 resolution, double_z=False) -> None:
        super().__init__()
        self.num_blocks = len(ch_mult)
        block_in = ch * ch_mult[self.num_blocks - 1]
        g_ch_recon = 320
        g_ch_src_d = 3 * 8 * 8  # 192 channels

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, padding=1)

        self.ada1 = AdaptiveGroupNorm(z_channels, block_in)
        self.stage1 = StageBlock(block_in, block_in, num_blocks=4)

        self.ada2 = AdaptiveGroupNorm(z_channels, block_in)
        self.stage2 = StageBlock(block_in, block_in, num_blocks=4)

        self.ada3 = AdaptiveGroupNorm(z_channels, block_in)
        self.upsample = Upsampler(block_in)
        self.stage3 = StageBlock(block_in, g_ch_recon, num_blocks=4)

        self.ada4 = AdaptiveGroupNorm(z_channels, g_ch_recon)
        self.stage4 = StageBlock(g_ch_recon, g_ch_recon, num_blocks=4)

        self.ada_final = AdaptiveGroupNorm(z_channels, g_ch_recon)
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, kernel_size=3, padding=1)

    def forward(self, x, quant_step=None):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)
        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv_in(x)

        out = self.ada1(out, x)
        out = self.stage1(out)

        out = self.ada2(out, x)
        out = self.stage2(out)

        out = self.ada3(out, x)
        out = self.upsample(out)
        out = self.stage3(out)

        out = self.ada4(out, x)
        out = self.stage4(out, quant_step)

        out = self.ada_final(out, x)
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)

        out = torch.clamp(out, -1., 1.)
        return out

    def forward_cuda(self, x, quant_step):
        out = self.conv_in(x)

        out = self.ada1(out, x)
        out = self.stage1(out)

        out = self.ada2(out, x)
        out = self.stage2(out)

        out = self.ada3(out, x)
        out = self.upsample(out)
        out = self.stage3(out)

        out = self.ada4(out, x)
        out = self.stage4(out, quant_step)

        out = self.ada_final(out, x)

        out = F.conv2d(out, self.head.weight, padding=1)
        return bias_pixel_shuffle_8(out, self.head.bias)

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    c, h, w = x.shape[-3:]
    s = block_size ** 2
    outer_dims = x.shape[:-3]
    x = x.view(-1, block_size, block_size, c // s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    return x

class Upsampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x
