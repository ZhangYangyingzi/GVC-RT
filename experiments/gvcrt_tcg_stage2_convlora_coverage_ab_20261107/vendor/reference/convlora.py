"""Generator-only parallel low-rank updates; contains no Router or TCG modification.

Attach only AFTER strict original codec/pilot/B1500 loading has passed. The
generator's selected stage3 blocks and CUDA head entry explicitly use their
PyTorch operators, since the original fused proxies bypass additional modules.
The comparison baseline must use configure_safe_dispatch() too.
"""
import math,types
import torch
from torch import nn
from fp32_island import FP32Island


class LowRankConvBranch(FP32Island):
    def __init__(self,in_channels,out_channels,rank=4):
        super().__init__()
        assert rank==4
        self.down=nn.Conv2d(in_channels,rank,1,bias=False)
        self.up=nn.Conv2d(rank,out_channels,3,padding=1,bias=False)
        nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.rank=rank
        self.alpha=rank
        self.forward_calls=0

    def forward(self,x):
        assert self.down.weight.dtype==self.up.weight.dtype==torch.float32
        self.forward_calls+=1
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.up(self.down(x.float()))*(self.alpha/self.rank)


class ParallelConvLoRA(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__()
        assert isinstance(base,nn.Conv2d) and base.kernel_size==(3,3)
        assert base.stride==(1,1) and base.padding==(1,1) and base.dilation==(1,1)
        self.base=base.requires_grad_(False)
        self.lora=LowRankConvBranch(base.in_channels,base.out_channels,rank).to(device=base.weight.device)

    def forward(self,x):
        base=self.base(x)
        # Add in FP32 and cast once; the base convolution retains its protocol dtype.
        return (base.float()+self.lora(x)).to(base.dtype)


def _safe_block_forward(self,x,quant_step=None,to_cat=None,cat_at_front=True):
    return self.forward_torch(x,quant_step,to_cat,cat_at_front)


def _safe_generator_cuda(self,x,quant_step=None,use_tcg=True,update_buffer=True):
    return self.forward_torch(x,quant_step,use_tcg=use_tcg,update_buffer=update_buffer)


def configure_safe_dispatch(generator):
    """Same local fallback for both baseline and candidate, no global codec flag."""
    assert len(generator.stage3.blocks)==4
    for block in generator.stage3.blocks:
        block.forward=types.MethodType(_safe_block_forward,block)
        block.forward_cuda=types.MethodType(_safe_block_forward,block)
        block.proxy=None
    generator.forward_cuda=types.MethodType(_safe_generator_cuda,generator)
    generator.convlora_dispatch='stage3 DepthConvBlock and generator head use PyTorch operators on CPU/CUDA; other codec paths unchanged'


def attach_generator_convlora(generator,seed=20261106):
    assert not hasattr(generator.head,'lora'),'LoRA already attached'
    generator.requires_grad_(False)
    devices=sorted({p.device.index for p in generator.parameters() if p.is_cuda})
    targets=[]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for i,block in enumerate(generator.stage3.blocks):
            conv=block.dc[2]
            assert isinstance(conv,nn.Conv2d) and conv.in_channels==conv.out_channels==conv.groups==320
            targets.append(dict(path=f'stage3.blocks.{i}.dc.2',kind='parallel low-rank update alongside existing depthwise3x3; not a dense-layer weight factorization',
                in_channels=320,out_channels=320,groups=320))
            block.dc[2]=ParallelConvLoRA(conv)
        head=generator.head
        assert isinstance(head,nn.Conv2d) and head.in_channels==320 and head.out_channels==192 and head.groups==1
        targets.append(dict(path='head',kind='parallel low-rank update alongside original dense3x3',in_channels=320,out_channels=192,groups=1))
        generator.head=ParallelConvLoRA(head)
    configure_safe_dispatch(generator)
    named=[(n,p) for n,p in generator.named_parameters() if p.requires_grad]
    assert all('.lora.' in n for n,p in named) and len(named)==10
    assert sum(p.numel() for n,p in named)==59392
    assert all(p.dtype==torch.float32 for n,p in named)
    assert not any(p.requires_grad for name,p in generator.named_parameters() if name.startswith(('tcg_low.','tcg_mid.','tcg_out.')))
    return dict(targets=targets,rank=4,alpha_div_rank=1.,parameter_count=59392,bias_parameters=False,
        head_geometry_note='Head output has192 channels; its branch is320->4->192, matching the actual frozen head output',
        precision='LoRA FP32 internally; add to original-conv output in FP32 and cast back once',dispatch=generator.convlora_dispatch)


def lora_state(generator):
    return {n:t for n,t in generator.state_dict().items() if '.lora.' in n}
