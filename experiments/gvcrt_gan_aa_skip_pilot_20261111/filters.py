"""FP32 Fourier resampling on integer-origin grids; even Nyquist is split/merged.

The finite reflected halo is part of the registered operator, not an
equivariance guarantee. All cutoffs are cycles per current grid pixel.
"""
import torch
import torch.nn.functional as F
from torch import nn


def resample_axis(x, size, axis):
    n = x.shape[axis]
    if n == size:
        return x
    # Full complex FFT avoids an implicit choice of a Nyquist phase.
    spectrum = torch.fft.fft((x.float() if not x.is_complex() else x).contiguous(), dim=axis)
    shape = list(spectrum.shape)
    shape[axis] = size
    result = spectrum.new_zeros(shape)
    k = min(n, size)
    def sl(a, b):
        s = [slice(None)] * x.ndim
        s[axis] = slice(a, b)
        return tuple(s)
    result[sl(0, (k + 1)//2)] = spectrum[sl(0, (k + 1)//2)]
    if (k-1)//2:
        result[sl(-((k-1)//2), None)] = spectrum[sl(-((k-1)//2), None)]
    if k % 2 == 0:
        if size > n:
            value = spectrum[sl(k//2, k//2+1)] * .5
            result[sl(k//2, k//2+1)] = value
            result[sl(size-k//2, size-k//2+1)] = value
        else:
            result[sl(k//2, k//2+1)] = (spectrum[sl(k//2, k//2+1)] +
                spectrum[sl(n-k//2, n-k//2+1)])
    return torch.fft.ifft(result, dim=axis) * (size/n)


def resize_periodic(x, shape):
    y = resample_axis(resample_axis(x.float(), shape[0], -2), shape[1], -1)
    return y.real


def lowpass_periodic(x, cutoff=.25):
    h, w = x.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device).abs()
    fx = torch.fft.fftfreq(w, device=x.device).abs()
    mask = (fy[:, None] <= cutoff) & (fx[None, :] <= cutoff)
    return torch.fft.ifft2(torch.fft.fft2(x.float()) * mask).real


def reflected(x):
    p = min(16, x.shape[-2]-1, x.shape[-1]-1)
    return F.pad(x.float(), (p,p,p,p), mode='reflect') if p else x.float(), p


def lowpass(x):
    value, p = reflected(x)
    value = lowpass_periodic(value, .25)
    return value[..., p:-p, p:-p] if p else value


class FilteredWSiLU(nn.Module):
    def forward(self, x):
        value, p = reflected(x)
        shape = value.shape[-2:]
        up = resize_periodic(value, (2*shape[0], 2*shape[1]))
        activated = up * torch.sigmoid(4*up)
        # Resampling merges both Nyquist frequencies and rejects the rest.
        result = resize_periodic(activated, shape)
        return result[..., p:-p, p:-p] if p else result


def replace_wsilus(module):
    changed = []
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == 'WSiLU':
            setattr(module, name, FilteredWSiLU())
            changed.append(name)
        else:
            changed.extend(name+'.'+n for n in replace_wsilus(child))
    return changed


def check(device='cpu'):
    records = []
    for h, w in [(31,32),(32,33),(7,9),(32,32)]:
        x = torch.ones(1,2,h,w,device=device,requires_grad=True)*1.25
        up = resize_periodic(x, (2*h,2*w))
        down = resize_periodic(up,(h,w))
        e = float((down-x).abs().max())
        dc = float((lowpass(x)-x).abs().max())
        activation = FilteredWSiLU()(x)
        ae = float((activation-x*torch.sigmoid(4*x)).abs().max())
        assert max(e,dc,ae) < 3e-6
        gradient = torch.autograd.grad(activation.sum(), x)[0]
        assert torch.isfinite(gradient).all() and gradient.norm()>0
        records.append(dict(shape=[h,w],halo=min(16,h-1,w-1),DC_error=dc,roundtrip=e,activation_DC_error=ae))
    errors = []
    n=64
    coordinates=torch.arange(n,device=device).float()
    for k in [1,8,16,17,24,32]:
        x=torch.cos(2*torch.pi*k*coordinates/n)[None,None,None,:].expand(1,1,n,n).contiguous()
        y=lowpass_periodic(x,.25)
        error=float((y-(x if k<=16 else 0)).abs().max())
        assert error < 1e-5
        if k<32:
            up=resize_periodic(x,(128,128))
            expected=torch.cos(2*torch.pi*k*torch.arange(128,device=device)/128)[None,None,None,:]
            phase=float((up-expected).abs().max());assert phase<1e-5
        else:phase=None
        errors.append(dict(frequency_cycles_per_pixel=k/n,lowpass_max_error=error,interpolation_origin_phase_error=phase,cutoff_inclusive=k==16))
    # A real even-grid Nyquist cosine must interpolate to cos(pi*t/2).
    x=(-1.)**coordinates
    up=resize_periodic(x[None,None,None,:].expand(1,1,n,n),(128,128))
    error=float((up-torch.cos(torch.pi*torch.arange(128,device=device)/2)[None,None,None,:]).abs().max())
    assert error<2e-5
    return dict(status='PASS',precision='FP32',cutoff=.25,DC_shapes=records,frequency_phase=errors,Nyquist_interpolation_error=error)
