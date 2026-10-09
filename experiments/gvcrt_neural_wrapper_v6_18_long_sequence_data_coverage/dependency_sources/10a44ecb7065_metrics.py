import math

import torch
import torch.nn.functional as F


def unit(value):
    return ((value[:, :, :1080, :1920].float().clamp(-1, 1) + 1) / 2).clamp(0, 1)


def spatial_metrics(output, reference, lpips_model, dists_model):
    mse = float((output - reference).square().mean())
    return {"psnr": 99.9 if mse <= 1e-10 else -10 * math.log10(mse),
            "lpips": float(lpips_model(output, reference, normalize=True)),
            "dists": float(dists_model(output, reference))}


def temporal_delta_l1(current, previous, reference, previous_reference):
    return float(((current - previous) - (reference - previous_reference)).abs().mean())


def torch_msssim_rgb(outputs, reference):
    first, second = outputs.float(), reference.float().expand_as(outputs)
    coordinates = torch.arange(-5, 6, device=first.device, dtype=first.dtype)
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
    window = (gaussian[:, None] * gaussian[None, :])
    window = (window / window.sum()).expand(3, 1, 11, 11)
    mssim, mcs = [], []
    for _ in range(5):
        mu1, mu2 = F.conv2d(first, window, groups=3), F.conv2d(second, window, groups=3)
        mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1 = F.conv2d(first.square(), window, groups=3) - mu1_sq
        sigma2 = F.conv2d(second.square(), window, groups=3) - mu2_sq
        sigma12 = F.conv2d(first * second, window, groups=3) - mu12
        mssim.append((((2 * mu12 + 0.01 ** 2) * (2 * sigma12 + 0.03 ** 2)) /
                      ((mu1_sq + mu2_sq + 0.01 ** 2) * (sigma1 + sigma2 + 0.03 ** 2))).mean((-2, -1)))
        mcs.append(((2 * sigma12 + 0.03 ** 2) /
                    (sigma1 + sigma2 + 0.03 ** 2)).mean((-2, -1)))
        first = F.avg_pool2d(first, 2, 2, ceil_mode=True)
        second = F.avg_pool2d(second, 2, 2, ceil_mode=True)
    weights = first.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:, None, None]
    return (torch.prod(torch.stack(mcs)[:-1].clamp_min(0) ** weights[:-1], dim=0) *
            torch.stack(mssim)[-1].clamp_min(0) ** weights[-1]).mean(1)
