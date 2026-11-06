import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V1 = REPO / "experiments/gvcrt_neural_wrapper_v1"
V9 = REPO / "expericent_generation_input/expericent_interface_causal_controls_v9/src"
for path in (REPO, V9):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gvc_hooks import load_models
from src.layers.cuda_inference import round_and_to_int8


def unit(x, h=None, w=None):
    value = (x.float().clamp(-1, 1) + 1) / 2
    return value if h is None else value[:, :, :h, :w]


def codec_input(x):
    h, w = x.shape[-2:]
    ph, pw = (64 - h % 64) % 64, (64 - w % 64) % 64
    return F.pad(x, (0, pw, 0, ph), mode="replicate").half() * 2 - 1


def named_hash(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_hash(module):
    return named_hash(module.state_dict().items())


def compression_hash(i_model, p_model):
    items = [(f"i.{n}", p) for n, p in i_model.named_parameters()]
    items += [(f"p.{n}", p) for n, p in p_model.named_parameters()
              if not n.startswith("recon_generation_net.")]
    return named_hash(items)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def quality_models(device):
    import lpips
    from DISTS_pytorch import DISTS
    return (lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False),
            DISTS().to(device).eval().requires_grad_(False))


def perceptual(output, target, quality):
    lpips_value = quality[0](output, target, normalize=True).mean()
    dists_value = quality[1](output, target).mean()
    return lpips_value + dists_value, lpips_value, dists_value


def load_train_pair(index, device):
    cache = REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11/cache/train_steps"
    payload = torch.load(cache / f"step_{index:04d}.pt", map_location="cpu", weights_only=True)
    pair = [unit(record["gt_rgb"].to(device).float()) for record in payload["pair"]]
    return pair, payload["plan"]


def establish_dpb(i_model, p_model, previous_proxy, requested_qp):
    p_model.clear_dpb()
    p_model.set_curr_poc(0)
    with torch.no_grad():
        encoded = i_model.compress(codec_input(previous_proxy), requested_qp)
        p_model.add_ref_frame(None, encoded["x_hat"])


def prepare_joint_models(device):
    i_model, p_model = load_models(device)
    i_model.requires_grad_(False)
    p_model.requires_grad_(False)
    bridge = p_model.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator = p_model.recon_generation_net.decoder.float().train().requires_grad_(True)
    return i_model, p_model, bridge, generator


def prepare_p(model, qp):
    qf = model.q_scale_feature[qp:qp + 1]
    qe = model.q_scale_enc[qp:qp + 1]
    qd = model.q_scale_dec[qp:qp + 1]
    qr = model.q_scale_recon[qp:qp + 1]
    feature = model.apply_feature_adaptor()
    context, temporal = model.feature_extractor(feature, qf)
    return context, temporal, qe, qd, qr


def joint_ste_forward(model, proxy, qp):
    context, temporal, qe, qd, qr = prepare_p(model, qp)
    y = model.enc(codec_input(proxy), context, qe)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hard = torch.clamp(torch.round(z), -128, 127)
    z_ste = z + (z_hard - z).detach()
    params = model.res_prior_param_decoder(z_ste, temporal)
    qdec, scale, mean = params.chunk(3, 1)
    qdec = torch.clamp_min(qdec, .5)
    y_scaled = y * torch.reciprocal(qdec)
    batch, channels, height, width = y_scaled.shape
    mask0, mask1 = model.get_mask_2x(batch, channels, height, width,
                                     y_scaled.dtype, y_scaled.device)

    def process(value, sigma, mu, mask):
        sigma, mu = sigma * mask, mu * mask
        residual = (value - mu) * mask
        hard = torch.clamp(torch.round(residual), -128, 127)
        active = mask.bool() & (sigma > .12)
        hard = hard * active
        ste = residual + (hard - residual).detach()
        return hard, ste, sigma, mu, active

    hard0, quant0, scale0, mean0, active0 = process(y_scaled, scale, mean, mask0)
    y_hat0 = quant0 + mean0
    scale1, mean1 = model.y_spatial_prior(torch.cat((y_hat0, params), 1)).chunk(2, 1)
    hard1, quant1, scale1, mean1, active1 = process(y_scaled, scale1, mean1, mask1)
    y_hat = (y_hat0 + quant1 + mean1) * qdec
    feature = model.dec(y_hat, context, qd)
    reconstruction = unit(model.recon_generation_net(feature.float(), qr.float()),
                          proxy.shape[-2], proxy.shape[-1])
    upper = model.bit_estimator_z.get_cdf(z_ste.float() + .5, qp)
    lower = model.bit_estimator_z.get_cdf(z_ste.float() - .5, qp)
    rate = (-torch.log2((upper - lower).clamp_min(1e-9))).sum()

    def gaussian_bits(quantized, sigma, active):
        quantized = quantized.float()[active]
        sigma = sigma.float()[active].clamp(.11, 16)
        index = ((torch.log(sigma) - model.gaussian_encoder.log_scale_min) *
                 model.gaussian_encoder.log_step_recip).long().clamp(0, 127)
        stable = model.gaussian_encoder.scale_table.to(sigma.device)[index]
        normal = torch.distributions.Normal(torch.zeros_like(stable), stable)
        return (-torch.log2((normal.cdf(quantized + .5) -
                            normal.cdf(quantized - .5)).clamp_min(1e-9))).sum()

    rate = rate + gaussian_bits(quant0, scale0, active0) + gaussian_bits(quant1, scale1, active1)
    with torch.no_grad():
        z_true, _ = round_and_to_int8(z)
        true_params = model.res_prior_param_decoder(z_true, temporal)
        _, _, _, _, true_y_hat = model.compress_prior_2x(y, true_params, model.y_spatial_prior)
        ste_difference = float((true_y_hat - y_hat.detach()).abs().max())
    return reconstruction, rate, ste_difference


def basic_metrics(output, target):
    mse = float((output - target).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-15))
    return psnr


def ms_ssim_rgb(a, b):
    channels = a.shape[1]
    axis = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    kernel_1d = torch.exp(-(axis * axis) / (2 * 1.5 * 1.5))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(channels, 1, 11, 11)
    weights = torch.tensor([.0448, .2856, .3001, .2363, .1333],
                           device=a.device, dtype=a.dtype)
    x, y, values, contrasts = a, b, [], []
    for level in range(5):
        ux, uy = F.conv2d(x, kernel, groups=channels), F.conv2d(y, kernel, groups=channels)
        vx = F.conv2d(x * x, kernel, groups=channels) - ux * ux
        vy = F.conv2d(y * y, kernel, groups=channels) - uy * uy
        covariance = F.conv2d(x * y, kernel, groups=channels) - ux * uy
        contrast = (2 * covariance + .03 ** 2) / (vx + vy + .03 ** 2)
        ssim = ((2 * ux * uy + .01 ** 2) / (ux * ux + uy * uy + .01 ** 2)) * contrast
        values.append(ssim.mean((0, 2, 3)))
        contrasts.append(contrast.mean((0, 2, 3)))
        if level < 4:
            x = F.avg_pool2d(F.pad(x, (0, 1, 0, 1), mode="reflect"), 2, 2)
            y = F.avg_pool2d(F.pad(y, (0, 1, 0, 1), mode="reflect"), 2, 2)
    result = (torch.stack(contrasts[:-1]).clamp_min(1e-12).pow(weights[:-1, None]).prod(0) *
              values[-1].clamp_min(1e-12).pow(weights[-1])).mean()
    return float(result)


def grad_norm(parameters):
    total = sum(float(p.grad.detach().double().square().sum())
                for p in parameters if p.grad is not None)
    return math.sqrt(total)


def param_norm(parameters):
    return math.sqrt(sum(float(p.detach().double().square().sum()) for p in parameters))


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_config_betas(betas):
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text())
    config["beta_weights"] = betas
    config_path.write_text(json.dumps(config, indent=2) + "\n")
