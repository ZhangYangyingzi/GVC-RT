import csv, hashlib, io, json, math, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V9 = REPO / "expericent_generation_input/expericent_interface_causal_controls_v9/src"
for p in (REPO, V9):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from gvc_hooks import load_models, INDEX_MAP
from src.layers.cuda_inference import round_and_to_int8


def unit(x, h=None, w=None):
    y = (x.float().clamp(-1, 1) + 1) / 2
    return y if h is None else y[:, :, :h, :w]


def codec_input(x):
    h, w = x.shape[-2:]
    ph, pw = (64 - h % 64) % 64, (64 - w % 64) % 64
    return F.pad(x, (0, pw, 0, ph), mode="replicate").half() * 2 - 1


def model_hash(*models):
    h = hashlib.sha256()
    for model in models:
        for name, p in model.named_parameters():
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def quality_models(device):
    import lpips
    from DISTS_pytorch import DISTS
    return (lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False),
            DISTS().to(device).eval().requires_grad_(False))


def perceptual(output, target, quality):
    lp = quality[0](output, target, normalize=True).mean()
    di = quality[1](output, target).mean()
    return lp + di, lp, di


def prepare_p(model, qp):
    qf = model.q_scale_feature[qp:qp+1]; qe = model.q_scale_enc[qp:qp+1]
    qd = model.q_scale_dec[qp:qp+1]; qr = model.q_scale_recon[qp:qp+1]
    feature = model.apply_feature_adaptor()
    context, temporal = model.feature_extractor(feature, qf)
    return context, temporal, qe, qd, qr


def ste_forward_p(model, proxy, qp):
    context, temporal, qe, qd, qr = prepare_p(model, qp)
    y = model.enc(codec_input(proxy), context, qe)
    z = model.hyper_enc(model.pad_for_y(y))
    zhard = torch.clamp(torch.round(z), -128, 127)
    zste = z + (zhard - z).detach()
    params = model.res_prior_param_decoder(zste, temporal)
    qdec, scale, mean = params.chunk(3, 1); qdec = torch.clamp_min(qdec, .5)
    yscaled = y * torch.reciprocal(qdec)
    B, C, H, W = yscaled.shape
    mask0, mask1 = model.get_mask_2x(B, C, H, W, yscaled.dtype, yscaled.device)

    def proc(value, sc, mu, mask):
        sc = sc * mask; mu = mu * mask; residual = (value - mu) * mask
        hard = torch.clamp(torch.round(residual), -128, 127)
        active = mask.bool() & (sc > .12); hard = hard * active
        ste = residual + (hard - residual).detach()
        return hard, ste, sc, mu, active

    hard0, q0, sc0, mu0, active0 = proc(yscaled, scale, mean, mask0)
    yhat0 = q0 + mu0
    sc1, mu1 = model.y_spatial_prior(torch.cat((yhat0, params), 1)).chunk(2, 1)
    hard1, q1, sc1, mu1, active1 = proc(yscaled, sc1, mu1, mask1)
    yhat = (yhat0 + q1 + mu1) * qdec
    output = unit(model.recon_generation_net(model.dec(yhat, context, qd), qr), proxy.shape[-2], proxy.shape[-1])
    upper = model.bit_estimator_z.get_cdf(zste.float() + .5, qp)
    lower = model.bit_estimator_z.get_cdf(zste.float() - .5, qp)
    rate = (-torch.log2((upper - lower).clamp_min(1e-9))).sum()

    def gaussian_bits(q, sc, active):
        q = q.float()[active]; sc = sc.float()[active].clamp(.11, 16)
        idx = ((torch.log(sc) - model.gaussian_encoder.log_scale_min) * model.gaussian_encoder.log_step_recip).long().clamp(0, 127)
        stable_scale = model.gaussian_encoder.scale_table.to(sc.device)[idx]
        normal = torch.distributions.Normal(torch.zeros_like(stable_scale), stable_scale)
        return (-torch.log2((normal.cdf(q + .5) - normal.cdf(q - .5)).clamp_min(1e-9))).sum()
    rate = rate + gaussian_bits(q0, sc0, active0) + gaussian_bits(q1, sc1, active1)
    with torch.no_grad():
        ztrue, _ = round_and_to_int8(z)
        ptrue = model.res_prior_param_decoder(ztrue, temporal)
        _, _, _, _, true_yhat = model.compress_prior_2x(y, ptrue, model.y_spatial_prior)
        ste_diff = float((true_yhat - yhat.detach()).abs().max())
    return output, rate, ste_diff


def load_train_pair(index, device):
    cache = REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11/cache/train_steps"
    payload = torch.load(cache / f"step_{index:04d}.pt", map_location="cpu", weights_only=True)
    pair = [unit(x["gt_rgb"].to(device).float()) for x in payload["pair"]]
    return pair, payload["plan"]


def establish_dpb(i_model, p_model, previous_proxy, requested_qp):
    p_model.clear_dpb(); p_model.set_curr_poc(0)
    with torch.no_grad():
        encoded = i_model.compress(codec_input(previous_proxy), requested_qp)
        p_model.add_ref_frame(None, encoded["x_hat"])


def csv_write(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def basic_metrics(output, target):
    diff = output - target; sse = float(diff.square().sum()); mse = float(diff.square().mean())
    psnr = -10 * math.log10(max(mse, 1e-15))
    ux, uy = output.mean(), target.mean(); vx = ((output-ux)**2).mean(); vy = ((target-uy)**2).mean(); cov = ((output-ux)*(target-uy)).mean()
    ssim = float(((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2)))
    return sse, mse, psnr, ssim

