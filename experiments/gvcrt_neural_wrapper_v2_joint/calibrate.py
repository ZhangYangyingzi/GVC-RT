#!/usr/bin/env python3
import argparse
import json
import math
import os

import torch

from core import (ROOT, establish_dpb, joint_ste_forward, load_train_pair,
                  perceptual, prepare_joint_models, quality_models,
                  update_config_betas)
from wrapper_model import NeuralWrapper


def norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in gradients if g is not None))


def load_states(path, wrapper, bridge, generator):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    wrapper.load_state_dict(payload["wrapper"], strict=True)
    bridge.load_state_dict(payload["bridge"], strict=True)
    generator.load_state_dict(payload["generator"], strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    wrapper = NeuralWrapper().to(device).float().train()
    i_model, p_model, bridge, generator = prepare_joint_models(device)
    checkpoint = ROOT / "checkpoints/joint_warmup/step_2000.pt"
    load_states(checkpoint, wrapper, bridge, generator)
    quality = quality_models(device)
    (previous, current), plan = load_train_pair(1499, device)
    qp = int(plan.get("requested_qp", 2)) % 4
    with torch.no_grad():
        previous_proxy = wrapper(previous)
    establish_dpb(i_model, p_model, previous_proxy, qp)
    proxy = wrapper(current)
    output, rate, _ = joint_ste_forward(p_model, proxy, qp)
    distortion, lpips_value, dists_value = perceptual(output, current, quality)
    rate_bpp = rate / (current.shape[-2] * current.shape[-1])
    parameters = list(wrapper.parameters()) + list(bridge.parameters()) + list(generator.parameters())
    perceptual_norm = norm(distortion, parameters)
    rate_norm = norm(rate_bpp, parameters)
    if not all(math.isfinite(value) and value > 0 for value in (perceptual_norm, rate_norm)):
        raise RuntimeError("calibration gradient norms must be finite and positive")
    balance = perceptual_norm / rate_norm
    multipliers = json.loads((ROOT / "config.json").read_text())["training"]["beta_multipliers"]
    betas = {label: balance * float(multiplier) for label, multiplier in multipliers.items()}
    update_config_betas(betas)
    result = {
        "status": "PASS",
        "source_checkpoint": str(checkpoint),
        "qp": qp,
        "LPIPS": float(lpips_value),
        "DISTS": float(dists_value),
        "R_est_bpp": float(rate_bpp),
        "perceptual_gradient_norm": perceptual_norm,
        "rate_gradient_norm": rate_norm,
        "balance_beta": balance,
        "betas": betas,
    }
    (ROOT / "sanity").mkdir(exist_ok=True)
    (ROOT / "sanity/beta_calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
