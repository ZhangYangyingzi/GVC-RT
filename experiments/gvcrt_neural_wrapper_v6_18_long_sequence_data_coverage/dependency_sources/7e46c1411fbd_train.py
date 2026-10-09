#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from core import (ROOT, basic_metrics, compression_hash, csv_write, establish_dpb,
                  grad_norm, joint_ste_forward, load_train_pair, module_hash,
                  ms_ssim_rgb, parameter_count, param_norm, perceptual,
                  prepare_joint_models, quality_models)
from wrapper_model import NeuralWrapper


BRANCHES = ("beta_low", "beta_mid", "beta_high")


def checkpoint_steps(stage):
    return {0, 500, 1000, 2000} if stage == "joint_warmup" else {0, 500, 1000, 2000, 5000}


def checkpoint_payload(stage, beta_label, beta, step, wrapper, bridge, generator,
                       optimizer, initial, compression_before):
    return {
        "schema": "gvcrt_neural_wrapper_v2_joint_checkpoint",
        "stage": stage,
        "beta_label": beta_label,
        "beta": beta,
        "step": step,
        "wrapper": wrapper.state_dict(),
        "bridge": bridge.state_dict(),
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "initial_hashes": initial,
        "compression_hash": compression_before,
    }


def save_checkpoint(path, *args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(*args), path)


def load_states(path, wrapper, bridge, generator, optimizer=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    wrapper.load_state_dict(payload["wrapper"], strict=True)
    bridge.load_state_dict(payload["bridge"], strict=True)
    generator.load_state_dict(payload["generator"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def read_history(path, step):
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return [row for row in csv.DictReader(handle) if int(row["step"]) <= step]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--stage", choices=("joint_warmup", "rate"), required=True)
    parser.add_argument("--beta-label", choices=BRANCHES)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Test-only cap; production runs omit this option.")
    args = parser.parse_args()
    if (args.stage == "rate") != (args.beta_label is not None):
        parser.error("--beta-label is required exactly for --stage rate")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed = 20260924
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    config = json.loads((ROOT / "config.json").read_text())
    training = config["training"]
    total_steps = training["stage_a_updates"] if args.stage == "joint_warmup" else training["stage_b_updates"]
    run_steps = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    beta = 0.0 if args.stage == "joint_warmup" else float(config["beta_weights"][args.beta_label])
    label = "joint_warmup" if args.stage == "joint_warmup" else args.beta_label
    output_dir = ROOT / "checkpoints" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    part_path = ROOT / "parts" / f"training_{label}.csv"

    wrapper = NeuralWrapper().to(device).float().train()
    i_model, p_model, bridge, generator = prepare_joint_models(device)
    wrapper_params = list(wrapper.parameters())
    bridge_params = list(bridge.parameters())
    generator_params = list(generator.parameters())
    optimizer = torch.optim.AdamW([
        {"name": "wrapper", "params": wrapper_params, "lr": training["wrapper_lr"]},
        {"name": "bridge", "params": bridge_params, "lr": training["bridge_lr"]},
        {"name": "generator", "params": generator_params, "lr": training["generator_lr"]},
    ], weight_decay=training["weight_decay"])
    compression_before = compression_hash(i_model, p_model)
    initial = {
        "compression": compression_before,
        "wrapper": module_hash(wrapper),
        "bridge": module_hash(bridge),
        "generator": module_hash(generator),
    }
    identity_sample = torch.rand(1, 3, 64, 64, device=device)
    identity_difference = float((wrapper(identity_sample) - identity_sample).abs().max())

    history, start_step = [], 0
    existing = sorted(output_dir.glob("step_*.pt"))
    if existing:
        latest = existing[-1]
        payload = load_states(latest, wrapper, bridge, generator, optimizer)
        start_step = int(payload["step"])
        initial = payload["initial_hashes"]
        if payload["compression_hash"] != compression_before:
            raise RuntimeError("pretrained compression core hash differs from checkpoint")
        history = read_history(part_path, start_step)
    elif args.stage == "rate":
        warmup = ROOT / "checkpoints/joint_warmup/step_2000.pt"
        if not warmup.exists():
            raise RuntimeError("Stage A step_2000.pt is required before Stage B")
        load_states(warmup, wrapper, bridge, generator)
        initial = {
            "compression": compression_before,
            "wrapper": module_hash(wrapper),
            "bridge": module_hash(bridge),
            "generator": module_hash(generator),
        }
    if not existing:
        save_checkpoint(output_dir / "step_0000.pt", args.stage, args.beta_label,
                        beta, 0, wrapper, bridge, generator, optimizer, initial,
                        compression_before)

    quality = quality_models(device)
    started = time.time()
    for step in range(start_step + 1, run_steps + 1):
        cache_index = ((step - 1) * 977 + 37) % 1500 + 1
        (previous, current), plan = load_train_pair(cache_index, device)
        qp = (step - 1) % 4
        with torch.no_grad():
            previous_proxy = wrapper(previous)
        establish_dpb(i_model, p_model, previous_proxy, qp)
        optimizer.zero_grad(set_to_none=True)
        proxy = wrapper(current)
        output, rate, ste_difference = joint_ste_forward(p_model, proxy, qp)
        distortion, lpips_value, dists_value = perceptual(output, current, quality)
        rate_bpp = rate / (current.shape[-2] * current.shape[-1])
        proxy_l1 = F.l1_loss(proxy, current)
        loss = distortion + beta * rate_bpp + training["lambda_proxy"] * proxy_l1
        values = (loss, distortion, rate_bpp, lpips_value, dists_value, proxy_l1)
        if not all(torch.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite value at step {step}")
        loss.backward()
        wrapper_gradient = grad_norm(wrapper_params)
        bridge_gradient = grad_norm(bridge_params)
        generator_gradient = grad_norm(generator_params)
        all_gradient = float(torch.nn.utils.clip_grad_norm_(
            wrapper_params + bridge_params + generator_params,
            training["gradient_clip_norm"]))
        group_gradients = (wrapper_gradient, bridge_gradient, generator_gradient)
        if (not all(math.isfinite(value) and value >= 0 for value in group_gradients)
                or not math.isfinite(all_gradient) or all_gradient <= 0):
            raise RuntimeError(
                f"invalid gradient at step {step}: wrapper={wrapper_gradient}, "
                f"bridge={bridge_gradient}, generator={generator_gradient}, all={all_gradient}")
        optimizer.step()
        row = {
            "stage": "A" if args.stage == "joint_warmup" else "B",
            "beta_label": args.beta_label or "none",
            "beta": beta,
            "step": step,
            "cache_index": cache_index,
            "source_video_index": plan["video_index"],
            "requested_qp": qp,
            "loss": float(loss),
            "LPIPS": float(lpips_value),
            "DISTS": float(dists_value),
            "PSNR": basic_metrics(output.detach(), current),
            "MS_SSIM": ms_ssim_rgb(output.detach(), current),
            "R_est_bpp": float(rate_bpp),
            "proxy_source_L1": float(proxy_l1),
            "wrapper_gradient_norm": wrapper_gradient,
            "bridge_gradient_norm": bridge_gradient,
            "generator_gradient_norm": generator_gradient,
            "all_gradient_norm_before_clip": all_gradient,
            "wrapper_parameter_norm": param_norm(wrapper_params),
            "bridge_parameter_norm": param_norm(bridge_params),
            "generator_parameter_norm": param_norm(generator_params),
            "STE_forward_max_abs_diff": ste_difference,
            "all_finite": True,
        }
        history.append(row)
        if step % 25 == 0 or step in checkpoint_steps(args.stage):
            print(f"{label} {step}/{total_steps} loss={float(loss):.6f} "
                  f"R={float(rate_bpp):.6f} LP={float(lpips_value):.6f} "
                  f"DI={float(dists_value):.6f}", flush=True)
        if step in checkpoint_steps(args.stage):
            save_checkpoint(output_dir / f"step_{step:04d}.pt", args.stage,
                            args.beta_label, beta, step, wrapper, bridge, generator,
                            optimizer, initial, compression_before)
        if step % 100 == 0 or step == run_steps:
            csv_write(part_path, history)

    compression_after = compression_hash(i_model, p_model)
    result = {
        "status": "PASS" if run_steps == total_steps else "SMOKE_PASS",
        "stage": args.stage,
        "beta_label": args.beta_label,
        "beta": beta,
        "steps": run_steps,
        "production_steps": total_steps,
        "seconds_this_run": time.time() - started,
        "identity_initialization_max_abs_diff": identity_difference,
        "wrapper_identity_init_pass": identity_difference == 0 if args.stage == "joint_warmup" and start_step == 0 else True,
        "trainable_counts": {
            "wrapper": parameter_count(wrapper),
            "bridge": parameter_count(bridge),
            "generator": parameter_count(generator),
        },
        "compression_hash_before": compression_before,
        "compression_hash_after": compression_after,
        "compression_hash_unchanged": compression_before == compression_after,
        "initial_hashes": initial,
        "final_hashes": {
            "wrapper": module_hash(wrapper),
            "bridge": module_hash(bridge),
            "generator": module_hash(generator),
        },
    }
    (ROOT / "parts").mkdir(exist_ok=True)
    (ROOT / "parts" / f"train_{label}_done.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
