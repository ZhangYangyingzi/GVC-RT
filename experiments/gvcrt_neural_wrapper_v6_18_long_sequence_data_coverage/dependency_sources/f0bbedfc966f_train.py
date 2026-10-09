#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from bridge_adapter import FeatureAdapter
from common import ROOT, config, torch_load, write_json
from copied_bridge import CopiedBridge
from current_only_refiner import CurrentOnlyRefiner
from gvc_hooks import decoder_for, load_models
from losses import WEIGHTS, pair_loss
from rgb_postprocessor import RGBPostprocessor
from temporal_refiner import TemporalRefiner


METHOD_DIRS = {"m1": "m1_bridge", "m2": "m2_bridge_adapter", "m3": "m3_current_only",
               "m4": "m4_temporal", "m5": "m5_rgb_post"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_lr(value):
    return f"lr_{value:.0e}"


def build_method(name, codec_models, device):
    if name == "m1":
        model = CopiedBridge(codec_models[1].recon_generation_net.mlp, train_bridge=True)
    elif name == "m2":
        model = CopiedBridge(codec_models[1].recon_generation_net.mlp, train_bridge=False,
                             adapter=FeatureAdapter())
    elif name == "m3":
        model = CurrentOnlyRefiner()
    elif name == "m4":
        model = TemporalRefiner()
    elif name == "m5":
        model = RGBPostprocessor()
    else:
        raise ValueError(name)
    return model.to(device).train()


def render_pair(method, model, payload, codec_models, device):
    records = payload["records"]
    targets = [payload["targets"][records[index]["frame"]].to(device).float()
               for index in (1, 2)]
    outputs, residuals = [], []
    if method in ("m1", "m2"):
        for index in (1, 2):
            record = records[index]
            base = record["codeword"].to(device)
            if record["frame_type"] == "I":
                candidate, residual = base, torch.zeros_like(base)
            else:
                candidate = model(record["bridge_feature"].to(device).float())
                residual = candidate.float() - base.float()
            output = decoder_for(record["frame_type"], codec_models)(
                candidate.half(), record["quant"].to(device))
            outputs.append(output)
            residuals.append(residual)
    elif method in ("m3", "m4"):
        for index in (1, 2):
            current = records[index]["codeword"].to(device).float()
            previous = records[index - 1]["codeword"].to(device).float()
            result = model(current, previous)
            output = decoder_for(records[index]["frame_type"], codec_models)(
                result["codeword"].half(), records[index]["quant"].to(device))
            outputs.append(output)
            residuals.append(result["delta"])
    else:
        baselines = []
        with torch.no_grad():
            for record in records:
                baselines.append(decoder_for(record["frame_type"], codec_models)(
                    record["codeword"].to(device), record["quant"].to(device)))
        for index in (1, 2):
            result = model(baselines[index].float(), baselines[index - 1].float())
            outputs.append(result["image"])
            residuals.append(result["delta"])
    return outputs, targets, residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHOD_DIRS), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config()
    if args.lr not in cfg["learning_rates"]:
        raise ValueError("LR is outside the equal preregistered study")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed_all(cfg["seed"])
    device = torch.device("cuda:0")
    codec_models = load_models(device)
    model = build_method(args.method, codec_models, device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    stored_parameters = sum(parameter.numel() for parameter in model.parameters())
    import lpips
    from DISTS_pytorch import DISTS
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    dists_model = DISTS().to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.lr, weight_decay=cfg["weight_decay"])
    checkpoint_dir = ROOT / "checkpoints" / METHOD_DIRS[args.method] / label_lr(args.lr)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history, start_step = [], 0
    candidates = sorted(checkpoint_dir.glob("update_*.pt"))
    if candidates and not args.smoke:
        resume = torch_load(candidates[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        history, start_step = resume["history"], int(resume["update"])
    started = time.perf_counter()
    final_step = 1 if args.smoke else cfg["training_updates"]
    for step in range(start_step + 1, final_step + 1):
        payload = torch_load(ROOT / f"cache/receiver/train_steps/step_{step:03d}.pt",
                             map_location="cpu", weights_only=False)
        outputs, targets, residuals = render_pair(args.method, model, payload, codec_models, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = pair_loss(outputs, targets, residuals, lpips_model, dists_model)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            cfg["gradient_clip_norm"]))
        if not np.isfinite(gradient_norm) or gradient_norm == 0:
            raise RuntimeError(f"Invalid gradient norm at update {step}: {gradient_norm}")
        if not args.smoke:
            optimizer.step()
        row = {"update": step, "video_index": payload["plan"]["video_index"],
               "requested_qp": payload["plan"]["requested_qp"],
               "gradient_norm": gradient_norm, **components}
        history.append(row)
        print(f"{args.method} lr={args.lr:g} update {step:03d}/{final_step}: "
              f"loss={components['total']:.6f} grad={gradient_norm:.6f}", flush=True)
        if not args.smoke and step in cfg["checkpoint_updates"]:
            torch.save({"method": args.method, "lr": args.lr, "update": step,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "loss_weights": WEIGHTS,
                        "trainable_parameters": parameters, "stored_parameters": stored_parameters,
                        "training_plan_sha256": __import__("common").sha256(ROOT / "analysis/training_plan.json")},
                       checkpoint_dir / f"update_{step:03d}.pt")
        del payload, outputs, targets, residuals, loss
    result = {"method": args.method, "lr": args.lr, "gpu": args.gpu,
              "smoke": args.smoke, "updates": final_step,
              "trainable_parameters": parameters, "stored_parameters": stored_parameters,
              "seconds_this_run": time.perf_counter() - started,
              "loss_weights": WEIGHTS, "history": history if not args.smoke else history[-1:]}
    destination = ROOT / "logs" / (("smoke_" if args.smoke else "train_") +
                                    f"{args.method}_{label_lr(args.lr)}.json")
    write_json(destination, result)


if __name__ == "__main__":
    main()
