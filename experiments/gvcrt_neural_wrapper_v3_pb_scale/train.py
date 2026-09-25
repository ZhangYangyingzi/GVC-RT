#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "gvcrt_neural_wrapper_v2_joint"
sys.path.insert(0, str(V2))
from core import (codec_input, compression_hash, grad_norm, joint_ste_forward,
                  module_hash, ms_ssim_rgb, perceptual, quality_models)
from wrapper_model import NeuralWrapper
from gvc_hooks import load_models


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sample_clip(records, rng, device, crop=256):
    for _ in range(20):
        record = rng.choice(records)
        capture = cv2.VideoCapture(record["path"])
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if count < 4 or width < crop or height < crop:
            capture.release(); continue
        start = rng.randrange(count - 3)
        x = rng.randrange(width - crop + 1)
        y = rng.randrange(height - crop + 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(4):
            ok, bgr = capture.read()
            if not ok: break
            rgb = cv2.cvtColor(bgr[y:y+crop, x:x+crop], cv2.COLOR_BGR2RGB).copy()
            frames.append(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device).float() / 255)
        capture.release()
        if len(frames) == 4:
            return frames, {"video": record["filename"], "start": start, "crop_x": x, "crop_y": y}
    raise RuntimeError("cannot sample four consecutive frames")


def save(path, stage, step, beta, wrapper, bridge, generator, optimizer, hashes, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "gvcrt_v3_pb", "stage": stage, "step": step, "beta": beta,
                "wrapper": wrapper.state_dict(), "bridge": bridge.state_dict(),
                "generator": generator.state_dict(), "optimizer": optimizer.state_dict(),
                "initial_hashes": hashes, "sampling_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state()}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(4,5,6,7), required=True)
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--target-steps", type=int,
                        help="Optional extended total for a resumed trajectory.")
    parser.add_argument("--checkpoint-steps", default=None,
                        help="Comma-separated additional checkpoint steps.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = json.loads((ROOT / "config.json").read_text())
    tc = cfg[args.stage]; total = args.target_steps or tc["updates"]
    checkpoint_steps = set(tc["checkpoint_steps"])
    if args.checkpoint_steps:
        checkpoint_steps.update(int(value) for value in args.checkpoint_steps.split(","))
    limit = min(total, args.max_steps) if args.max_steps else total
    beta = tc["beta"]
    rng = random.Random(cfg["seed"] + (0 if args.stage == "stage_a" else 5000))
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda:0")
    wrapper = NeuralWrapper().to(device).float().train()
    i_model, p_model = load_models(device)
    i_model.requires_grad_(False); p_model.requires_grad_(False)
    bridge = p_model.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator = p_model.recon_generation_net.decoder.float().eval().requires_grad_(False)
    wp, bp = list(wrapper.parameters()), list(bridge.parameters())
    optcfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": wp, "lr": optcfg["wrapper_lr"], "name": "wrapper"},
        {"params": bp, "lr": optcfg["bridge_lr"], "name": "bridge"}],
        weight_decay=optcfg["weight_decay"])
    hashes = {"compression": compression_hash(i_model,p_model), "generator": module_hash(generator),
              "wrapper": module_hash(wrapper), "bridge": module_hash(bridge)}
    sample = torch.rand(1,3,64,64,device=device)
    identity_diff = float((wrapper(sample)-sample).abs().max())
    outdir = ROOT / "checkpoints" / args.stage
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / "parts" / f"training_{args.stage}.csv"
    history = []; start_step = 0
    checkpoints = sorted(outdir.glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if checkpoints:
        cp = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        wrapper.load_state_dict(cp["wrapper"]); bridge.load_state_dict(cp["bridge"])
        optimizer.load_state_dict(cp["optimizer"])
        hashes = cp["initial_hashes"]; start_step = int(cp["step"])
        rng.setstate(cp["sampling_rng_state"])
        torch.set_rng_state(cp["torch_rng_state"])
        torch.cuda.set_rng_state(cp["cuda_rng_state"])
        if history_path.exists():
            with open(history_path, newline="") as f:
                history = [r for r in csv.DictReader(f) if int(r["step"]) <= start_step]
    elif args.stage == "stage_b":
        cp = torch.load(ROOT / "checkpoints/stage_a/step_5000.pt",map_location="cpu",weights_only=True)
        wrapper.load_state_dict(cp["wrapper"]); bridge.load_state_dict(cp["bridge"])
        hashes["wrapper"] = module_hash(wrapper); hashes["bridge"] = module_hash(bridge)
    if not checkpoints:
        save(outdir / "step_0000.pt", args.stage, 0, beta, wrapper, bridge, generator, optimizer, hashes, rng)
    quality = quality_models(device)
    records = json.loads((ROOT / "train_manifest.json").read_text())["videos"]
    started = time.time()
    for step in range(start_step+1, limit+1):
        frames, plan = sample_clip(records, rng, device)
        qp = rng.randrange(4)
        p_model.clear_dpb(); p_model.set_curr_poc(0)
        with torch.no_grad():
            first_proxy = wrapper(frames[0]); encoded = i_model.compress(codec_input(first_proxy),qp)
            p_model.add_ref_frame(None, encoded["x_hat"])
        optimizer.zero_grad(set_to_none=True)
        captured = []
        hook = p_model.dec.register_forward_hook(lambda _m,_a,out: captured.append(out))
        sums = {k: 0.0 for k in ("loss","LPIPS","DISTS","R_est_bpp","proxy_L1","PSNR","MS_SSIM")}
        try:
            for frame in frames[1:]:
                proxy = wrapper(frame)
                reconstruction, rate, _ = joint_ste_forward(p_model, proxy, qp)
                distortion, lpips, dists = perceptual(reconstruction,frame,quality)
                rate_bpp = rate / (frame.shape[-2]*frame.shape[-1])
                l1 = F.l1_loss(proxy,frame)
                loss = (distortion + beta*rate_bpp + cfg["lambda_proxy"]*l1)/3
                if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at {step}")
                loss.backward()
                with torch.no_grad():
                    p_model.add_ref_frame(captured[-1].detach(), (reconstruction.detach()*2-1).half())
                sums["loss"] += float(loss); sums["LPIPS"] += float(lpips)/3
                sums["DISTS"] += float(dists)/3; sums["R_est_bpp"] += float(rate_bpp)/3
                sums["proxy_L1"] += float(l1)/3
                sums["PSNR"] += -10*math.log10(max(float((reconstruction.detach()-frame).square().mean()),1e-15))/3
                sums["MS_SSIM"] += ms_ssim_rgb(reconstruction.detach(),frame)/3
        finally:
            hook.remove()
        wg, bg = grad_norm(wp), grad_norm(bp)
        all_grad = float(torch.nn.utils.clip_grad_norm_(wp+bp,optcfg["gradient_clip_norm"]))
        if not all(math.isfinite(x) for x in (wg,bg,all_grad)) or all_grad <= 0:
            raise RuntimeError(f"nonfinite/zero gradient at {step}: {wg},{bg},{all_grad}")
        optimizer.step()
        history.append({"stage":args.stage,"step":step,"beta":beta,"qp":qp,**plan,**sums,
                        "wrapper_gradient_norm":wg,"bridge_gradient_norm":bg,"all_finite":True})
        if step%100==0:
            write_csv(history_path, history)
            print(f"{args.stage} {step}/{total} loss={sums['loss']:.6f} LPIPS={sums['LPIPS']:.6f}",flush=True)
        if step in checkpoint_steps:
            save(outdir / f"step_{step:04d}.pt",args.stage,step,beta,wrapper,bridge,generator,optimizer,hashes,rng)
    write_csv(history_path,history)
    result={"status":"PASS" if limit==total else "SMOKE_PASS","stage":args.stage,"steps":limit,
            "seconds":time.time()-started,"identity_max_diff":identity_diff,
            "initial_hashes":hashes,"final_hashes":{"compression":compression_hash(i_model,p_model),
            "generator":module_hash(generator),"wrapper":module_hash(wrapper),"bridge":module_hash(bridge)},
            "wrapper_trainable":sum(p.numel() for p in wp),"bridge_trainable":sum(p.numel() for p in bp),
            "generator_trainable":sum(p.numel() for p in generator.parameters() if p.requires_grad)}
    (ROOT / "parts").mkdir(exist_ok=True)
    (ROOT / "parts" / f"train_{args.stage}_done.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
