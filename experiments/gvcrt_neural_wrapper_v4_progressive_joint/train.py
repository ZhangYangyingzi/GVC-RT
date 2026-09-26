#!/usr/bin/env python3
import argparse
import csv
import fcntl
import hashlib
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


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


def save(path, branch, step, beta, wrapper, bridge, generator, optimizer, hashes, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"schema": "gvcrt_v4_progressive_joint", "branch": branch,
                "stage": branch, "step": step, "beta": beta,
                "wrapper": wrapper.state_dict(), "bridge": bridge.state_dict(),
                "generator": generator.state_dict(), "optimizer": optimizer.state_dict(),
                "initial_hashes": hashes, "sampling_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state()}, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(4,5,6,7), required=True)
    parser.add_argument("--branch", choices=("beta_low", "beta_mid", "beta_high"), required=True)
    args = parser.parse_args()
    (ROOT / "parts").mkdir(exist_ok=True)
    lock = open(ROOT / "parts" / f"train_{args.branch}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = json.loads((ROOT / "config.json").read_text())
    branch = cfg["branches"][args.branch]; total = cfg["updates"]; limit = total
    checkpoint_steps = set(cfg["checkpoint_steps"]); beta = branch["beta"]
    rng = random.Random(cfg["seed"] + branch["seed_offset"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda:0")
    wrapper = NeuralWrapper().to(device).float().train()
    i_model, p_model = load_models(device)
    i_model.requires_grad_(False); p_model.requires_grad_(False)
    init_path = Path(cfg["initial_checkpoint"])
    cp = torch.load(init_path, map_location="cpu", weights_only=True)
    if int(cp["step"]) != 20000:
        raise RuntimeError("expected V3 step 20000 initialization")
    wrapper.load_state_dict(cp["wrapper"])
    bridge = p_model.recon_generation_net.mlp.float().train().requires_grad_(True)
    bridge.load_state_dict(cp["bridge"])
    generator = p_model.recon_generation_net.decoder.float().train().requires_grad_(True)
    generator.load_state_dict(cp["generator"])
    equal = {name: all(torch.equal(t.cpu(), cp[name][k])
                      for k, t in module.state_dict().items())
             for name, module in (("wrapper", wrapper), ("bridge", bridge), ("generator", generator))}
    if not all(equal.values()):
        raise RuntimeError(f"V3 initialization mismatch: {equal}")
    source_digest = hashlib.sha256(init_path.read_bytes()).hexdigest()
    generator_initial = {n: cp["generator"][n].clone() for n, _ in generator.named_parameters()}
    generator_norm_before = math.sqrt(sum(float(t.double().square().sum()) for t in generator_initial.values()))
    wp, bp, gp = list(wrapper.parameters()), list(bridge.parameters()), list(generator.parameters())
    optcfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": wp, "lr": optcfg["wrapper_lr"], "name": "wrapper"},
        {"params": bp, "lr": optcfg["bridge_lr"], "name": "bridge"},
        {"params": gp, "lr": optcfg["generator_lr"], "name": "generator"}],
        weight_decay=optcfg["weight_decay"])
    hashes = {"compression": compression_hash(i_model,p_model), "generator": module_hash(generator),
              "wrapper": module_hash(wrapper), "bridge": module_hash(bridge)}
    bridge_ids = {id(p) for p in bp}
    generator_ids = {id(p) for p in gp}
    compression_trainable = (
        sum(p.numel() for p in i_model.parameters() if p.requires_grad) +
        sum(p.numel() for p in p_model.parameters()
            if p.requires_grad and id(p) not in bridge_ids and id(p) not in generator_ids)
    )
    audit = {
        "branch": args.branch,
        "initial_checkpoint": str(init_path),
        "initial_checkpoint_sha256": source_digest,
        "state_equal_to_v3": equal,
        "optimizer_initialized_independently": True,
        "gpu": args.gpu,
        "compression_hash": hashes["compression"],
        "wrapper_hash": hashes["wrapper"],
        "bridge_hash": hashes["bridge"],
        "generator_hash": hashes["generator"],
        "wrapper_trainable_parameters": sum(p.numel() for p in wp if p.requires_grad),
        "bridge_trainable_parameters": sum(p.numel() for p in bp if p.requires_grad),
        "generator_trainable_parameters": sum(p.numel() for p in gp if p.requires_grad),
        "compression_trainable_parameters": compression_trainable,
    }
    if audit["compression_trainable_parameters"] != 0:
        raise RuntimeError("compression core has trainable parameters")
    if audit["generator_trainable_parameters"] <= 0:
        raise RuntimeError("generator has no trainable parameters")
    (ROOT / "parts").mkdir(exist_ok=True)
    (ROOT / "parts" / f"parameter_audit_initial_{args.branch}.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    if args.branch == "beta_low":
        (ROOT / "parameter_audit_initial.json").write_text(json.dumps(audit, indent=2) + "\n")
    sample = torch.rand(1,3,64,64,device=device)
    identity_diff = float((wrapper(sample)-sample).abs().max())
    outdir = ROOT / "checkpoints" / args.branch
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / "training_logs" / f"training_{args.branch}.csv"
    history = []; start_step = 0
    checkpoints = sorted(outdir.glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if checkpoints:
        cp = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if cp["branch"] != args.branch or cp["beta"] != beta or cp["initial_hashes"] != hashes:
            raise RuntimeError("resume checkpoint identity mismatch")
        wrapper.load_state_dict(cp["wrapper"]); bridge.load_state_dict(cp["bridge"]); generator.load_state_dict(cp["generator"])
        optimizer.load_state_dict(cp["optimizer"]); start_step = int(cp["step"])
        rng.setstate(cp["sampling_rng_state"]); torch.set_rng_state(cp["torch_rng_state"]); torch.cuda.set_rng_state(cp["cuda_rng_state"])
        if history_path.exists():
            with open(history_path, newline="") as f: history = [r for r in csv.DictReader(f) if int(r["step"]) <= start_step]
    else:
        save(outdir / "step_0000.pt", args.branch, 0, beta, wrapper, bridge, generator, optimizer, hashes, rng)
    del cp
    print(f"INITIALIZED {args.branch} GPU={args.gpu} start={start_step} compression_trainable={compression_trainable} generator_trainable={audit['generator_trainable_parameters']}", flush=True)
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
        wg, bg, gg = grad_norm(wp), grad_norm(bp), grad_norm(gp)
        all_grad = float(torch.nn.utils.clip_grad_norm_(wp+bp+gp,optcfg["gradient_clip_norm"]))
        if not all(math.isfinite(x) for x in (wg,bg,gg,all_grad)) or all_grad <= 0:
            raise RuntimeError(f"nonfinite/zero gradient at {step}: {wg},{bg},{gg},{all_grad}")
        optimizer.step()
        history.append({"branch":args.branch,"step":step,"beta":beta,"qp":qp,**plan,**sums,
                        "wrapper_gradient_norm":wg,"bridge_gradient_norm":bg,"generator_gradient_norm":gg,"all_finite":True})
        if step == start_step+1 or step%100==0:
            write_csv(history_path, history)
            print(f"{args.branch} {step}/{total} loss={sums['loss']:.6f} LPIPS={sums['LPIPS']:.6f}",flush=True)
        if step in checkpoint_steps:
            save(outdir / f"step_{step:04d}.pt",args.branch,step,beta,wrapper,bridge,generator,optimizer,hashes,rng)
    write_csv(history_path,history)
    result={"status":"PASS","branch":args.branch,"steps":total,
            "seconds":time.time()-started,"identity_max_diff":identity_diff,
            "initial_hashes":hashes,"final_hashes":{"compression":compression_hash(i_model,p_model),
            "generator":module_hash(generator),"wrapper":module_hash(wrapper),"bridge":module_hash(bridge)},
            "wrapper_trainable":sum(p.numel() for p in wp),"bridge_trainable":sum(p.numel() for p in bp),
            "generator_trainable":sum(p.numel() for p in gp),
            "compression_trainable":compression_trainable}
    (ROOT / "parts").mkdir(exist_ok=True)
    generator_after = {n: p.detach().cpu() for n, p in generator.named_parameters()}
    adaptation = {
        "initial_generator_hash": hashes["generator"],
        "final_generator_hash": result["final_hashes"]["generator"],
        "hash_changed": hashes["generator"] != result["final_hashes"]["generator"],
        "generator_parameter_norm_before": generator_norm_before,
        "generator_parameter_norm_after": math.sqrt(sum(float(p.double().square().sum()) for p in generator_after.values())),
        "generator_update_norm": math.sqrt(sum(float((p.double()-generator_initial[n].double()).square().sum()) for n,p in generator_after.items())),
        "generator_gradient_norm_mean": sum(float(r["generator_gradient_norm"]) for r in history)/len(history),
        "generator_gradient_norm_max": max(float(r["generator_gradient_norm"]) for r in history),
        "compression_hash_changed": hashes["compression"] != result["final_hashes"]["compression"],
    }
    (ROOT / "parts" / f"generator_update_audit_{args.branch}.json").write_text(json.dumps(adaptation, indent=2)+"\n")
    if adaptation["compression_hash_changed"] or not adaptation["hash_changed"] or len(history) != total:
        raise RuntimeError("final training audit failed")
    (ROOT / "parts" / f"train_{args.branch}_done.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
