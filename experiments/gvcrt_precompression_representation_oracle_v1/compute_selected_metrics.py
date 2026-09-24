#!/usr/bin/env python3
import argparse, csv, gc, json, math, os, sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
GVC = ROOT.parents[1]
SOURCE_ROOT = GVC / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames"
sys.path.insert(0, str(GVC / "expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src"))


def load_png(path, device):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float().div_(255).to(device)


def ms_ssim_rgb(a, b):
    c = a.shape[0]
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(x * x) / (2 * 1.5 * 1.5)); g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(c, 1, 11, 11)
    weights = torch.tensor([.0448, .2856, .3001, .2363, .1333], device=a.device, dtype=a.dtype)
    xx, yy, mss, mcs = a[None], b[None], [], []
    for level in range(5):
        ux = F.conv2d(xx, w, groups=c); uy = F.conv2d(yy, w, groups=c)
        vx = F.conv2d(xx * xx, w, groups=c) - ux * ux
        vy = F.conv2d(yy * yy, w, groups=c) - uy * uy
        cov = F.conv2d(xx * yy, w, groups=c) - ux * uy
        cs = (2 * cov + .03 ** 2) / (vx + vy + .03 ** 2)
        ss = ((2 * ux * uy + .01 ** 2) / (ux * ux + uy * uy + .01 ** 2)) * cs
        mss.append(ss.mean(dim=(0, 2, 3))); mcs.append(cs.mean(dim=(0, 2, 3)))
        if level < 4:
            xx = F.avg_pool2d(F.pad(xx, (0, 1, 0, 1), mode="reflect"), 2, 2)
            yy = F.avg_pool2d(F.pad(yy, (0, 1, 0, 1), mode="reflect"), 2, 2)
    vals = torch.stack(mcs[:-1]).clamp_min(1e-12).pow(weights[:-1, None]).prod(0)
    vals *= mss[-1].clamp_min(1e-12).pow(weights[-1])
    return float(vals.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--tags", required=True)
    args = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = torch.device("cuda:0")
    import lpips
    from DISTS_pytorch import DISTS
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
    di = DISTS().to(dev).eval().requires_grad_(False)
    rows = []
    for tag in args.tags.split(","):
        selected = list(csv.DictReader(open(ROOT / "parts" / f"{tag}_selected.csv")))
        streams = [("baseline", "", ROOT / "baseline_frames" / tag, None)]
        for r in selected:
            if r["target_reached"] == "True":
                streams.append(("selected", r["target_ratio"], ROOT / "reconstruction_frames" / tag / f"target_{r['target_ratio']}", ROOT / "proxy_frames" / tag / f"target_{r['target_ratio']}"))
        src = [load_png(SOURCE_ROOT / tag / f"im{i}.png", dev) for i in range(1, 17)]
        for kind, target, recdir, proxydir in streams:
            rec = [load_png(recdir / f"frame_{i:06d}.png", dev) for i in range(16)]
            ms, ppsnr, plp, pdi = [], [], [], []
            with torch.inference_mode():
                for i in range(16):
                    ms.append(ms_ssim_rgb(rec[i], src[i]))
                    if proxydir:
                        p = load_png(proxydir / f"frame_{i:06d}.png", dev)
                        mse = float((p - src[i]).square().mean())
                        ppsnr.append(-10 * math.log10(max(mse, 1e-15)))
                        plp.append(float(lp(p[None], src[i][None], normalize=True)))
                        pdi.append(float(di(p[None], src[i][None])))
            rows.append({"video_tag": tag, "stream_type": kind, "target_ratio": target,
                         "MS_SSIM": sum(ms) / len(ms),
                         "proxy_PSNR": sum(ppsnr) / len(ppsnr) if ppsnr else "",
                         "proxy_LPIPS": sum(plp) / len(plp) if plp else "",
                         "proxy_DISTS": sum(pdi) / len(pdi) if pdi else ""})
            print(tag, kind, target, "spatial", flush=True)
        del src; torch.cuda.empty_cache()
    del lp, di; gc.collect(); torch.cuda.empty_cache()
    from flolpips_metric import flolpips_for_videos
    from flolpips_compat import load_models, reference_flows
    flow, flp = load_models(dev)
    for row in rows:
        tag = row["video_tag"]
        src = torch.stack([load_png(SOURCE_ROOT / tag / f"im{i}.png", dev) for i in range(1, 17)])
        if row["stream_type"] == "baseline": recdir = ROOT / "baseline_frames" / tag
        else: recdir = ROOT / "reconstruction_frames" / tag / f"target_{row['target_ratio']}"
        rec = torch.stack([load_png(recdir / f"frame_{i:06d}.png", dev) for i in range(16)])
        flows = reference_flows(src, flow)
        score, _ = flolpips_for_videos(src, rec, flow, flp, flows)
        row["FloLPIPS"] = float(score)
        if not all(math.isfinite(float(row[k])) for k in ("MS_SSIM", "FloLPIPS")):
            raise RuntimeError(f"nonfinite metrics {row}")
        del src, rec, flows; torch.cuda.empty_cache()
        print(tag, row["stream_type"], row["target_ratio"], "FloLPIPS", flush=True)
    out = ROOT / "parts" / f"selected_metrics_gpu{args.gpu}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
