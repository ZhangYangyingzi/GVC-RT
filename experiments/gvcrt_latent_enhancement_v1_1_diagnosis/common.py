"""Read-only reuse of V1 models/metrics; all new artifacts stay in V1.1."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/"config.json").read_text())
V1 = Path(CFG["v1_directory"])
ROOT = HERE.parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["gpu_uuid"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0,str(V1))
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from PIL import Image
from codec import strict_load, digest_state, tensor_hash, source_frame, load_models, decode_base, dpb_hash
from enhancement import Enhancement, BaseOnly, FactorizedEntropy
from evaluation import rgb01, read_container, write_video
from src.utils.metrics import calc_msssim_rgb, calc_psnr
from src.models.video_model_gvcrt import DMC


def configure():
    torch.set_num_threads(CFG["torch_threads"])
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def save_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def models(checkpoint, device="cuda"):
    vcfg=json.loads((V1/"config.json").read_text())
    ck=torch.load(V1/checkpoint,map_location="cpu",weights_only=False)
    e=Enhancement(vcfg["architecture"])
    control=BaseOnly(vcfg["architecture"])
    entropy=FactorizedEntropy(vcfg["architecture"]["enhancement_channels"])
    e.load_state_dict(ck["enhancement"],strict=True)
    control.load_state_dict(ck["base_only"],strict=True)
    entropy.load_state_dict(ck["entropy"],strict=True)
    for m in (e,control,entropy):
        m.eval().requires_grad_(False)
    return e.to(device),control.to(device),entropy,ck


def generator():
    m=DMC()
    strict_load(m,ROOT/"checkpoints/GVC-RT_P.pt","P")
    return m.recon_generation_net.decoder.float().cuda().eval().requires_grad_(False)


def lpips_model():
    if not (Path(torch.hub.get_dir())/"checkpoints/alexnet-owt-7be5be79.pth").exists():
        raise RuntimeError("LPIPS weights unavailable; download prohibited")
    import lpips
    return lpips.LPIPS(net="alex",version="0.1").cuda().eval().requires_grad_(False)


def quality(x,y,perceptual):
    # Same functions/range/order as V1 evaluation. Threading only moves CPU work.
    a,b=x.detach().float().cpu().numpy()[0],y.detach().float().cpu().numpy()[0]
    with torch.no_grad():
        lp=perceptual(x.cuda().float(),y.cuda().float(),normalize=True).item()
    return {"mse":float(np.mean(np.square(a.astype(np.float64)-b.astype(np.float64)))),
            "psnr":calc_psnr(a,b,data_range=1),"ms_ssim":float(calc_msssim_rgb(a,b,data_range=1)),"lpips":lp}


def stats(t,prefix):
    t=t.detach().double()
    return {prefix+"_mean":t.mean().item(),prefix+"_std":t.std(unbiased=False).item(),
            prefix+"_l2":t.norm().item(),prefix+"_rms":t.square().mean().sqrt().item(),
            prefix+"_max_abs":t.abs().max().item()}


def image(path,t):
    a=(t.detach().cpu()[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
    Image.fromarray(a).save(path)


def difference(a,b):
    return {"equal":torch.equal(a,b),"max_abs":(a-b).abs().max().item()}


def dequantize(symbols,delta):
    # Exact V1 receiver expression, deliberately no centering or learned scale.
    return symbols.cuda().float()*delta
