"""Immutable historical inputs and shared reconstruction path for V1.2."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"config.json").read_text())
V1=Path(CFG["v1"])
V11=Path(CFG["v11"])
ROOT=HERE.parents[1]
os.environ["CUDA_VISIBLE_DEVICES"]=CFG["gpu_uuid"]
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(V1))
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from codec import strict_load,digest_state,tensor_hash,source_frame,decode_base,load_models,dpb_hash
from enhancement import Enhancement,BaseOnly,Residual,Synthesis,FactorizedEntropy,arithmetic_encode,arithmetic_decode
from evaluation import rgb01,write_video
from src.models.video_model_gvcrt import DMC
from src.utils.metrics import calc_psnr,calc_msssim_rgb


def configure():
    torch.set_num_threads(CFG["torch_threads"])
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def save_json(path,data):
    with Path(path).open("x") as f:
        json.dump(data,f,indent=2,allow_nan=False)


def save_pt(path,data):
    with Path(path).open("xb") as f:
        torch.save(data,f)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            if v=="":
                r[k]=None
            elif v in ["True","False"]:
                r[k]=v=="True"
            else:
                try:
                    r[k]=float(v)
                except ValueError:
                    pass
    return rows


def image(path,x):
    a=(x.detach().cpu()[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
    Image.fromarray(a).save(path)


def quality(x,y,lp):
    a,b=x.detach().float().cpu().numpy()[0],y.detach().float().cpu().numpy()[0]
    with torch.no_grad():
        p=lp(x.cuda().float(),y.cuda().float(),normalize=True).item()
    return {"psnr":calc_psnr(a,b,data_range=1),"ms_ssim":float(calc_msssim_rgb(a,b,data_range=1)),"lpips":p,
            "mse":float(np.square(a.astype(np.float64)-b.astype(np.float64)).mean())}


class FixedModels:
    def __init__(self):
        vcfg=json.loads((V1/"config.json").read_text())
        d=torch.load(V1/CFG["baseline_checkpoint"],map_location="cpu",weights_only=False)
        c=torch.load(V1/CFG["control_C_checkpoint"],map_location="cpu",weights_only=False)
        self.zero_D=Enhancement(vcfg["architecture"]).float().cuda().eval().requires_grad_(False)
        self.zero_D.load_state_dict(d["enhancement"],strict=True)
        self.zero_C=Enhancement(vcfg["architecture"]).float().cuda().eval().requires_grad_(False)
        self.zero_C.load_state_dict(c["enhancement"],strict=True)
        self.control=BaseOnly(vcfg["architecture"]).float().cuda().eval().requires_grad_(False)
        self.control.load_state_dict(c["base_only"],strict=True)
        p=DMC()
        strict_load(p,ROOT/"checkpoints/GVC-RT_P.pt","P")
        self.g=p.recon_generation_net.decoder.float().cuda().eval().requires_grad_(False)
        self.fingerprint=self.hashes()

    def hashes(self):
        return {k:digest_state(getattr(self,k)) for k in ["zero_D","zero_C","control","g"]}

    def assert_frozen(self):
        assert self.hashes()==self.fingerprint
        for k in self.fingerprint:
            assert all(p.grad is None and not p.requires_grad for p in getattr(self,k).parameters())

    def ell_c(self,ell):
        return ell+self.zero_D.decoder(torch.zeros((1,8,17,30),device=ell.device,dtype=ell.dtype),ell)

    def old_start(self,ell,initialization):
        if initialization=="ORIGINAL_START":
            return ell
        if initialization=="ZERO_START":
            return ell+self.zero_C.decoder(torch.zeros((1,8,17,30),device=ell.device,dtype=ell.dtype),ell)
        raise ValueError(initialization)


def lpips_model():
    if not (Path(torch.hub.get_dir())/"checkpoints/alexnet-owt-7be5be79.pth").exists():
        raise RuntimeError("Missing local LPIPS weights; download disabled")
    import lpips
    return lpips.LPIPS(net="alex",version="0.1").cuda().eval().requires_grad_(False)


def base_rows(entry,qp=0):
    return torch.load(V1/"run_v1/base"/f'{entry["video_id"]}_q{qp}.pt',map_location="cpu",weights_only=False)


def manifest():
    m=json.loads((V11/"manifest.json").read_text())
    assert len(m["train2"])==2 and len(m["val6"])==6
    return m


def eligible(candidate,baseline):
    return candidate["psnr"]>baseline["psnr"] and candidate["lpips"]<=baseline["lpips"]
