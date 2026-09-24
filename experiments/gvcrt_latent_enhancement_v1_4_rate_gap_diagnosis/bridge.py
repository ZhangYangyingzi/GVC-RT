"""Read-only V1-V1.3 bridge. V1.4 may only create code variables and new artifacts."""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HOME = Path(__file__).resolve().parent
CONFIG = json.loads((HOME / "config.json").read_text())
V13 = Path(CONFIG["v13"])
OLD = Path(CONFIG["v12"])
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_uuid"]
sys.path.insert(0, str(OLD))
from support import (torch, nn, F, np, V1, V11, ROOT, FixedModels, lpips_model,
                     base_rows, manifest, source_frame, sha, tensor_hash, digest_state,
                     save_json, save_pt, write_csv, read_csv, rgb01, image, write_video,
                     calc_psnr, calc_msssim_rgb, load_models, decode_base, dpb_hash)
from model import ResidualCodec
import bitstream as ORC2

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_uuid"]
MODEL = OLD / CONFIG["model"]
DELTA = CONFIG["delta"]
PIXELS = CONFIG["valid_hw"][0] * CONFIG["valid_hw"][1]


def setup():
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class Bundle:
    def __init__(self, perceptual=False):
        self.fixed = FixedModels()
        ck = torch.load(MODEL, map_location="cpu", weights_only=False)
        assert ck["step"] == 1500 and ck["deltas"][3] == DELTA
        self.net = ResidualCodec(ck["residual_scale"]).cuda().eval().requires_grad_(False)
        self.net.load_state_dict(ck["model"], strict=True)
        self.lp = lpips_model() if perceptual else None
        self.fingerprints = self.hashes()

    def hashes(self):
        return {"fixed": self.fixed.hashes(), "residual_codec": digest_state(self.net)}

    def check(self):
        assert self.hashes() == self.fingerprints
        self.fixed.assert_frozen()
        assert all(not p.requires_grad and p.grad is None for p in self.net.parameters())

    def context(self, data):
        ec = data["ell_c"].cuda().detach()
        q = data["q_recon"].cuda().detach()
        with torch.no_grad():
            d0 = self.net.decoder(torch.zeros(CONFIG["code_shape"], device="cuda"), ec).detach()
        return ec, q, d0

    def render(self, symbols, context):
        ec, q, d0 = context
        u = symbols.cuda().float() * DELTA
        r = (self.net.decoder(u, ec) - d0) * self.net.residual_scale
        return rgb01(self.fixed.g(ec + r, q)), r


def entries(split):
    m = manifest()
    return m["train2"] if split == "Train2" else m["val6"]


def specs(split="Train2"):
    return [s for s in json.loads((V13 / "results_v1/samples.json").read_text()) if s["split"] == split]


def base_path(video):
    return V1 / "run_v1/base" / f"{video}_q0.bin"


def historical_code_path(index, spec, filename="selected.pt"):
    stage = f"B6q_lambda{index}" if spec["group"] == "network_train6" else f"Bfull_lambda{index}"
    return V13 / "results_v1/optimizations" / stage / spec["sample"] / filename


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def csv_dump(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
