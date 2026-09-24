"""Frozen historical bridge for V1.5; only per-frame O1/O2 variables may update."""
import concurrent.futures
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
V14, V13, OLD = Path(CONFIG["v14"]), Path(CONFIG["v13"]), Path(CONFIG["v12"])
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_uuid"]
sys.path.insert(0, str(OLD))
from support import (torch, nn, F, np, V1, V11, ROOT, FixedModels, lpips_model,
                     base_rows, manifest, source_frame, sha, tensor_hash, digest_state,
                     save_pt, rgb01, write_video, calc_psnr, calc_msssim_rgb,
                     load_models, decode_base, dpb_hash)
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
    def __init__(self, perceptual=True):
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

    def context(self, item):
        ec, q = item["ell_c"].cuda().detach(), item["q_recon"].cuda().detach()
        with torch.no_grad():
            d0 = self.net.decoder(torch.zeros(CONFIG["code_shape"], device="cuda"), ec).detach()
        return ec, q, d0

    def render(self, symbols, context):
        ec, q, d0 = context
        u = symbols.cuda().float() * DELTA
        residual = (self.net.decoder(u, ec) - d0) * self.net.residual_scale
        return rgb01(self.fixed.g(ec + residual, q)), residual


class Data:
    def __init__(self, bundle):
        self.bundle, self.base, self.contexts, self.timings = bundle, {}, {}, []

    def get(self, spec):
        if spec["sample"] in self.contexts:
            return self.contexts[spec["sample"]]
        started = time.perf_counter()
        if spec["video"] not in self.base:
            entry = next(e for e in entries(spec["split"]) if e["video_id"] == spec["video"])
            self.base[spec["video"]] = base_rows(entry)
        row = self.base[spec["video"]][spec["frame"]]
        with torch.no_grad():
            ec = self.bundle.fixed.ell_c(row["ell"].cuda().float()).cpu()
        item = {"ell_c": ec, "q_recon": row["q_recon"].float(), "row": row, "spec": spec}
        self.contexts[spec["sample"]] = item
        self.timings.append({"sample": spec["sample"], "base_cache_and_context_seconds": time.perf_counter() - started})
        return item


class Metrics:
    def __init__(self, bundle):
        self.bundle = bundle
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def submit(self, source, reconstruction):
        with torch.no_grad():
            lpips = self.bundle.lp(source, reconstruction, normalize=True).item()
        a, b = source.detach().cpu().numpy()[0], reconstruction.detach().cpu().numpy()[0]
        def calculate():
            mse = np.square(a.astype(np.float64) - b.astype(np.float64)).mean().item()
            return {"mse": mse, "psnr": calc_psnr(a, b, data_range=1),
                    "ms_ssim": float(calc_msssim_rgb(a, b, data_range=1)), "lpips": lpips,
                    "d": mse + CONFIG["lambda_p"] * lpips}
        return self.pool.submit(calculate)


def entries(split):
    data = manifest()
    return data["train2"] if split == "Train2" else data["val6"]


def specs(split):
    return [s for s in json.loads((V13 / "results_v1/samples.json").read_text()) if s["split"] == split]


def base_path(video):
    return V1 / "run_v1/base" / f"{video}_q0.bin"


def e_stream(split, video):
    prefix = "train2_full" if split == "Train2" else "val6"
    return OLD / "results_v1" / f"{prefix}_CODED_l3" / f"{video}.orc"


def json_dump(path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with path.open(mode) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def csv_dump(path, rows, replace=False):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    def parse(value):
        if value == "": return None
        if value == "True": return True
        if value == "False": return False
        try: return float(value)
        except ValueError: return value
    with path.open(newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
