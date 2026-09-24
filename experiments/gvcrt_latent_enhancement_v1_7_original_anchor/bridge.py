"""V1.7 original-anchor bridge over frozen historical inputs."""
import csv
import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent
CONFIG = json.loads((HOME / "config.json").read_text())
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_uuid"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

V15 = Path(CONFIG["v15"]); V16 = Path(CONFIG["v16"])
V14 = Path(CONFIG["v14"]); V13 = Path(CONFIG["v13"]); V12 = Path(CONFIG["v12"])
V15_RESULTS = V15 / "results_v1"; V16_RESULTS = V16 / "results_v1"

sys.path.insert(0, str(V15))
spec = importlib.util.spec_from_file_location("gvcrt_v15_bridge_for_v17", V15 / "bridge.py")
v15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v15)
alloc_spec = importlib.util.spec_from_file_location("gvcrt_v15_allocator_for_v17", V15 / "allocator.py")
allocator = importlib.util.module_from_spec(alloc_spec); alloc_spec.loader.exec_module(allocator)

torch, nn, F, np, ORC2 = v15.torch, v15.nn, v15.F, v15.np, v15.ORC2
Bundle, Data, Metrics = v15.Bundle, v15.Data, v15.Metrics
entries, specs, base_path = v15.entries, v15.specs, v15.base_path
source_frame, base_rows, rgb01 = v15.source_frame, v15.base_rows, v15.rgb01
write_video, tensor_hash, dpb_hash = v15.write_video, v15.tensor_hash, v15.dpb_hash
load_models, decode_base, setup = v15.load_models, v15.decode_base, v15.setup
MODEL, DELTA, PIXELS = v15.MODEL, v15.DELTA, v15.PIXELS
V1, V11 = v15.V1, v15.V11
exact_frame_dp, self_test = allocator.exact_frame_dp, allocator.self_test

WRAP = struct.Struct("<4sI")
PROFILE_MAGIC = CONFIG["wrapper_magic"].encode("ascii")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def csv_dump(path, rows, replace=False):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def json_dump(path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(split, video):
    return float(next(e for e in entries(split) if e["video_id"] == video)["duration_seconds"])


def alpha_label(alpha):
    return str(alpha).replace(".", "p")


def legacy_symbol_path(split, candidate, sample):
    if split == "Val6":
        return V15_RESULTS / "optimizations" / candidate / sample / "selected.pt"
    spec_row = next(s for s in specs("Train2") if s["sample"] == sample)
    if candidate == "O1":
        return V14 / "results_v1/optimizations/adaptive_lambda1" / sample / "selected.pt"
    tag = "B6q_lambda3" if spec_row["group"] == "network_train6" else "Bfull_lambda3"
    return V13 / "results_v1/optimizations" / tag / sample / "selected.pt"


def load_legacy_symbols(split, candidate, sample):
    path = legacy_symbol_path(split, candidate, sample)
    if not path.exists():
        return None, str(path), "missing"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["symbols"].short(), str(path), "found"


def original_context(bundle, row):
    ell_b = row["ell"].cuda().float().detach()
    ell_c = bundle.fixed.ell_c(ell_b).detach()
    q = row["q_recon"].cuda().float().detach()
    return ell_b, ell_c, q


def render_original_anchor(bundle, symbols, row):
    ell_b, ell_c, q = original_context(bundle, row)
    residual = bundle.net.synthesize(symbols.cuda().float() * DELTA, ell_c)
    return rgb01(bundle.fixed.g(ell_b + residual, q)), residual


def render_base_original(bundle, row):
    return rgb01(bundle.fixed.g(row["ell"].cuda().float(), row["q_recon"].cuda().float()))


def render_zero_legacy(bundle, row):
    ell_c = bundle.fixed.ell_c(row["ell"].cuda().float())
    return rgb01(bundle.fixed.g(ell_c, row["q_recon"].cuda().float()))


def encode_inner(tmp_path, frames, video, bundle):
    return ORC2.encode(tmp_path, base_path(video), MODEL, 1, CONFIG["level"], DELTA,
                       tuple(CONFIG["code_shape"][1:]), frames, bundle.net.entropy)


def wrap_stream(inner_path, output_path):
    payload = Path(inner_path).read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("xb") as handle:
        handle.write(WRAP.pack(PROFILE_MAGIC, len(payload)))
        handle.write(payload)
    return Path(output_path).stat().st_size


def unwrap_stream(path):
    blob = Path(path).read_bytes()
    if len(blob) < WRAP.size:
        raise ValueError("Truncated original-anchor wrapper")
    magic, length = WRAP.unpack_from(blob)
    if magic != PROFILE_MAGIC or length != len(blob) - WRAP.size:
        raise ValueError("Wrong original-anchor profile")
    return blob[WRAP.size:]


def decode_profiled(path, video, bundle, tmp_dir):
    payload = unwrap_stream(path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / (Path(path).name + ".inner.orc2")
    tmp.write_bytes(payload)
    try:
        return ORC2.decode(tmp, base_path(video), MODEL, bundle.net.entropy)
    finally:
        tmp.unlink(missing_ok=True)


def packet_bytes_for_symbols(symbols, split, video, frame, bundle, tmp_dir):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"packet_{video}_{frame}_{time.time_ns()}.orc2"
    frames = [None] + [torch.zeros(CONFIG["code_shape"], dtype=torch.int16) for _ in range(15)]
    frames[frame] = symbols.cpu().short()
    coded = encode_inner(tmp, frames, video, bundle)
    packet = coded["frames"][frame]["actual_bits"] // 8
    tmp.unlink(missing_ok=True)
    return int(packet)


def profiled_overhead_bytes():
    return WRAP.size + ORC2.HEADER.size + 1
