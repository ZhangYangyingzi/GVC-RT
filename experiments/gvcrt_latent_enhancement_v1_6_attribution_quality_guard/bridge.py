"""Read-only bridge to the frozen V1.5 experiment and historical inputs."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
CONFIG = json.loads((HOME / "config.json").read_text())
V15 = Path(CONFIG["v15"])
V15_RESULTS = V15 / "results_v1"
V12 = Path(CONFIG["v12"])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_uuid"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V15))
spec = importlib.util.spec_from_file_location("gvcrt_v15_bridge", V15 / "bridge.py")
v15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v15)
torch, np, ORC2 = v15.torch, v15.np, v15.ORC2
MODEL, DELTA, PIXELS = v15.MODEL, v15.DELTA, v15.PIXELS
Bundle, Data, Metrics = v15.Bundle, v15.Data, v15.Metrics
entries, specs, base_path = v15.entries, v15.specs, v15.base_path
source_frame, base_rows, rgb01 = v15.source_frame, v15.base_rows, v15.rgb01
write_video, tensor_hash, dpb_hash = v15.write_video, v15.tensor_hash, v15.dpb_hash
load_models, decode_base, setup = v15.load_models, v15.decode_base, v15.setup
V1, V11, OLD, V13, V14 = v15.V1, v15.V11, v15.OLD, v15.V13, v15.V14

allocator_spec = importlib.util.spec_from_file_location("gvcrt_v15_allocator", V15 / "allocator.py")
allocator = importlib.util.module_from_spec(allocator_spec)
allocator_spec.loader.exec_module(allocator)
exact_frame_dp, self_test = allocator.exact_frame_dp, allocator.self_test

OVERHEAD_BYTES = ORC2.HEADER.size + 1
ORDER = CONFIG["candidate_order"]


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


def csv_dump(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(split, video):
    entry = next(item for item in entries(split) if item["video_id"] == video)
    return float(entry["duration_seconds"])


def load_video_candidates(split, video, bundle):
    stream = v15.e_stream(split, video)
    decoded = ORC2.decode(stream, base_path(video), MODEL, bundle.net.entropy)
    result = {}
    for sample in sorted((s for s in specs(split) if s["video"] == video), key=lambda s: s["frame"]):
        if split == "Train2":
            o1 = V14 / "results_v1/optimizations/adaptive_lambda1" / sample["sample"] / "selected.pt"
            tag = "B6q_lambda3" if sample["group"] == "network_train6" else "Bfull_lambda3"
            o2 = V13 / "results_v1/optimizations" / tag / sample["sample"] / "selected.pt"
        else:
            o1 = V15_RESULTS / "optimizations/O1" / sample["sample"] / "selected.pt"
            o2 = V15_RESULTS / "optimizations/O2" / sample["sample"] / "selected.pt"
        result[sample["frame"]] = {
            "Z": torch.zeros(CONFIG["code_shape"], dtype=torch.int16),
            "E": decoded["frames"][sample["frame"]].short(),
            "O1": torch.load(o1, map_location="cpu", weights_only=False)["symbols"].short(),
            "O2": torch.load(o2, map_location="cpu", weights_only=False)["symbols"].short(),
        }
    return result
