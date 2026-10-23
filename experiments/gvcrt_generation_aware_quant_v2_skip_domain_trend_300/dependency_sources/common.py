import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
REPO = ROOT.parents[1]
V4 = RESEARCH / "expericent_perceptual_generator_refiner_v4"
V6 = RESEARCH / "expericent_perceptual_temporal_refiner_v6"
V8 = RESEARCH / "expericent_residual_strength_control_v8"
QPS = (0, 1, 2, 3)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config():
    return read_json(ROOT / "config.json")


def split():
    return read_json(ROOT / "v9_split.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path, **kwargs):
    import torch
    try:
        return torch.load(path, weights_only=kwargs.pop("weights_only", False), **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def fid_terms(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    first_mean, second_mean = first.mean(0), second.mean(0)
    first_centered, second_centered = first - first_mean, second - second_mean
    cross = (first_centered @ second_centered.T) / math.sqrt((len(first) - 1) * (len(second) - 1))
    mean_term = float(np.square(first_mean - second_mean).sum())
    covariance_term = (float(np.square(first_centered).sum() / (len(first) - 1)) +
                       float(np.square(second_centered).sum() / (len(second) - 1)) -
                       2 * float(np.linalg.svd(cross, compute_uv=False).sum()))
    return {"fid": float(max(mean_term + covariance_term, 0.0)),
            "mean_term": mean_term, "covariance_term": covariance_term}


def kid_unbiased(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    scale = first.shape[1]
    k_xx = (first @ first.T / scale + 1) ** 3
    k_yy = (second @ second.T / scale + 1) ** 3
    k_xy = (first @ second.T / scale + 1) ** 3
    n, m = len(first), len(second)
    return float((k_xx.sum() - np.trace(k_xx)) / (n * (n - 1)) +
                 (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1)) - 2 * k_xy.mean())
