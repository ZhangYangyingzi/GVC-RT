"""Writes are confined to this experiment; historical modules are read only."""
import hashlib
import json
import os
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXPS = REPO / 'experiments'
V29 = EXPS / 'gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control'
V31 = EXPS / 'gvcrt_neural_wrapper_v6_31_p_color_loss_effective_update_audit'
V13 = EXPS / 'gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation'
B1 = EXPS / 'gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/branches/B1_reconstruction/checkpoints/adaptation_step_1000.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    p = ROOT / name
    assert p.resolve().is_relative_to(ROOT)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, tensor in state.items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def write(name, rows, fields=None):
    p = ROOT / name
    assert p.resolve().is_relative_to(ROOT)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.with_suffix('.tmp').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    p.with_suffix('.tmp').replace(p)
