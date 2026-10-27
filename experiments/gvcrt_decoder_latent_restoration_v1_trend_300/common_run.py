"""Pinned local dependency loading and atomic experiment receipts."""
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = Path('/Huang_group/zyyz/Projects/GVC-RT')
SNAP = HERE / 'dependency_snapshot'
PYTHON = '/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
A_PATH = ORIGINAL / 'experiments/gvcrt_generation_aware_quant_visual_condition_v1_trend_300/server_artifacts/A/step_300.pt'
G_BASIC = ORIGINAL / 'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/evaluation/checkpoints/G_basic_100.pt'
SEED = 20261027


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def table(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in records for k in r))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def setup(gpu):
    assert gpu in (1, 4)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.path[:0] = [str(SNAP), str(REPO), str(REPO / 'experiments/gvcrt_neural_wrapper_v2_joint'),
                    str(REPO / 'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies')]
    import torch
    import numpy as np
    module('common', SNAP / 'common.py')
    module('gvc_hooks', SNAP / 'gvc_hooks.py')
    module('runtime', SNAP / 'runtime.py')
    module('flolpips_compat', SNAP / 'flolpips_compat.py')
    module('core', REPO / 'experiments/gvcrt_neural_wrapper_v2_joint/core.py')
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.device_count() == 1
    return torch, np


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(sd):
    h = hashlib.sha256()
    for k, t in sorted(sd.items()):
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_base(torch):
    assert sha(A_PATH) == '17a7635ae6e4f74922b5bfe368a790bf2f48b3a7bff2b2f516c4dea24148404f'
    for path, entry in load(HERE / 'weight_manifest.json').items():
        assert sha(path) == entry['sha256'] and Path(path).stat().st_size == entry['bytes']
    from gvc_hooks import load_models
    from runtime import upgrade
    im, pm = load_models(torch.device('cuda:0'), force_zero_thres=.12)
    basic = torch.load(G_BASIC, map_location='cpu', weights_only=True)
    assert not basic['wrapper']
    pm.recon_generation_net.mlp.load_state_dict(basic['bridge'], strict=True)
    pm.recon_generation_net.decoder.load_state_dict(basic['generator'], strict=True)
    pm = upgrade(pm, 'A')
    pm.recon_generation_net.mlp.float()
    pm.recon_generation_net.decoder.float()
    initial = torch.load(A_PATH, map_location='cpu', weights_only=False)
    assert initial['branch'] == 'A' and initial['step'] == 300
    pm.recon_generation_net.decoder.load_state_dict(initial['G'], strict=True)
    im.eval().requires_grad_(False); pm.eval().requires_grad_(False)
    return im, pm


def rng(torch, np):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(s, torch, np):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state(s['cuda'])


def save_pt(torch, path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
