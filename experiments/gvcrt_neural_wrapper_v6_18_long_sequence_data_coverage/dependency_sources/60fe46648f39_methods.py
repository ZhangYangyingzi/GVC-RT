import time
from pathlib import Path

import torch

from bridge_adapter import FeatureAdapter
from common import ROOT, config, sha256, torch_load
from copied_bridge import CopiedBridge
from current_only_refiner import CurrentOnlyRefiner
from gvc_hooks import decoder_for
from rgb_postprocessor import RGBPostprocessor
from temporal_refiner import TemporalRefiner


METHOD_DIRS = {"m1": "m1_bridge", "m2": "m2_bridge_adapter", "m3": "m3_current_only",
               "m4": "m4_temporal", "m5": "m5_rgb_post"}


def build_method(name, codec_models, device):
    if name == "m1":
        model = CopiedBridge(codec_models[1].recon_generation_net.mlp, train_bridge=True)
    elif name == "m2":
        model = CopiedBridge(codec_models[1].recon_generation_net.mlp, train_bridge=False,
                             adapter=FeatureAdapter())
    elif name == "m3":
        model = CurrentOnlyRefiner()
    elif name == "m4":
        model = TemporalRefiner()
    elif name == "m5":
        model = RGBPostprocessor()
    else:
        raise ValueError(name)
    return model.to(device)


def load_candidate(name, path, codec_models, device):
    model = build_method(name, codec_models, device)
    payload = torch_load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False), payload


def load_historical_v8(device):
    model = TemporalRefiner()
    path = Path(config()["historical_v8_checkpoint"])
    if sha256(path) != config()["historical_v8_sha256"]:
        raise RuntimeError("Historical V8 checkpoint hash mismatch")
    payload = torch_load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def candidate_specs():
    specs = []
    for method, directory in METHOD_DIRS.items():
        for path in sorted((ROOT / "checkpoints" / directory).glob("lr_*/update_*.pt")):
            payload = torch_load(path, map_location="cpu", weights_only=False)
            specs.append({"method": method, "candidate": f"{method}_{path.parent.name}_{path.stem}",
                          "path": str(path), "lr": float(payload["lr"]),
                          "update": int(payload["update"]),
                          "checkpoint_sha256": sha256(path),
                          "trainable_parameters": int(payload["trainable_parameters"]),
                          "stored_parameters": int(payload["stored_parameters"])})
    return specs


@torch.inference_mode()
def baseline_image(record, codec_models, device):
    return decoder_for(record["frame_type"], codec_models)(
        record["codeword"].to(device), record["quant"].to(device))


@torch.inference_mode()
def render(name, model, record, previous_record, baseline, previous_baseline,
           codec_models, device):
    torch.cuda.synchronize()
    started = time.perf_counter()
    if name in ("m1", "m2"):
        if record["frame_type"] == "I":
            output = baseline
        else:
            codeword = model(record["bridge_feature"].to(device).float())
            output = decoder_for(record["frame_type"], codec_models)(
                codeword.half(), record["quant"].to(device))
    elif name in ("m3", "m4", "m6"):
        current = record["codeword"].to(device).float()
        previous = torch.zeros_like(current) if previous_record is None else \
            previous_record["codeword"].to(device).float()
        codeword = model(current, previous)["codeword"]
        output = decoder_for(record["frame_type"], codec_models)(
            codeword.half(), record["quant"].to(device))
    elif name == "m5":
        previous = torch.zeros_like(baseline) if previous_baseline is None else previous_baseline
        output = model(baseline.float(), previous.float())["image"]
    else:
        raise ValueError(name)
    torch.cuda.synchronize()
    return output, time.perf_counter() - started
