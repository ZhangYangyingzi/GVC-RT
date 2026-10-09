#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

from common import ROOT, read_json, split, torch_load, write_json


def convolution_macs(height, width, input_channels, output_channels, kernel=1, groups=1):
    return height * width * output_channels * (input_channels // groups) * kernel * kernel


def residual_stack_macs(height, width, input_channels, hidden_channels, output_channels):
    projection = convolution_macs(height, width, input_channels, hidden_channels)
    block = (convolution_macs(height, width, hidden_channels, hidden_channels, 3, hidden_channels) +
             convolution_macs(height, width, hidden_channels, hidden_channels))
    head = convolution_macs(height, width, hidden_channels, output_channels)
    return projection + 3 * block + head


def main():
    frozen = read_json(ROOT / "checkpoints/final/SELECTIONS_FROZEN.json")
    method_info = {"m0": {"trainable_parameters": 0, "stored_parameters": 0,
                           "checkpoint_mb": 0.0}}
    for method, selected in frozen["selected"].items():
        payload = torch_load(selected["path"], map_location="cpu", weights_only=False)
        method_info[method] = {
            "trainable_parameters": int(payload["trainable_parameters"]),
            "stored_parameters": int(payload["stored_parameters"]),
            "checkpoint_mb": Path(selected["path"]).stat().st_size / 1e6}
    method_info["m6"] = {"trainable_parameters": 0, "stored_parameters": 19474,
                         "checkpoint_mb": Path(read_json(ROOT / "config.json")["historical_v8_checkpoint"]).stat().st_size / 1e6}
    timings = {method: [] for method in method_info}
    peak_memory = {method: [] for method in method_info}
    for video in split()["test"]:
        name = Path(video["path"]).stem
        for qp in range(4):
            path = ROOT / f"analysis/test_pairs/{name}_qp{qp}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for method, seconds in payload["seconds"].items():
                timings.setdefault(method, []).append(float(seconds) / payload["frames"])
            for method, value in payload["peak_incremental_memory_bytes"].items():
                peak_memory.setdefault(method, []).append(int(value))
    for method, values in timings.items():
        method_info.setdefault(method, {})["mean_candidate_seconds_per_frame"] = (
            float(np.mean(values)) if values else None)
        method_info[method]["peak_incremental_memory_bytes"] = (
            max(peak_memory.get(method, []), default=None))
    method_info["m1"].update({"added_structural_parameters": 0,
                              "note": "801,124 copied bridge weights are modified/stored; no new architecture parameters."})
    method_info["m2"]["added_structural_parameters"] = 19237
    method_info["m3"]["added_structural_parameters"] = 19545
    method_info["m4"]["added_structural_parameters"] = 19474
    method_info["m5"].update({"added_structural_parameters": 19299,
                              "note": "Runs at full 1920x1088 RGB resolution; matched parameters do not imply matched MACs."})
    module_macs = {
        "m2": (convolution_macs(136, 240, 256, 37) +
               convolution_macs(136, 240, 37, 256)),
        "m3": residual_stack_macs(68, 120, 18, 69, 18),
        "m4": residual_stack_macs(68, 120, 54, 64, 18),
        "m5": residual_stack_macs(1088, 1920, 9, 72, 3),
        "m6": residual_stack_macs(68, 120, 54, 64, 18),
    }
    method_info["m1"]["added_structural_module_macs"] = 0
    for method, macs in module_macs.items():
        method_info[method]["added_structural_module_macs"] = int(macs)
    payload = {"scope": "candidate branch execution measured during full-frame test rendering",
               "macs_scope": "analytic multiply-accumulates for added structural module only; excludes the shared generator and M1 copied bridge re-execution",
               "methods": method_info, "additional_transmitted_bits": 0}
    write_json(ROOT / "analysis/complexity.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
