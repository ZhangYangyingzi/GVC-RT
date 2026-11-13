"""Modern PyTorch compatibility layer for the official FloLPIPS implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OFFICIAL_ROOT = ROOT / "third_party/flolpips"
OFFICIAL_COMMIT = "66dd39937e961c56cf162651074eacbfb5f9aab1"
OFFICIAL_SOURCE_SHA256 = "2f1c976ca4d63ef9545cd6032c081868dda19245602f2f9bd584ae4685056e34"
OFFICIAL_PWC_SOURCE_SHA256 = "9a0b01f10fa0625c2a0f5d26cb859cd47174747588d38e8bd1f270ed0cf3309d"
OFFICIAL_ALEX_WEIGHT_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
PWC_WEIGHTS = Path("/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/pwc-network-default.pytorch")
PWC_WEIGHT_SHA256 = "ad2dc62b63b9d10ad42a6969179b3f996d4115e2ec6421bf532f99cf278ffcd2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correlation_volume(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Numerically equivalent forward path for the official 9x9 CuPy cost volume."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("FloLPIPS correlation inputs must have identical BCHW shapes")
    height, width = first.shape[-2:]
    padded = F.pad(second, (4, 4, 4, 4))
    return torch.cat([(first * padded[:, :, y:y + height, x:x + width]).mean(1, keepdim=True)
                      for y in range(9) for x in range(9)], dim=1)


class _PureTorchCorrelation:
    @staticmethod
    def FunctionCorrelation(tenFirst: torch.Tensor, tenSecond: torch.Tensor) -> torch.Tensor:
        return correlation_volume(tenFirst, tenSecond)


def load_models(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    audits = ((OFFICIAL_ROOT / "flolpips.py", OFFICIAL_SOURCE_SHA256),
              (OFFICIAL_ROOT / "pwcnet.py", OFFICIAL_PWC_SOURCE_SHA256),
              (OFFICIAL_ROOT / "weights/v0.1/alex.pth", OFFICIAL_ALEX_WEIGHT_SHA256),
              (PWC_WEIGHTS, PWC_WEIGHT_SHA256))
    for path, expected in audits:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"FloLPIPS provenance changed: {path}")
    correlation_package = types.ModuleType("correlation")
    correlation_package.correlation = _PureTorchCorrelation
    sys.modules["correlation"] = correlation_package
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from flolpips import FloLPIPS  # type: ignore[import-not-found]  # noqa: PLC0415
        from pwcnet import Network  # type: ignore[import-not-found]  # noqa: PLC0415
        original_loader = torch.hub.load_state_dict_from_url
        torch.hub.load_state_dict_from_url = lambda *args, **kwargs: torch.load(
            PWC_WEIGHTS, map_location="cpu", weights_only=True)
        try:
            flow = Network()
        finally:
            torch.hub.load_state_dict_from_url = original_loader
        perceptual = FloLPIPS(net="alex", version="0.1")
    finally:
        sys.path.pop(0)
    return flow.to(device).eval().requires_grad_(False), perceptual.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def reference_flows(reference: torch.Tensor, flow_model: torch.nn.Module) -> list[torch.Tensor]:
    if reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("FloLPIPS reference must be a TCHW tensor")
    return [flow_model(reference[index:index + 1], reference[index + 1:index + 2]).cpu()
            for index in range(reference.shape[0] - 1)]


@torch.inference_mode()
def video_flolpips(reference: torch.Tensor, candidate: torch.Tensor,
                   flow_model: torch.nn.Module, perceptual_model: torch.nn.Module,
                   cached_reference_flows: list[torch.Tensor] | None = None) -> tuple[float, list[float]]:
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("FloLPIPS videos must be identically shaped TCHW tensors")
    if cached_reference_flows is not None and len(cached_reference_flows) != reference.shape[0] - 1:
        raise ValueError("cached FloLPIPS reference-flow count is invalid")
    values = []
    for index in range(reference.shape[0] - 1):
        ref_now, ref_next = reference[index:index + 1], reference[index + 1:index + 2]
        dis_now, dis_next = candidate[index:index + 1], candidate[index + 1:index + 2]
        ref_flow = (flow_model(ref_now, ref_next) if cached_reference_flows is None
                    else cached_reference_flows[index].to(reference.device))
        flow_difference = ref_flow - flow_model(dis_now, dis_next)
        magnitude_sum = flow_difference.square().sum(1, keepdim=True).sqrt().sum()
        if not torch.isfinite(magnitude_sum) or magnitude_sum <= 0:
            raise RuntimeError("official FloLPIPS motion weighting is undefined for zero flow difference")
        score = perceptual_model(ref_now, dis_now, flow_difference, normalize=True)
        if score.numel() != 1 or not torch.isfinite(score).all():
            raise RuntimeError("FloLPIPS produced a non-finite score")
        values.append(float(score.item()))
    return float(sum(values) / len(values)), values
