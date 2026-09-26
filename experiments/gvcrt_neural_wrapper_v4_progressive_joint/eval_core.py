import hashlib
import io
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from core import (ROOT, REPO, codec_input, compression_hash, load_models,
                  ms_ssim_rgb, quality_models, unit)
from wrapper_model import NeuralWrapper

V9 = REPO / "expericent_generation_input/expericent_interface_causal_controls_v9/src"
if str(V9) not in sys.path:
    sys.path.insert(0, str(V9))
from gvc_hooks import INDEX_MAP
from src.utils.stream_helper import (NalType, SPSHelper, read_header,
                                     read_ip_remaining, read_sps_remaining,
                                     write_ip, write_sps)

SPS = {"sps_id": 0, "height": 1088, "width": 1920, "ec_part": 1, "use_ada_i": 0}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_joint(path, device):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    wrapper = NeuralWrapper().to(device).float().eval()
    wrapper.load_state_dict(payload["wrapper"], strict=True)
    wrapper.requires_grad_(False)
    return payload, wrapper


def apply_joint(model, payload):
    model.recon_generation_net.mlp.load_state_dict(payload["bridge"], strict=True)
    model.recon_generation_net.decoder.load_state_dict(payload["generator"], strict=True)
    model.recon_generation_net.eval().half().requires_grad_(False)


def matched_frames(tag, count):
    base = REPO / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames" / tag
    return [torch.from_numpy(np.asarray(Image.open(base / f"im{index}.png").convert("RGB"),
                                        dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
            for index in range(1, count + 1)]


def video_frames(path, count):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(count),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, frame_size = [], 1920 * 1080 * 3
    try:
        for _ in range(count):
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                raise RuntimeError("incomplete validation source")
            array = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).copy()
            output.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255)
    finally:
        process.stdout.close()
        if process.wait():
            raise RuntimeError("ffmpeg validation decode failed")
    return output


def save_png(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = torch.clamp(value * 255, 0, 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path, compress_level=1)


def frame_metrics(output, target, quality):
    difference = output - target
    sse = float(difference.square().sum())
    mse = float(difference.square().mean())
    ux, uy = output.mean(), target.mean()
    vx, vy = ((output - ux) ** 2).mean(), ((target - uy) ** 2).mean()
    covariance = ((output - ux) * (target - uy)).mean()
    ssim = float(((2 * ux * uy + .01 ** 2) * (2 * covariance + .03 ** 2)) /
                 ((ux ** 2 + uy ** 2 + .01 ** 2) * (vx + vy + .03 ** 2)))
    return {
        "pixel_SSE": sse,
        "MSE": mse,
        "PSNR": -10 * math.log10(max(mse, 1e-15)),
        "SSIM": ssim,
        "MS_SSIM": ms_ssim_rgb(output, target),
        "LPIPS": float(quality[0](output, target, normalize=True)),
        "DISTS": float(quality[1](output, target)),
    }


def high_frequency_energy(value):
    dx = value[:, :, :, 1:] - value[:, :, :, :-1]
    dy = value[:, :, 1:, :] - value[:, :, :-1, :]
    return float((dx.square().mean() + dy.square().mean()) / 2)


def build_models(device, payload=None):
    i_model, p_model = load_models(device)
    if payload is not None:
        apply_joint(p_model, payload)
    return i_model, p_model


def run_stream(frames_cpu, qp, joint, device, fps, bitstream_path,
               save_root=None, quality=None):
    payload, wrapper = (None, None) if joint is None else joint
    quality = quality or quality_models(device)
    i_encoder, p_encoder = build_models(device, payload)
    i_decoder, p_decoder = build_models(device, payload)
    core_before = compression_hash(i_encoder, p_encoder)
    p_encoder.clear_dpb(); p_decoder.clear_dpb()
    p_encoder.set_curr_poc(0); p_decoder.set_curr_poc(0)
    stream, rows, decoded_hashes, state_errors = io.BytesIO(), [], [], []
    write_sps(stream, SPS)
    with torch.inference_mode():
        for frame_index, cpu in enumerate(frames_cpu):
            target = cpu.to(device)
            proxy = target if wrapper is None else wrapper(target)
            actual_qp = qp if frame_index == 0 else p_encoder.shift_qp(qp, INDEX_MAP[frame_index % 8])
            is_i = frame_index == 0
            if is_i:
                encoded = i_encoder.compress(codec_input(proxy), actual_qp)
                p_encoder.clear_dpb(); p_encoder.add_ref_frame(None, encoded["x_hat"])
                decoded = i_decoder.decompress(encoded["bit_stream"], SPS, actual_qp)
                p_decoder.clear_dpb(); p_decoder.add_ref_frame(None, decoded["x_hat"])
            else:
                encoded = p_encoder.compress(codec_input(proxy), actual_qp)
                decoded = p_decoder.decompress(encoded["bit_stream"], SPS, actual_qp)
                state_errors.append(float((p_encoder.dpb[0].feature -
                                           p_decoder.dpb[0].feature).abs().max()))
            write_ip(stream, is_i, 0, actual_qp, encoded["bit_stream"])
            output = unit(decoded["x_hat"], 1080, 1920)
            metrics = frame_metrics(output, target, quality)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError("non-finite evaluation metric")
            rows.append({"frame": frame_index, "actual_qp": actual_qp,
                         "payload_bytes": len(encoded["bit_stream"]), **metrics})
            if save_root is not None and wrapper is not None:
                proxy_metrics = frame_metrics(proxy, target, quality)
                rows[-1].update({
                    "proxy_PSNR": proxy_metrics["PSNR"],
                    "proxy_LPIPS": proxy_metrics["LPIPS"],
                    "proxy_DISTS": proxy_metrics["DISTS"],
                    "proxy_mean_absolute_difference": float((proxy - target).abs().mean()),
                    "source_high_frequency_energy": high_frequency_energy(target),
                    "proxy_high_frequency_energy": high_frequency_energy(proxy),
                    "high_frequency_energy_difference": high_frequency_energy(proxy) - high_frequency_energy(target),
                })
            decoded_hashes.append(tensor_sha(decoded["x_hat"]))
            if save_root is not None:
                save_png(proxy, Path(save_root) / "proxy" / f"frame_{frame_index:06d}.png")
                save_png(output, Path(save_root) / "recon" / f"frame_{frame_index:06d}.png")
    data = stream.getvalue()
    bitstream_path = Path(bitstream_path)
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    bitstream_path.write_bytes(data)
    pixels = len(rows) * 3 * 1080 * 1920
    total_sse = sum(row["pixel_SSE"] for row in rows)
    aggregate_mse = total_sse / pixels
    summary = {
        "bytes": len(data),
        "bits": len(data) * 8,
        "bpp": len(data) * 8 / (len(rows) * 1080 * 1920),
        "kbps": len(data) * 8 * fps / len(rows) / 1000,
        "total_pixel_SSE": total_sse,
        "aggregate_MSE": aggregate_mse,
        "PSNR": -10 * math.log10(max(aggregate_mse, 1e-15)),
        "SSIM": sum(row["SSIM"] for row in rows) / len(rows),
        "MS_SSIM": sum(row["MS_SSIM"] for row in rows) / len(rows),
        "LPIPS": sum(row["LPIPS"] for row in rows) / len(rows),
        "DISTS": sum(row["DISTS"] for row in rows) / len(rows),
        "bitstream_path": str(bitstream_path),
        "bitstream_sha256": sha256_bytes(data),
        "state_sync_pass": all(value == 0 for value in state_errors),
        "compression_hash_before": core_before,
        "compression_hash_after": compression_hash(i_encoder, p_encoder),
        "reconstruction_sha256": sha256_bytes("".join(decoded_hashes).encode()),
    }

    audit_i, audit_p = build_models(device, payload)
    audit_p.clear_dpb(); audit_p.set_curr_poc(0)
    buffer, helper, audit_hashes = io.BytesIO(data), SPSHelper(), []
    with torch.inference_mode():
        while buffer.tell() < len(data):
            header = read_header(buffer)
            while header["nal_type"] == NalType.NAL_SPS:
                helper.add_sps_by_id(read_sps_remaining(buffer, header["sps_id"]))
                header = read_header(buffer)
            sps = helper.get_sps_by_id(header["sps_id"])
            actual_qp, bit_stream = read_ip_remaining(buffer)
            if header["nal_type"] == NalType.NAL_I:
                decoded = audit_i.decompress(bit_stream, sps, actual_qp)
                audit_p.clear_dpb(); audit_p.add_ref_frame(None, decoded["x_hat"])
            else:
                decoded = audit_p.decompress(bit_stream, sps, actual_qp)
            audit_hashes.append(tensor_sha(decoded["x_hat"]))
    summary["bytes_consumed"] = buffer.tell()
    summary["decode_status"] = "PASS" if (audit_hashes == decoded_hashes and
                                                buffer.tell() == len(data)) else "FAIL"
    summary["independent_decode_pass"] = summary["decode_status"] == "PASS"
    summary["finite"] = True
    if (not summary["independent_decode_pass"] or not summary["state_sync_pass"] or
            summary["compression_hash_before"] != summary["compression_hash_after"]):
        raise RuntimeError("real RANS stream audit failed")
    return summary, rows
