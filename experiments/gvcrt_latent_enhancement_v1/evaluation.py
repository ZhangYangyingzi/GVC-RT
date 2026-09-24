"""Valid-region metrics, lossless review videos, and a self-contained enhancement container."""
import hashlib
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.utils.metrics import calc_msssim_rgb, calc_psnr

CONTAINER = struct.Struct("<4sHHI32s")


def write_container(path, base_path, valid_hw, streams):
    blob = CONTAINER.pack(b"GEC1", *valid_hw, len(streams), hashlib.sha256(Path(base_path).read_bytes()).digest())
    for stream in streams:
        blob += struct.pack("<I",len(stream)) + stream
    with Path(path).open("xb") as f:
        f.write(blob)
    return len(blob)*8


def read_container(path, base_path):
    blob = Path(path).read_bytes()
    if len(blob) < CONTAINER.size:
        raise ValueError("Truncated enhancement container")
    magic,h,w,n,digest = CONTAINER.unpack_from(blob)
    if magic != b"GEC1" or digest != hashlib.sha256(Path(base_path).read_bytes()).digest():
        raise ValueError("Base stream identity mismatch")
    pos,streams = CONTAINER.size,[]
    for _ in range(n):
        size, = struct.unpack_from("<I",blob,pos)
        pos += 4
        if pos+size > len(blob):
            raise ValueError("Truncated frame")
        streams.append(blob[pos:pos+size])
        pos += size
    if pos != len(blob):
        raise ValueError("Trailing container bytes")
    return (h,w),streams


def rgb01(x, hw=(1080,1920)):
    return (x[:,:,:hw[0],:hw[1]].float().clamp(-1,1)+1)/2


def quality(source, decoded, perceptual):
    a,b = source.detach().float().cpu().numpy()[0], decoded.detach().float().cpu().numpy()[0]
    with torch.no_grad():
        lp = perceptual(source.cuda().float(),decoded.cuda().float(),normalize=True).item()
    return {"psnr": calc_psnr(a,b,data_range=1), "ms_ssim": float(calc_msssim_rgb(a,b,data_range=1)), "lpips": lp}


def write_video(path, frames, fps):
    h,w = frames[0].shape[-2:]
    cmd = ["ffmpeg","-v","error","-n","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}",
           "-r",str(fps),"-i","-","-an","-c:v","ffv1","-level","3",str(path)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for frame in frames:
            image = (frame[0].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy()
            proc.stdin.write(image.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg video write failed")


def temporal(source, outputs):
    # Diagnostic unwarped error-difference, NOT a claim of perceptual temporal stability.
    errors = [(y-x).float().cpu() for x,y in zip(source,outputs)]
    return {"error_temporal_mse": float(np.mean([(b-a).square().mean().item() for a,b in zip(errors,errors[1:])])),
            "mean_rgb_bias": torch.stack([e.mean((0,2,3)) for e in errors]).mean(0).tolist(),
            "frame_error_bias_std_rgb": torch.stack([e.mean((0,2,3)) for e in errors]).std(0).tolist()}
