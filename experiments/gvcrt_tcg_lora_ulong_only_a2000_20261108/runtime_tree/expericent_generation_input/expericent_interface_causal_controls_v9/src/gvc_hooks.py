import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from common import REPO, sha256, torch_load

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.layers.cuda_inference import replicate_pad
from src.models.image_model_gvcrt import DMCI
from src.models.video_model_gvcrt import DMC
from src.utils.common import set_torch_env
from src.utils.stream_helper import (NalType, SPSHelper, read_header, read_ip_remaining,
                                     read_sps_remaining, write_ip, write_sps)


INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]


def load_models(device, force_zero_thres=0.12):
    set_torch_env()
    i_path, p_path = REPO / "checkpoints/GVC-RT_I.pt", REPO / "checkpoints/GVC-RT_P.pt"
    i_model = DMCI(encoder_ckpt_path=str(i_path)).to(device).eval().half()
    p_model = DMC()
    checkpoint = torch_load(p_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("student_ema", checkpoint.get("student", checkpoint.get("state_dict", checkpoint)))
    p_model.load_state_dict(state, strict=True)
    p_model = p_model.to(device).eval().half()
    for model in (i_model, p_model):
        model.requires_grad_(False)
        model.update(force_zero_thres)
        model.set_use_two_entropy_coders(True)
    return i_model, p_model


def iter_frames(video, count, device):
    command = ["ffmpeg", "-v", "error", "-i", video["path"], "-map", "0:v:0",
               "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    frame_size = 1920 * 1080 * 3
    try:
        for index in range(count):
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                raise RuntimeError(f"Incomplete frame {index} from {video['path']}")
            array = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float16) / 255
            yield replicate_pad(tensor, 8, 0) * 2 - 1
    finally:
        process.stdout.close()
        if process.wait():
            raise RuntimeError(f"ffmpeg failed: {video['path']}")


def encode(frames, i_model, p_model, output_path, requested_qp):
    stream, helper, bits, actual_qps = io.BytesIO(), SPSHelper(), [], []
    p_model.set_curr_poc(0)
    p_model.clear_dpb()
    with torch.no_grad():
        for index, frame in enumerate(frames):
            if index == 0:
                qp, is_i, encoded = requested_qp, True, i_model.compress(frame, requested_qp)
                p_model.clear_dpb()
                p_model.add_ref_frame(None, encoded["x_hat"])
            else:
                qp, is_i = p_model.shift_qp(requested_qp, INDEX_MAP[index % 8]), False
                encoded = p_model.compress(frame, qp)
            sps = {"sps_id": -1, "height": 1088, "width": 1920,
                   "ec_part": 1, "use_ada_i": 0}
            sps_id, new = helper.get_sps_id(sps)
            sps["sps_id"] = sps_id
            size = write_sps(stream, sps) if new else 0
            size += write_ip(stream, is_i, sps_id, qp, encoded["bit_stream"])
            bits.append(size * 8)
            actual_qps.append(qp)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(stream.getvalue())
    return bits, actual_qps


def _capture(model, bit_stream, sps, qp, keep_feature):
    captured = {}
    hooks = [
        model.recon_generation_net.register_forward_pre_hook(
            lambda _, args: captured.update(bridge_input=args[0].detach())),
        model.recon_generation_net.decoder.register_forward_pre_hook(
            lambda _, args: captured.update(codeword=args[0].detach(), quant=args[1].detach())),
        model.recon_generation_net.decoder.register_forward_hook(
            lambda _, args, output: captured.update(generator_output=output.detach()))]
    try:
        with torch.no_grad():
            baseline = model.decompress(bit_stream, sps, qp)["x_hat"].detach()
    finally:
        for hook in hooks:
            hook.remove()
    record = {"baseline": baseline, "codeword": captured["codeword"],
              "quant": captured["quant"], "generator_output": captured["generator_output"]}
    if keep_feature:
        record["bridge_feature"] = captured["bridge_input"]
    return record


def decode_records(stream_bytes, count, i_model, p_model, keep_feature=True, record_indexes=None):
    stream, helper, records = io.BytesIO(stream_bytes), SPSHelper(), []
    p_model.set_curr_poc(0)
    p_model.clear_dpb()
    for index in range(count):
        header = read_header(stream)
        while header["nal_type"] == NalType.NAL_SPS:
            helper.add_sps_by_id(read_sps_remaining(stream, header["sps_id"]))
            header = read_header(stream)
        sps = helper.get_sps_by_id(header["sps_id"])
        qp, bit_stream = read_ip_remaining(stream)
        is_i = header["nal_type"] == NalType.NAL_I
        if is_i:
            model, kind = i_model, "I"
            captured = _capture(model, bit_stream, sps, qp, False)
            p_model.clear_dpb()
            p_model.add_ref_frame(None, captured["baseline"])
        else:
            model, kind = p_model, "P"
            capture_feature = keep_feature and (record_indexes is None or index in record_indexes)
            captured = _capture(model, bit_stream, sps, qp, capture_feature)
        with torch.no_grad():
            rerun = model.recon_generation_net.decoder(captured["codeword"], captured["quant"])
        if record_indexes is None or index in record_indexes:
            records.append({"frame": index, "frame_type": kind, "actual_qp": qp,
                            "codeword": captured["codeword"].cpu(),
                            "quant": captured["quant"].cpu(),
                            "bridge_feature": captured.get("bridge_feature", None).cpu()
                                              if captured.get("bridge_feature") is not None else None,
                            "rerun_max_abs": float((rerun - captured["generator_output"]).abs().max())})
        del captured, rerun
    return records


def decoder_for(kind, models):
    return models[0].recon_generation_net.decoder if kind == "I" else models[1].recon_generation_net.decoder
