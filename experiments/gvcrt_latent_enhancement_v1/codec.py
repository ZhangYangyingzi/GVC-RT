"""Original codec calls, strict weights, and read-only extraction of decoded ell.

Decode API accepts only serialized base stream and shared checkpoints/config.
Enhancement output is computed AFTER DMC.decompress updates its base-only DPB.
"""
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.models.image_model_gvcrt import DMCI
from src.models.video_model_gvcrt import DMC
from src.utils.stream_helper import (SPSHelper, NalType, write_sps, write_ip,
                                     read_header, read_sps_remaining, read_ip_remaining)


def digest_state(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_hash(t):
    if t is None:
        return None
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dpb_hash(p):
    return [{"poc": r.poc, "feature": tensor_hash(r.feature), "frame": tensor_hash(r.frame)} for r in p.dpb]


def strict_load(model, path, tag):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if tag == "I":
        sd = ck.get("student", ck.get("ema_shadow", ck.get("state_dict", ck)))
    else:
        sd = ck.get("student_ema", ck.get("student", ck.get("state_dict", ck)))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "keys": len(sd), "missing": [], "unexpected": [], "strict": True}


def load_models(cfg, dtype=torch.float16):
    i, p = DMCI(), DMC()
    evidence = {}
    for tag, model, key in [("I", i, "checkpoint_i"), ("P", p, "checkpoint_p")]:
        evidence[tag] = strict_load(model, ROOT / cfg[key], tag)
        model.requires_grad_(False).eval().cuda()
        model.update(cfg["force_zero_thres"])
        model.to(dtype=dtype)
        model.set_use_two_entropy_coders(True)
    return i, p, evidence


def source_frame(video_id, frame, device="cuda"):
    rgb = np.array(Image.open(HERE / "data" / video_id / f"{frame:04d}.png").convert("RGB"), copy=True)
    return torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255


@torch.no_grad()
def encode_base(i, p, cfg, entry, qp, path):
    output = io.BytesIO()
    helper = SPSHelper()
    p.clear_dpb()
    p.set_curr_poc(0)
    rows = []
    last_qp = 0
    index_map = [0,1,0,2,0,2,0,2]
    ph, pw = cfg["padded_hw"]
    vh, vw = entry["valid_hw"]
    for frame in range(entry["frames"]):
        x = source_frame(entry["video_id"], frame).half()
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = F.pad(x, (0,pw-vw,0,ph-vh), mode="replicate") * 2 - 1
        intra = frame == 0 or (cfg["intra_period"] > 0 and frame % cfg["intra_period"] == 0)
        ada = int(not intra and cfg["reset_interval"] > 0 and frame % cfg["reset_interval"] == 1)
        if intra:
            current_qp = qp
            encoded = i.compress(x, current_qp)
            p.clear_dpb()
            p.add_ref_frame(None, encoded["x_hat"])
        else:
            if ada:
                p.prepare_feature_adaptor_i(last_qp)
            current_qp = p.shift_qp(qp, index_map[frame % 8])
            encoded = p.compress(x, current_qp)
            last_qp = current_qp
        sps = {"sps_id": -1, "height": ph, "width": pw, "ec_part": 1, "use_ada_i": ada}
        sid, new = helper.get_sps_id(sps)
        sps["sps_id"] = sid
        size = write_sps(output, sps) if new else 0
        size += write_ip(output, intra, sid, current_qp, encoded["bit_stream"])
        torch.cuda.synchronize()
        rows.append({"frame": frame, "type": "I" if intra else "P", "qp": current_qp,
                     "use_ada_i": ada, "base_bits": 8*size, "encode_seconds": time.perf_counter()-start})
    with Path(path).open("xb") as f:
        f.write(output.getvalue())
    assert sum(r["base_bits"] for r in rows) == Path(path).stat().st_size*8
    return rows


@torch.no_grad()
def decode_base(i, p, path, bypass_check=False):
    # Source RGB is deliberately not an argument and never read here.
    buf = io.BytesIO(Path(path).read_bytes())
    helper = SPSHelper()
    p.clear_dpb()
    p.set_curr_poc(0)
    outputs = []
    capture = {}

    def capture_ell(module, args):
        capture["ell"] = args[0].detach().clone()
        capture["q"] = args[1].detach().clone()

    def capture_feature(module, args):
        capture["feature_shape"] = list(args[0].shape)
        capture["feature_range"] = [args[0].min().item(), args[0].max().item()]

    def capture_y(module, args):
        capture["y_shape"] = list(args[0].shape)

    handles = [p.recon_generation_net.decoder.register_forward_pre_hook(capture_ell),
               p.recon_generation_net.register_forward_pre_hook(capture_feature),
               p.dec.register_forward_pre_hook(capture_y)]
    try:
        while buf.tell() < len(buf.getbuffer()):
            start_pos = buf.tell()
            header = read_header(buf)
            while header["nal_type"] == NalType.NAL_SPS:
                sps = read_sps_remaining(buf, header["sps_id"])
                helper.add_sps_by_id(sps)
                header = read_header(buf)
            sps = helper.get_sps_by_id(header["sps_id"])
            qp, stream = read_ip_remaining(buf)
            capture.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            intra = header["nal_type"] == NalType.NAL_I
            if intra:
                decoded = i.decompress(stream, sps, qp)
                p.clear_dpb()
                p.add_ref_frame(None, decoded["x_hat"])
            else:
                if sps["use_ada_i"]:
                    p.reset_ref_feature()
                decoded = p.decompress(stream, sps, qp)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter()-start
            x = decoded["x_hat"]
            row = {"frame": len(outputs), "qp": qp, "type": "I" if intra else "P",
                   "base_bits": 8*(buf.tell()-start_pos), "sps": sps.copy(), "x_base": x.cpu(),
                   "decode_seconds": decode_seconds, "dpb": dpb_hash(p)}
            if not intra:
                ell, q = capture["ell"], capture["q"]
                row.update({"ell": ell.cpu(), "q_recon": q.cpu(), "feature_shape": capture["feature_shape"],
                            "feature_range": capture["feature_range"], "y_shape": capture["y_shape"]})
                if bypass_check:
                    from enhancement import display_reconstruction
                    before_ell, before_dpb = tensor_hash(ell), dpb_hash(p)
                    repeated = display_reconstruction(p.recon_generation_net.decoder, ell, q, correction=None)
                    row["bypass_exact"] = torch.equal(repeated, x)
                    row["bypass_dpb_exact"] = before_dpb == dpb_hash(p)
                    row["ell_immutable"] = before_ell == tensor_hash(ell)
                    assert row["bypass_exact"] and row["bypass_dpb_exact"] and row["ell_immutable"]
            outputs.append(row)
    finally:
        for h in handles:
            h.remove()
    return outputs


def jsonable_rows(rows):
    result = []
    for row in rows:
        out = {k:v for k,v in row.items() if not torch.is_tensor(v)}
        for key in ["ell", "q_recon", "x_base"]:
            if key in row:
                t = row[key]
                out[key+"_shape"] = list(t.shape)
                out[key+"_range"] = [t.min().item(), t.max().item()]
                out[key+"_sha256"] = tensor_hash(t)
        result.append(out)
    return result
