import hashlib
import math
import os
import struct

import torch

from src.layers.cuda_inference import round_and_to_int8, restore_y_2x_with_cat_after, restore_y_2x, add_and_multiply
from src.models.entropy_models import EntropyCoder


MAGIC = b"GVM1"
HEADER = struct.Struct(">4sBBBBHHIIIII")
W1_HEADER = struct.Struct(">III")
MODE_TO_DELTA = {0: 1.0, 1: 2.0, 2: 4.0}


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class BitWriter:
    def __init__(self):
        self.bits = []

    def put(self, value, width):
        self.bits.extend((value >> i) & 1 for i in range(width - 1, -1, -1))

    def ue(self, value):
        code_num = int(value) + 1
        zeros = code_num.bit_length() - 1
        self.bits.extend([0] * zeros)
        self.put(code_num, zeros + 1)

    def bytes(self):
        out = bytearray((len(self.bits) + 7) // 8)
        for i, bit in enumerate(self.bits):
            out[i // 8] |= bit << (7 - i % 8)
        return bytes(out), len(self.bits)


class BitReader:
    def __init__(self, data, bit_count):
        self.data, self.bit_count, self.pos = data, bit_count, 0

    def get(self, width):
        if self.pos + width > self.bit_count:
            raise ValueError("mode-map bitstream exhausted")
        value = 0
        for _ in range(width):
            value = (value << 1) | ((self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1)
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while self.get(1) == 0:
            zeros += 1
        suffix = self.get(zeros) if zeros else 0
        return (1 << zeros) + suffix - 1


def encode_mode_map(modes):
    flat = [int(x) for x in modes]
    if not flat:
        return b"", 0, []
    runs = []
    start = 0
    while start < len(flat):
        mode = flat[start]
        if mode not in MODE_TO_DELTA:
            raise ValueError(f"invalid precision mode {mode}")
        end = start + 1
        while end < len(flat) and flat[end] == mode:
            end += 1
        runs.append((mode, end - start))
        start = end
    writer = BitWriter()
    for mode, length in runs:
        writer.put(mode, 2)
        writer.ue(length - 1)
    data, bits = writer.bytes()
    return data, bits, runs


def decode_mode_map(data, bit_count, count):
    reader, out, runs = BitReader(data, bit_count), [], []
    while len(out) < count:
        mode = reader.get(2)
        if mode not in MODE_TO_DELTA:
            raise ValueError(f"invalid decoded precision mode {mode}")
        length = reader.ue() + 1
        if len(out) + length > count:
            raise ValueError("mode-map run exceeds block count")
        out.extend([mode] * length)
        runs.append((mode, length))
    if reader.pos != bit_count:
        raise ValueError("unused mode-map bits")
    return out, runs


def block_grid(height, width, block_h, block_w):
    return (height + block_h - 1) // block_h, (width + block_w - 1) // block_w


def delta_tensor(modes, height, width, block_h, block_w, device, dtype):
    gh, gw = block_grid(height, width, block_h, block_w)
    if len(modes) != gh * gw:
        raise ValueError(f"mode count {len(modes)} != {gh * gw}")
    grid = torch.as_tensor(modes, device=device, dtype=torch.long).reshape(1, 1, gh, gw)
    lut = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype)
    return lut[grid].repeat_interleave(block_h, 2).repeat_interleave(block_w, 3)[:, :, :height, :width]


def mode_tensor(modes, height, width, block_h, block_w, device):
    gh, gw = block_grid(height, width, block_h, block_w)
    if len(modes) != gh * gw: raise ValueError("invalid mode count")
    grid = torch.as_tensor(modes, device=device, dtype=torch.uint8).reshape(1, 1, gh, gw)
    return grid.repeat_interleave(block_h, 2).repeat_interleave(block_w, 3)[:, :, :height, :width]


def all_one_modes(height, width, block_h, block_w):
    gh, gw = block_grid(height, width, block_h, block_w)
    return [0] * (gh * gw)


def prepare_frame(model, x, qp, modes, block_h=4, block_w=4):
    qf = model.q_scale_feature[qp:qp + 1]
    qe = model.q_scale_enc[qp:qp + 1]
    qd = model.q_scale_dec[qp:qp + 1]
    qr = model.q_scale_recon[qp:qp + 1]
    ref_feature = model.apply_feature_adaptor()
    ctx, ctx_t = model.feature_extractor(ref_feature, qf)
    y = model.enc(x, ctx, qe).detach()
    z = model.hyper_enc(model.pad_for_y(y))
    z_hat, z_write = round_and_to_int8(z)
    params = model.res_prior_param_decoder(z_hat, ctx_t)
    q_dec, scales, means = model.separate_prior_for_video_decoding(params)
    y_scaled = y * torch.reciprocal(q_dec)
    b, c, h, w = y_scaled.shape
    mask0, mask1 = model.get_mask_2x(b, c, h, w, y.dtype, y.device)
    r0, q0, yhat0, scale0 = model.process_with_mask(y_scaled, scales, means, mask0)
    cat = torch.cat((yhat0, params), dim=1)
    scales1, means1 = model.y_spatial_prior(cat).chunk(2, 1)
    r1 = (y_scaled - means1 * mask1) * mask1
    scale1 = scales1 * mask1
    r1w = model.single_part_for_writing_2x(r1)
    s1w = model.single_part_for_writing_2x(scale1)
    if modes is None:
        modes = all_one_modes(h, w, block_h, block_w)
    delta = delta_tensor(modes, h, w, block_h, block_w, y.device, y.dtype)
    q1w = torch.clamp(torch.round(r1w / delta), -128, 127)
    threshold = model.gaussian_encoder.force_zero_thres
    active = torch.ones_like(s1w, dtype=torch.bool) if threshold is None else s1w > threshold
    q1w = q1w * active
    r1hatw = q1w * delta
    yhat1 = restore_y_2x(r1hatw, means1, mask1)
    y_hat = add_and_multiply(yhat0, yhat1, q_dec)
    feature = model.dec(y_hat, ctx, qd)
    rgb = model.recon_generation_net(feature, qr)
    w0 = model.single_part_for_writing_2x(q0)
    s0 = model.single_part_for_writing_2x(scale0)
    return {
        "ctx": ctx.detach(), "ctx_t": ctx_t.detach(), "qd": qd.detach(), "qr": qr.detach(),
        "z": z_hat.detach(), "z_write": z_write.detach(), "w0": w0.detach(), "s0": s0.detach(),
        "w1": q1w.detach(), "w1_reconstructed": r1hatw.detach(), "w1_residual": r1w.detach(),
        "s1": s1w.detach(), "delta": delta.detach(), "modes": list(map(int, modes)),
        "feature": feature.detach(), "rgb": rgb.detach(), "latent": y_hat.detach(), "shape": tuple(q1w.shape),
        "_yhat0": yhat0.detach(), "_means1": means1.detach(), "_mask1": mask1.detach(),
        "_q_dec": q_dec.detach(),
    }


def requantize_frame(model, prepared, modes, block_h=4, block_w=4, reconstruct_rgb=True):
    """Requantize only the already-computed true w1 centered residual."""
    h, w = prepared["shape"][-2:]
    delta = delta_tensor(modes, h, w, block_h, block_w,
                         prepared["w1_residual"].device, prepared["w1_residual"].dtype)
    q1 = torch.clamp(torch.round(prepared["w1_residual"] / delta), -128, 127)
    threshold = model.gaussian_encoder.force_zero_thres
    active = torch.ones_like(prepared["s1"], dtype=torch.bool) if threshold is None else prepared["s1"] > threshold
    q1 = q1 * active
    rhat = q1 * delta
    yhat1 = restore_y_2x(rhat, prepared["_means1"], prepared["_mask1"])
    yhat = add_and_multiply(prepared["_yhat0"].clone(), yhat1, prepared["_q_dec"])
    out = dict(prepared)
    out.update(w1=q1.detach(), w1_reconstructed=rhat.detach(), delta=delta.detach(),
               modes=list(map(int, modes)), latent=yhat.detach())
    if reconstruct_rgb:
        feature = model.dec(yhat, prepared["ctx"], prepared["qd"])
        rgb = model.recon_generation_net(feature, prepared["qr"])
        out.update(feature=feature.detach(), rgb=rgb.detach())
    return out


def _encode_z(model, z_write, qp):
    model.entropy_coder.reset()
    model.bit_estimator_z.encode_z(z_write.to(torch.int8), qp)
    model.entropy_coder.flush()
    return model.entropy_coder.get_encoded_stream()


def _encode_y(model, symbols, scales, activity_scales=None):
    model.entropy_coder.reset()
    if activity_scales is None:
        model.gaussian_encoder.encode_y(symbols, scales)
    else:
        threshold = model.gaussian_encoder.force_zero_thres
        model.gaussian_encoder.force_zero_thres = None
        try:
            combined = model.gaussian_encoder.build_indexes_encoder(symbols, scales)
        finally:
            model.gaussian_encoder.force_zero_thres = threshold
        active = torch.ones_like(activity_scales, dtype=torch.bool).reshape(-1) if threshold is None else activity_scales.reshape(-1) > threshold
        model.entropy_coder.encode_y(combined[active], model.gaussian_encoder.cdf_group_index)
    model.entropy_coder.flush()
    return model.entropy_coder.get_encoded_stream()


def _coarse_cdf_group(model, mode):
    if mode == 0: return model.gaussian_encoder.cdf_group_index
    cache = getattr(model.gaussian_encoder, "_v13_coarse_cdf_groups", None)
    # Keep a strong reference. Comparing id() is unsafe after per-frame coder
    # renewal because CPython may reuse the released object's address.
    coder = model.entropy_coder
    if cache is None or cache.get("coder") is not coder:
        cache = {"coder": coder}; model.gaussian_encoder._v13_coarse_cdf_groups = cache
    if mode in cache: return cache[mode]
    delta = MODE_TO_DELTA[mode]
    effective_scales = model.gaussian_encoder.scale_table / delta
    center = torch.zeros_like(effective_scales) + 8
    normal = torch.distributions.Normal(torch.zeros_like(effective_scales), effective_scales)
    for i in range(8, 1, -1):
        probs = normal.cdf(torch.zeros_like(center) + i)
        center = torch.where(probs > torch.zeros_like(center) + .9999,
                             torch.zeros_like(center) + i, center)
    center = center.int(); lengths = 2 * center + 1; max_length = int(lengths.max())
    samples = torch.arange(max_length) - center[:, None]
    dist = torch.distributions.Normal(torch.zeros_like(samples, dtype=torch.float32), effective_scales[:, None])
    upper, lower = dist.cdf(samples.float() + .5), dist.cdf(samples.float() - .5)
    pmf, tail = upper - lower, 2 * lower[:, :1]
    cdf = EntropyCoder.pmf_to_cdf(pmf, tail, lengths, max_length)
    group = model.entropy_coder.add_cdf(cdf.numpy(), (lengths + 2).int().numpy(), (-center).int().numpy())
    cache[mode] = group
    return group


def _scale_indexes_no_skip(model, scales):
    threshold = model.gaussian_encoder.force_zero_thres
    model.gaussian_encoder.force_zero_thres = None
    try: indexes, _ = model.gaussian_encoder.build_indexes_decoder(scales)
    finally: model.gaussian_encoder.force_zero_thres = threshold
    return indexes.reshape(scales.shape)


def encode_w1_by_mode(model, prepared, block_h, block_w):
    symbols, scales = prepared["w1"], prepared["s1"]
    h, w = symbols.shape[-2:]
    spatial_modes = mode_tensor(prepared["modes"], h, w, block_h, block_w, symbols.device).expand_as(symbols)
    threshold = model.gaussian_encoder.force_zero_thres
    active = torch.ones_like(scales, dtype=torch.bool) if threshold is None else scales > threshold
    # Encode the three deterministic mode partitions into one RANS stream.
    # The native extension cannot flush a one-symbol standalone stream, while
    # sequential encode_y calls followed by one flush are its normal API.
    model.entropy_coder.reset()
    for mode in (0, 1, 2):
        select = (spatial_modes == mode) & active
        if not bool(select.any()): continue
        # q=round(r/delta) under N(0,scale) has the unit-bin law of
        # N(0,scale/delta).  Build indexes through the repository's original
        # 128-level CDF table, including its normal clamping/precision rules.
        indexes = _scale_indexes_no_skip(model, scales / MODE_TO_DELTA[mode])
        combined = (symbols.to(torch.int16) << 8) + indexes.to(torch.int16)
        if os.environ.get("V13_DEBUG_PROBE") == "1":
            vals = symbols[select]; idxs = indexes[select]
            print("w1_rans_mode", mode, "count", int(select.sum()), "q", int(vals.min()), int(vals.max()),
                  "index", int(idxs.min()), int(idxs.max()), flush=True)
        model.entropy_coder.encode_y(combined[select], model.gaussian_encoder.cdf_group_index)
    model.entropy_coder.flush()
    return model.entropy_coder.get_encoded_stream()


def decode_w1_by_mode(model, stream, scales, modes, block_h, block_w, dtype, device):
    h, w = scales.shape[-2:]
    spatial_modes = mode_tensor(modes, h, w, block_h, block_w, device).expand_as(scales)
    threshold = model.gaussian_encoder.force_zero_thres
    active = torch.ones_like(scales, dtype=torch.bool) if threshold is None else scales > threshold
    result = torch.zeros_like(scales)
    model.entropy_coder.set_stream(stream)
    for mode in (0, 1, 2):
        select = (spatial_modes == mode) & active
        count = int(select.sum())
        if count == 0:
            continue
        indexes = _scale_indexes_no_skip(model, scales / MODE_TO_DELTA[mode])
        selected_indexes = indexes[select]
        decoded = model.entropy_coder.decode_and_get_y(selected_indexes, model.gaussian_encoder.cdf_group_index, device, dtype)
        result.masked_scatter_(select, decoded)
    return result


def encode_original(model, prepared, qp):
    model.entropy_coder.set_use_two_entropy_coders(True)
    model.entropy_coder.reset()
    model.bit_estimator_z.encode_z(prepared["z_write"].to(torch.int8), qp)
    model.gaussian_encoder.encode_y(prepared["w0"], prepared["s0"])
    model.gaussian_encoder.encode_y(prepared["w1"], prepared["s1"])
    model.entropy_coder.flush()
    payload = model.entropy_coder.get_encoded_stream()
    # Prefix re-encodes give exact, telescoping attribution of the joint RANS payload.
    z_bytes = len(_encode_z(model, prepared["z_write"], qp))
    model.entropy_coder.reset()
    model.bit_estimator_z.encode_z(prepared["z_write"].to(torch.int8), qp)
    model.gaussian_encoder.encode_y(prepared["w0"], prepared["s0"])
    model.entropy_coder.flush()
    zw0_bytes = len(model.entropy_coder.get_encoded_stream())
    return payload, {"z_bytes": z_bytes, "w0_bytes": zw0_bytes - z_bytes,
                     "w1_bytes": len(payload) - zw0_bytes, "mode_map_bytes": 0,
                     "mixed_header_bytes": 0}


def encode_mixed(model, prepared, qp, block_h=4, block_w=4, global_mode=None):
    model.entropy_coder.set_use_two_entropy_coders(False)
    z_stream = _encode_z(model, prepared["z_write"], qp)
    w0_stream = _encode_y(model, prepared["w0"], prepared["s0"])
    w1_stream = encode_w1_by_mode(model, prepared, block_h, block_w)
    h, w = prepared["shape"][-2:]
    if global_mode is None:
        map_stream, map_bits, runs = encode_mode_map(prepared["modes"])
        kind, gm = 1, 255
    else:
        map_stream, map_bits, runs = b"", 0, [(int(global_mode), len(prepared["modes"]))]
        kind, gm = 2, int(global_mode)
    head = HEADER.pack(MAGIC, 1, kind, gm, 0, h, w, map_bits,
                       len(z_stream), len(w0_stream), len(w1_stream), len(map_stream))
    payload = head + map_stream + z_stream + w0_stream + w1_stream
    hist = {m: prepared["modes"].count(m) for m in MODE_TO_DELTA}
    return payload, {
        "z_bytes": len(z_stream), "w0_bytes": len(w0_stream), "w1_bytes": len(w1_stream),
        "mode_map_bytes": len(map_stream), "mixed_header_bytes": len(head),
        "raw_map_bits": 2 * len(prepared["modes"]), "coded_map_bits": map_bits,
        "mode_histogram": hist, "run_count": len(runs),
        "run_min": min(x[1] for x in runs), "run_max": max(x[1] for x in runs),
        "run_mean": sum(x[1] for x in runs) / len(runs),
    }


def parse_mixed(payload, block_h=4, block_w=4):
    if len(payload) < HEADER.size or payload[:4] != MAGIC:
        return None
    fields = HEADER.unpack(payload[:HEADER.size])
    _, version, kind, global_mode, flags, h, w, map_bits, zl, w0l, w1l, ml = fields
    if version != 1 or flags != 0 or len(payload) != HEADER.size + ml + zl + w0l + w1l:
        raise ValueError("invalid mixed payload header")
    pos = HEADER.size
    map_stream = payload[pos:pos + ml]; pos += ml
    gh, gw = block_grid(h, w, block_h, block_w); count = gh * gw
    if kind == 1:
        modes, runs = decode_mode_map(map_stream, map_bits, count)
    elif kind == 2 and global_mode in MODE_TO_DELTA and ml == 0 and map_bits == 0:
        modes, runs = [global_mode] * count, [(global_mode, count)]
    else:
        raise ValueError("invalid mixed payload kind")
    z_stream = payload[pos:pos + zl]; pos += zl
    w0_stream = payload[pos:pos + w0l]; pos += w0l
    w1_stream = payload[pos:pos + w1l]
    return {"kind": kind, "global_mode": global_mode, "h": h, "w": w, "modes": modes,
            "runs": runs, "map_bits": map_bits, "map_stream": map_stream,
            "z_stream": z_stream, "w0_stream": w0_stream, "w1_stream": w1_stream}


def decode_mixed(model, payload, sps, qp, block_h=4, block_w=4):
    parsed = parse_mixed(payload, block_h, block_w)
    if parsed is None:
        captured, calls = {}, [0]
        get_z, get_y, get_recon = model.bit_estimator_z.get_z, model.gaussian_encoder.get_y, model.get_recon_and_feature
        def capture_z(*args, **kwargs):
            value = get_z(*args, **kwargs); captured["z"] = value.detach(); return value
        def capture_y(*args, **kwargs):
            value = get_y(*args, **kwargs); captured["w0" if calls[0] == 0 else "w1"] = value.detach(); calls[0] += 1; return value
        def capture_recon(y_hat, *args, **kwargs):
            captured["latent"] = y_hat.detach(); return get_recon(y_hat, *args, **kwargs)
        model.bit_estimator_z.get_z = capture_z; model.gaussian_encoder.get_y = capture_y; model.get_recon_and_feature = capture_recon
        try:
            out = model.decompress(payload, sps, qp)["x_hat"]
        finally:
            model.bit_estimator_z.get_z = get_z; model.gaussian_encoder.get_y = get_y; model.get_recon_and_feature = get_recon
        captured.update(modes=None, feature=model.dpb[0].feature.detach(), mixed=False)
        return out, captured
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    qf = model.q_scale_feature[qp:qp + 1]
    qd = model.q_scale_dec[qp:qp + 1]
    qr = model.q_scale_recon[qp:qp + 1]
    model.entropy_coder.set_use_two_entropy_coders(False)
    z_size = model.get_downsampled_shape(sps["height"], sps["width"], 64)
    model.entropy_coder.set_stream(parsed["z_stream"])
    model.bit_estimator_z.decode_z(z_size, qp)
    feature_ref = model.apply_feature_adaptor()
    c1, ctx_t = model.feature_extractor.forward_part1(feature_ref, qf)
    z = model.bit_estimator_z.get_z(z_size, device, dtype)
    params = model.res_prior_param_decoder(z, ctx_t)
    q_dec, scales, means = model.separate_prior_for_video_decoding(params)
    b, c, h, w = means.shape
    if (h, w) != (parsed["h"], parsed["w"]):
        raise ValueError("mode-map dimensions disagree with latent")
    mask0, mask1 = model.get_mask_2x(b, c, h, w, dtype, device)
    s0 = model.single_part_for_writing_2x(scales * mask0)
    model.entropy_coder.set_stream(parsed["w0_stream"])
    w0 = model.gaussian_encoder.decode_and_get_y(s0, dtype, device)
    yhat0, cat = restore_y_2x_with_cat_after(w0, means, mask0, params)
    scales1, means1 = model.y_spatial_prior(cat).chunk(2, 1)
    s1 = model.single_part_for_writing_2x(scales1 * mask1)
    delta = delta_tensor(parsed["modes"], h, w, block_h, block_w, device, dtype)
    w1 = decode_w1_by_mode(model, parsed["w1_stream"], s1, parsed["modes"], block_h, block_w, dtype, device)
    yhat1 = restore_y_2x(w1 * delta, means1, mask1)
    y_hat = add_and_multiply(yhat0, yhat1, q_dec)
    ctx = model.feature_extractor.forward_part2(c1)
    rgb, feature = model.get_recon_and_feature(y_hat, ctx, qd, qr)
    model.add_ref_frame(feature, rgb)
    return rgb, {"modes": parsed["modes"], "z": z.detach(), "w0": w0.detach(),
                 "w1": w1.detach(), "latent": y_hat.detach(), "feature": feature.detach(), "mixed": True}


def prepare_probe_decode_cache(model, payload, sps, qp, block_h=4, block_w=4):
    """Build decoder-side probability/reconstruction context once for candidates sharing z/w0."""
    parsed = parse_mixed(payload, block_h, block_w)
    if parsed is None: raise ValueError("probe cache requires mixed payload")
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    qf = model.q_scale_feature[qp:qp + 1]; qd = model.q_scale_dec[qp:qp + 1]; qr = model.q_scale_recon[qp:qp + 1]
    model.entropy_coder.set_use_two_entropy_coders(False)
    z_size = model.get_downsampled_shape(sps["height"], sps["width"], 64)
    model.entropy_coder.set_stream(parsed["z_stream"]); model.bit_estimator_z.decode_z(z_size, qp)
    feature_ref = model.apply_feature_adaptor(); c1, ctx_t = model.feature_extractor.forward_part1(feature_ref, qf)
    z = model.bit_estimator_z.get_z(z_size, device, dtype); params = model.res_prior_param_decoder(z, ctx_t)
    q_dec, scales, means = model.separate_prior_for_video_decoding(params)
    b, c, h, w = means.shape; mask0, mask1 = model.get_mask_2x(b, c, h, w, dtype, device)
    s0 = model.single_part_for_writing_2x(scales * mask0)
    model.entropy_coder.set_stream(parsed["w0_stream"]); w0 = model.gaussian_encoder.decode_and_get_y(s0, dtype, device)
    yhat0, cat = restore_y_2x_with_cat_after(w0, means, mask0, params)
    scales1, means1 = model.y_spatial_prior(cat).chunk(2, 1); s1 = model.single_part_for_writing_2x(scales1 * mask1)
    ctx = model.feature_extractor.forward_part2(c1)
    return {"dtype": dtype, "device": device, "qp": qp, "z_size": z_size, "z": z.detach(), "w0": w0.detach(),
            "z_stream": parsed["z_stream"], "w0_stream": parsed["w0_stream"], "s0": s0.detach(),
            "s1": s1.detach(), "yhat0": yhat0.detach(), "means1": means1.detach(), "mask1": mask1.detach(),
            "q_dec": q_dec.detach(), "ctx": ctx.detach(), "qd": qd.detach(), "qr": qr.detach(), "h": h, "w": w}


def decode_probe_latent_cached(model, payload, cache, block_h=4, block_w=4):
    """Actually RANS-decode z/w0/w1, reusing only deterministic neural contexts shared by the probe set."""
    parsed = parse_mixed(payload, block_h, block_w)
    if parsed is None or parsed["z_stream"] != cache["z_stream"] or parsed["w0_stream"] != cache["w0_stream"]:
        raise ValueError("probe candidate does not share z/w0 streams")
    model.entropy_coder.set_stream(parsed["z_stream"]); model.bit_estimator_z.decode_z(cache["z_size"], cache["qp"])
    z = model.bit_estimator_z.get_z(cache["z_size"], cache["device"], cache["dtype"])
    model.entropy_coder.set_stream(parsed["w0_stream"])
    w0 = model.gaussian_encoder.decode_and_get_y(cache["s0"], cache["dtype"], cache["device"])
    if not torch.equal(z, cache["z"]) or not torch.equal(w0, cache["w0"]): raise ValueError("cached probe z/w0 mismatch")
    w1 = decode_w1_by_mode(model, parsed["w1_stream"], cache["s1"], parsed["modes"], block_h, block_w,
                           cache["dtype"], cache["device"])
    delta = delta_tensor(parsed["modes"], cache["h"], cache["w"], block_h, block_w, cache["device"], cache["dtype"])
    yhat1 = restore_y_2x(w1 * delta, cache["means1"], cache["mask1"])
    latent = add_and_multiply(cache["yhat0"].clone(), yhat1, cache["q_dec"])
    return latent, z, w0, w1, parsed


def gaussian_estimated_bits(model, symbols, scales, activity_scales=None):
    threshold = model.gaussian_encoder.force_zero_thres
    source = scales if activity_scales is None else activity_scales
    active = torch.ones_like(source, dtype=torch.bool) if threshold is None else source > threshold
    q = symbols.float()[active]
    s = scales.float()[active]
    if q.numel() == 0:
        return 0.0
    index_source = source.float()[active].clamp(model.gaussian_encoder.scale_min, model.gaussian_encoder.scale_max)
    idx = ((torch.log(index_source) - model.gaussian_encoder.log_scale_min) * model.gaussian_encoder.log_step_recip).long().clamp(0, 127)
    delta = (source.float()[active] / s).clamp_min(1.0)
    table_scale = model.gaussian_encoder.scale_table.to(s.device)[idx] / delta
    normal = torch.distributions.Normal(torch.zeros_like(table_scale), table_scale)
    prob = (normal.cdf(q + .5) - normal.cdf(q - .5)).clamp_min(1e-12)
    return float((-torch.log2(prob)).sum())


def map_stats(modes):
    stream, bits, runs = encode_mode_map(modes)
    hist = {m: modes.count(m) for m in MODE_TO_DELTA}
    return {"raw_map_bits": 2 * len(modes), "coded_map_bits": bits, "map_bytes": len(stream),
            "number_1x_blocks": hist[0], "number_2x_blocks": hist[1], "number_4x_blocks": hist[2],
            "run_count": len(runs), "run_min": min(r[1] for r in runs),
            "run_max": max(r[1] for r in runs), "run_mean": sum(r[1] for r in runs) / len(runs)}
