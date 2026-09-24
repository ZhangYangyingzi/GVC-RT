#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ORIG = ROOT.parent / "gvcrt_v13_generator_aware_mixed_precision"
V124 = ROOT.parent / "gvcrt_v12_4_sequence_discrete_beam"
for path in (REPO, ORIG, ROOT):
    sys.path.insert(0, str(path))

import v13_codec as codec
import run_v13 as v13
from src.layers.cuda_inference import replicate_pad
from src.utils.stream_helper import read_header, read_ip_remaining, read_sps_remaining, NalType

CFG = json.loads((ROOT / "config.json").read_text())
MAN = json.loads((ROOT / "manifest.json").read_text())
VIDEO = next(v for v in MAN["videos"] if v["dataset"] == CFG["dataset"] and int(v["video_id"]) == CFG["video_id"])
SPS = v13.SPS


def write_csv(name, rows, fields=None):
    rows = list(rows)
    path = ROOT / name
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open("w", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    y = x.detach().float()
    return {"shape": list(y.shape), "min": float(y.min()), "max": float(y.max()),
            "mean": float(y.mean()), "std": float(y.std()), "l1": float(y.abs().sum()),
            "l2": float(torch.linalg.vector_norm(y)), "nonzero": int(torch.count_nonzero(y)),
            "sha256": tensor_hash(x)}


def diff(a, b):
    d = (a.detach().float() - b.detach().float()).abs()
    return float(d.max()), float(d.mean()), int(torch.count_nonzero(d))


def read_frames8(video, device):
    n, size, out = CFG["frames"], 1920 * 1080 * 3, []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video["source_path"], "-map", "0:v:0",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    try:
        for i in range(n):
            raw = p.stdout.read(size)
            if len(raw) != size: raise RuntimeError(f"short frame {i}")
            a = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).copy()
            x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float16) / 255
            out.append(replicate_pad(x, 8, 0) * 2 - 1)
    finally:
        p.stdout.close(); p.wait()
    return out


def parse_any(path):
    out, sps = [], {}
    path = Path(path)
    with path.open("rb") as f:
        while f.tell() < path.stat().st_size:
            h = read_header(f)
            if h["nal_type"] == NalType.NAL_SPS:
                sps[h["sps_id"]] = read_sps_remaining(f, h["sps_id"]); continue
            qp, payload = read_ip_remaining(f)
            out.append((h["nal_type"] == NalType.NAL_I, qp, payload))
    return out


def b2_codeword(model, feature):
    net = model.recon_generation_net
    x = F.pixel_unshuffle(feature, 2)
    x = net.mlp[0](x); x = net.mlp[1](x); x = net.mlp[2](x)
    x = x * torch.sigmoid(x)
    return net.mlp[3](x)


def block_slices(prepared, block_index):
    h, w = prepared["shape"][-2:]
    gh, gw = codec.block_grid(h, w, CFG["block_h"], CFG["block_w"])
    by, bx = divmod(block_index, gw)
    ys = slice(by * CFG["block_h"], min((by + 1) * CFG["block_h"], h))
    xs = slice(bx * CFG["block_w"], min((bx + 1) * CFG["block_w"], w))
    return by, bx, ys, xs


def probability_tensor(model, q, scale, delta):
    source = scale.float()
    active = source > model.gaussian_encoder.force_zero_thres
    idx = ((torch.log(source.clamp(model.gaussian_encoder.scale_min, model.gaussian_encoder.scale_max))
            - model.gaussian_encoder.log_scale_min) * model.gaussian_encoder.log_step_recip).long().clamp(0, 127)
    table_scale = model.gaussian_encoder.scale_table.to(source.device)[idx] / float(delta)
    normal = torch.distributions.Normal(torch.zeros_like(table_scale), table_scale)
    p = normal.cdf(q.float() + .5) - normal.cdf(q.float() - .5)
    return p.clamp_min(1e-12), active


def baseline_8(frames, device):
    iframe, enc, dec = v13.load_triplet(device)
    before = v13.model_hash(iframe, enc, dec)
    stream, rgb0, ip = v13.init_sequence(iframe, enc, dec, frames[0], CFG["qp"])
    rows = [{"frame": 0, "actual_qp": CFG["qp"], "frame_bytes": len(v13.packet(ip, CFG["qp"], True)),
             "z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0, "MSE": v13.basic_metrics(rgb0, frames[0])[0],
             "PSNR": v13.basic_metrics(rgb0, frames[0])[1], "w1_hash": "", "latent_hash": "",
             "rgb_hash": tensor_hash(rgb0), "dpb_state_hash": v13.state_hash(v13.snap_cpu(dec))}]
    torch.save({"rgb": rgb0.cpu(), "dpb_state_hash": rows[0]["dpb_state_hash"]}, ROOT / "tensors/baseline_frame00.pt")
    cumulative_est = 0.0
    for fi in range(1, CFG["frames"]):
        aq = enc.shift_qp(CFG["qp"], v13.INDEX_MAP[fi % 8])
        prep = codec.prepare_frame(enc, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
        payload, comp = codec.encode_original(enc, prep, aq)
        out, cap = codec.decode_mixed(dec, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
        if not torch.equal(cap["w1"], prep["w1"]) or not torch.equal(out, prep["rgb"]):
            raise RuntimeError(f"baseline identity mismatch frame {fi}")
        enc.add_ref_frame(prep["feature"], None)
        pkt = v13.packet(payload, aq); stream += pkt
        mse, psnr, _ = v13.basic_metrics(out, frames[fi])
        est = codec.gaussian_estimated_bits(enc, prep["w1"], prep["s1"], prep["s1"]); cumulative_est += est
        row = {"frame": fi, "actual_qp": aq, "frame_bytes": len(pkt), **comp, "MSE": mse, "PSNR": psnr,
               "estimated_w1_bits": est, "w1_hash": tensor_hash(cap["w1"]), "latent_hash": tensor_hash(cap["latent"]),
               "rgb_hash": tensor_hash(out), "dpb_state_hash": v13.state_hash(v13.snap_cpu(dec))}
        rows.append(row)
        torch.save({"z": cap["z"].cpu(), "w0": cap["w0"].cpu(), "w1": cap["w1"].cpu(),
                    "latent": cap["latent"].cpu(), "rgb": out.cpu(), "feature": cap["feature"].cpu(),
                    "dpb_state_hash": row["dpb_state_hash"]}, ROOT / f"tensors/baseline_frame{fi:02d}.pt")
    path = ROOT / "bitstreams/baseline_8frames.bin"; path.write_bytes(stream)
    after = v13.model_hash(iframe, enc, dec)
    ref_packets = parse_any(V124 / "bitstreams/fresh_ulong_00_qp3_BASELINE.bin")[:CFG["frames"]]
    ref_stream = io.BytesIO()
    from src.utils.stream_helper import write_sps
    write_sps(ref_stream, SPS)
    for is_i, qp, payload in ref_packets: ref_stream.write(v13.packet(payload, qp, is_i))
    identity = stream == ref_stream.getvalue() and before == after
    write_csv("baseline_frame_debug.csv", rows)
    (ROOT / "baseline_summary.json").write_text(json.dumps({"total_bytes": len(stream), "sha256": sha256(path),
        "identity_pass": identity, "model_hash_before": before, "model_hash_after": after,
        "cumulative_estimated_w1_bits": cumulative_est}, indent=2) + "\n")
    return identity, rows, stream


def frame1_setup(frames, device):
    iframe, enc, dec = v13.load_triplet(device)
    initial, _, _ = v13.init_sequence(iframe, enc, dec, frames[0], CFG["qp"])
    aq = enc.shift_qp(CFG["qp"], v13.INDEX_MAP[1])
    state = v13.snap_cpu(dec)
    prep = codec.prepare_frame(enc, frames[1], aq, None, CFG["block_h"], CFG["block_w"])
    return initial, aq, iframe, enc, dec, state, prep


def pick_block(enc, prep):
    h, w = prep["shape"][-2:]; gh, gw = codec.block_grid(h, w, CFG["block_h"], CFG["block_w"])
    scored = []
    p, active = probability_tensor(enc, prep["w1"], prep["s1"], 1)
    for bi in range(gh * gw):
        _, _, ys, xs = block_slices(prep, bi)
        cost = float((-torch.log2(p[..., ys, xs][active[..., ys, xs]])).sum())
        nonzero = int(torch.count_nonzero(prep["w1_residual"][..., ys, xs]))
        if nonzero: scored.append((cost, bi))
    scored.sort(reverse=True)
    for order, (_, bi) in enumerate(scored[:64], 1):
        modes = [0] * (gh * gw); modes[bi] = 1
        c2 = codec.requantize_frame(enc, prep, modes, CFG["block_h"], CFG["block_w"])
        modes[bi] = 2
        c4 = codec.requantize_frame(enc, prep, modes, CFG["block_h"], CFG["block_w"])
        _, _, ys, xs = block_slices(prep, bi)
        q1 = prep["w1"][..., ys, xs]
        if not torch.equal(q1, c2["w1"][..., ys, xs]) and not torch.equal(q1, c4["w1"][..., ys, xs]):
            return bi, order
    return scored[0][1], min(64, len(scored))


def independent_decode_case(path, expected, device):
    packets = parse_any(path)
    iframe, _, dec = v13.load_triplet(device)
    dec.clear_dpb(); dec.set_curr_poc(0)
    i_is, i_qp, i_payload = packets[0]
    rgb0 = iframe.decompress(i_payload, SPS, i_qp)["x_hat"]
    dec.add_ref_frame(None, rgb0)
    is_i, qp, payload = packets[1]
    out, cap = codec.decode_mixed(dec, payload, SPS, qp, CFG["block_h"], CFG["block_w"])
    parsed = codec.parse_mixed(payload, CFG["block_h"], CFG["block_w"])
    if parsed is None:
        decoded_delta = torch.ones_like(expected["delta"])
    else:
        decoded_delta = codec.delta_tensor(parsed["modes"], expected["delta"].shape[-2], expected["delta"].shape[-1],
                                           CFG["block_h"], CFG["block_w"], device, expected["delta"].dtype)
    decoded_rhat = cap["w1"] * decoded_delta
    checks = {"mode_map_match": (parsed is None and expected["mode"] == 0) or parsed["modes"] == expected["modes"],
              "delta_match": torch.equal(expected["delta"].cpu(), decoded_delta.cpu()),
              "reconstructed_residual_match": torch.equal(decoded_rhat.cpu(), expected["reconstructed_residual"].cpu()),
              "w1_match": torch.equal(cap["w1"].cpu(), expected["w1"].cpu()),
              "latent_match": torch.equal(cap["latent"].cpu(), expected["latent"].cpu()),
              "feature_match": torch.equal(cap["feature"].cpu(), expected["feature"].cpu()),
              "rgb_match": torch.equal(out.cpu(), expected["rgb"].cpu())}
    return checks, cap, out


def single_block(frames, device):
    initial, aq, iframe, enc, dec, dec_state, prep = frame1_setup(frames, device)
    bi, searched = pick_block(enc, prep)
    by, bx, ys, xs = block_slices(prep, bi)
    cases, saved, decoded = {}, {}, {}
    base_feature = base_code = base_rgb = None
    for mode in (0, 1, 2):
        modes = [0] * len(prep["modes"]); modes[bi] = mode
        cand = prep if mode == 0 else codec.requantize_frame(enc, prep, modes, CFG["block_h"], CFG["block_w"])
        if mode == 0: payload, comp = codec.encode_original(enc, cand, aq)
        else: payload, comp = codec.encode_mixed(enc, cand, aq, CFG["block_h"], CFG["block_w"])
        v13.restore_device(dec, dec_state, device)
        out, cap = codec.decode_mixed(dec, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
        code = b2_codeword(dec, cap["feature"])
        path = ROOT / f"bitstreams/single_block_{codec.MODE_TO_DELTA[mode]:.0f}x.bin"
        path.write_bytes(initial + v13.packet(payload, aq))
        decoded_delta = codec.delta_tensor(modes, prep["shape"][-2], prep["shape"][-1], CFG["block_h"], CFG["block_w"], device, prep["w1"].dtype)
        expected = {"mode": mode, "modes": modes, "delta": cand["delta"], "decoded_delta": decoded_delta,
                    "reconstructed_residual": cand["w1_reconstructed"],
                    "w1": cand["w1"], "latent": cand["latent"], "feature": cand["feature"], "rgb": cand["rgb"]}
        checks, cap2, out2 = independent_decode_case(path, expected, device)
        p, active = probability_tensor(enc, cand["w1"][..., ys, xs], cand["s1"][..., ys, xs], codec.MODE_TO_DELTA[mode])
        pv = p[active[..., ys, xs] if active.shape != p.shape else active]
        r = prep["w1_residual"][..., ys, xs]
        q = cand["w1"][..., ys, xs]
        rh = cand["w1_reconstructed"][..., ys, xs]
        mse, psnr, _ = v13.basic_metrics(out, frames[1])
        cases[mode] = {"mode": mode, "delta": codec.MODE_TO_DELTA[mode], "frame": 1, "block_index": bi,
                       "block_y": by, "block_x": bx, "blocks_searched": searched,
                       "q_hash": tensor_hash(q), "rhat_hash": tensor_hash(rh), "decoded_w1_hash": tensor_hash(cap["w1"]),
                       "actual_payload_bytes": len(payload), "actual_packet_bytes": len(v13.packet(payload, aq)),
                       "actual_w1_RANS_bytes": comp["w1_bytes"], "mode_map_bytes": comp["mode_map_bytes"],
                       "entropy_probability_min": float(pv.min()), "entropy_probability_mean": float(pv.mean()),
                       "entropy_probability_max": float(pv.max()), "estimated_block_bits": float((-torch.log2(pv)).sum()),
                       "estimated_w1_bits": codec.gaussian_estimated_bits(enc, cand["w1"], cand["s1"] / cand["delta"], cand["s1"]),
                       "RGB_MSE": mse, "PSNR": psnr, "bitstream_path": str(path), "bitstream_sha256": sha256(path)}
        saved[mode] = {"residual": r.cpu(), "r_div_delta": (r / codec.MODE_TO_DELTA[mode]).cpu(), "q": q.cpu(),
                       "r_hat": rh.cpu(), "probability": p.cpu(), "active": active.cpu(), "decoded_w1": cap["w1"].cpu(),
                       "latent": cap["latent"].cpu(), "feature": cap["feature"].cpu(), "b2_codeword": code.cpu(), "rgb": out.cpu()}
        decoded[mode] = (checks, cap2, out2)
        if mode == 0: base_feature, base_code, base_rgb = cap["feature"], code, out
    rows, audit_rows, tensor_json = [], [], {"selected_block": bi, "block_y": by, "block_x": bx, "blocks_searched": searched, "cases": {}}
    q1, r1 = saved[0]["q"], saved[0]["r_hat"]
    for mode in (0, 1, 2):
        qmax, qmean, qnz = diff(saved[mode]["q"], q1)
        rmax, rmean, rnz = diff(saved[mode]["r_hat"], r1)
        wmax, wmean, wnz = diff(saved[mode]["decoded_w1"], saved[0]["decoded_w1"])
        lmax, lmean, lnz = diff(saved[mode]["latent"], saved[0]["latent"])
        fmax, fmean, fnz = diff(saved[mode]["feature"], saved[0]["feature"])
        cmax, cmean, cnz = diff(saved[mode]["b2_codeword"], saved[0]["b2_codeword"])
        xmax, xmean, xnz = diff(saved[mode]["rgb"], saved[0]["rgb"])
        row = dict(cases[mode], q_equal_1x=torch.equal(saved[mode]["q"], q1), q_max_abs_diff=qmax, q_mean_abs_diff=qmean,
                   reconstructed_residual_max_abs_diff=rmax, reconstructed_residual_mean_abs_diff=rmean,
                   decoded_w1_max_abs_diff=wmax, decoded_w1_mean_abs_diff=wmean,
                   decoded_compression_feature_max_abs_diff=fmax, decoded_compression_feature_mean_abs_diff=fmean,
                   B2_input_max_abs_diff=fmax, B2_input_mean_abs_diff=fmean,
                   B2_output_max_abs_diff=cmax, B2_output_mean_abs_diff=cmean,
                   final_RGB_max_abs_diff=xmax, final_RGB_mean_abs_diff=xmean,
                   PSNR_difference=cases[mode]["PSNR"] - cases[0]["PSNR"])
        rows.append(row)
        checks = decoded[mode][0]
        audit_rows.append({"mode": mode, "delta": codec.MODE_TO_DELTA[mode], "frame": 1, "block_index": bi,
                           **checks, "decode_pass": all(checks.values()), "bitstream_path": cases[mode]["bitstream_path"],
                           "bitstream_sha256": cases[mode]["bitstream_sha256"]})
        tensor_json["cases"][str(mode)] = {"residual": stats(saved[mode]["residual"]), "r_div_delta": stats(saved[mode]["r_div_delta"]),
            "q": stats(saved[mode]["q"]), "reconstructed_residual": stats(saved[mode]["r_hat"]),
            "probability": stats(saved[mode]["probability"]), "decoded_w1": stats(saved[mode]["decoded_w1"]),
            "latent": stats(saved[mode]["latent"]), "compression_feature_B2_input": stats(saved[mode]["feature"]),
            "B2_output": stats(saved[mode]["b2_codeword"]), "RGB": stats(saved[mode]["rgb"])}
    torch.save(saved, ROOT / "tensors/single_block_cases.pt")
    write_csv("single_block_debug.csv", rows); write_csv("single_block_decode_audit.csv", audit_rows)
    (ROOT / "single_block_tensor_stats.json").write_text(json.dumps(tensor_json, indent=2) + "\n")
    return rows, audit_rows


def forced_fractions(frames, baseline_rows, device):
    rows, frame_rows = [], []
    for mode in CFG["forced_modes"]:
        for fraction in CFG["forced_fractions"]:
            iframe, enc, dec = v13.load_triplet(device)
            stream, rgb0, ip = v13.init_sequence(iframe, enc, dec, frames[0], CFG["qp"])
            mses = [v13.basic_metrics(rgb0, frames[0])[0]]; psnrs = [v13.basic_metrics(rgb0, frames[0])[1]]
            totals = {"z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0, "mode_map_bytes": 0, "estimated_w1_bits": 0.0}
            wh = hashlib.sha256(); symbols_changed = 0; count0 = count2 = count4 = 0
            for fi in range(1, CFG["frames"]):
                aq = enc.shift_qp(CFG["qp"], v13.INDEX_MAP[fi % 8])
                prep = codec.prepare_frame(enc, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
                n = len(prep["modes"]); coarse = int(math.ceil(n * fraction)); modes = [mode] * coarse + [0] * (n - coarse)
                cand = codec.requantize_frame(enc, prep, modes, CFG["block_h"], CFG["block_w"])
                payload, comp = codec.encode_mixed(enc, cand, aq, CFG["block_h"], CFG["block_w"])
                out, cap = codec.decode_mixed(dec, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
                if not torch.equal(cap["w1"], cand["w1"]) or not torch.equal(out, cand["rgb"]): raise RuntimeError("forced decode mismatch")
                enc.add_ref_frame(cand["feature"], None); pkt = v13.packet(payload, aq); stream += pkt
                mse, psnr, _ = v13.basic_metrics(out, frames[fi]); mses.append(mse); psnrs.append(psnr)
                est = codec.gaussian_estimated_bits(enc, cand["w1"], cand["s1"] / cand["delta"], cand["s1"])
                for k in ("z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"): totals[k] += comp[k]
                totals["estimated_w1_bits"] += est; wh.update(cap["w1"].cpu().contiguous().numpy().tobytes())
                symbols_changed += int(torch.count_nonzero(cand["w1"] != prep["w1"]))
                count0 += modes.count(0); count2 += modes.count(1); count4 += modes.count(2)
                frame_rows.append({"mode": mode, "delta": codec.MODE_TO_DELTA[mode], "fraction": fraction, "frame": fi,
                    "number_1x_blocks": modes.count(0), "number_2x_blocks": modes.count(1), "number_4x_blocks": modes.count(2),
                    "actual_frame_bytes": len(pkt), "z_bytes": comp["z_bytes"], "w0_bytes": comp["w0_bytes"],
                    "w1_bytes": comp["w1_bytes"], "mode_map_bytes": comp["mode_map_bytes"], "estimated_w1_bits": est,
                    "RGB_MSE": mse, "PSNR": psnr, "decoded_w1_hash": tensor_hash(cap["w1"]),
                    "DPB_state_hash": v13.state_hash(v13.snap_cpu(dec))})
            path = ROOT / f"bitstreams/forced_{int(codec.MODE_TO_DELTA[mode])}x_{int(fraction*100):03d}pct.bin"; path.write_bytes(stream)
            rows.append({"mode": mode, "delta": codec.MODE_TO_DELTA[mode], "fraction": fraction,
                "number_1x_blocks": count0, "number_2x_blocks": count2, "number_4x_blocks": count4,
                **totals, "I_bytes": len(ip), "total_bytes": len(stream), "symbols_changed_vs_natural": symbols_changed,
                "RGB_MSE": sum(mses)/len(mses), "PSNR": sum(psnrs)/len(psnrs), "decoded_w1_tensor_hash": wh.hexdigest(),
                "DPB_state_hash": v13.state_hash(v13.snap_cpu(dec)), "bitstream_path": str(path), "bitstream_sha256": sha256(path)})
    write_csv("forced_fraction_debug.csv", rows); write_csv("forced_fraction_frame_debug.csv", frame_rows)
    return rows


def filter_audit():
    baseline_est = {}
    baseline_w1 = {}
    with (ORIG / "frame_candidate_maps.csv").open(newline="") as f:
        maps = list(csv.DictReader(f))
    for r in maps:
        if r["candidate_map"] == "all_1x":
            key = (r["dataset"], r["video_id"], r["qp"], r["frame"])
            baseline_est[key] = float(r["estimated_w1_entropy_bits"])
            baseline_w1[key] = int(r["actual_w1_RANS_bytes"])
    groups = {}
    with (ORIG / "block_probe.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], r["video_id"], r["qp"]); g = groups.setdefault(k, {"total":0,"m1":0,"m2":0,"est":0,"actual":0,"est_zero":0,"filtered":0})
            g["total"] += 1; g["m1" if r["mode"] == "1" else "m2"] += 1
            saving = baseline_est[(r["dataset"],r["video_id"],r["qp"],r["frame"])] - float(r["estimated_w1_entropy_bits"])
            delta = int(r["actual_delta_bytes"])
            if saving > 0: g["est"] += 1
            if int(r["actual_w1_RANS_bytes"]) < baseline_w1[(r["dataset"],r["video_id"],r["qp"],r["frame"])]: g["actual"] += 1
            if saving > 0 and delta == 0: g["est_zero"] += 1
            if delta >= 0: g["filtered"] += 1
    map_counts = {}
    for r in maps:
        if r["candidate_map"] not in ("mild","medium","aggressive"): continue
        k=(r["dataset"],r["video_id"],r["qp"]); m=map_counts.setdefault(k,[0,0])
        m[0] += int(r["number_2x_blocks"]); m[1] += int(r["number_4x_blocks"])
    rows=[]
    for k,g in sorted(groups.items()):
        m=map_counts.get(k,[0,0]); rows.append({"dataset":k[0],"video_id":k[1],"qp":k[2],"total_block_probes":g["total"],
          "one_to_two_candidates":g["m1"],"one_to_four_candidates":g["m2"],"estimated_bits_decrease_candidates":g["est"],
          "actual_RANS_bytes_decrease_candidates":g["actual"],"estimated_saving_zero_actual_byte_saving":g["est_zero"],
          "filtered_by_actual_saved_bits_le_zero":g["filtered"],"entered_maps_2x_blocks":m[0],"entered_maps_4x_blocks":m[1]})
    pooled={k:sum(int(r[k]) for r in rows) for k in rows[0] if k not in ("dataset","video_id","qp")}
    rows.append({"dataset":"pooled","video_id":"","qp":"",**pooled})
    write_csv("candidate_filter_audit.csv",rows)
    logic = "Old V13 filtering facts\n" \
      "experiments/gvcrt_v13_generator_aware_mixed_precision/run_v13.py:286-287 computes saved_bits=max(0,-delta_bytes*8).\n" \
      "experiments/gvcrt_v13_generator_aware_mixed_precision/run_v13.py:299 appends to ranked only when saved_bits > 0.\n" \
      "experiments/gvcrt_v13_generator_aware_mixed_precision/run_v13.py:337-340 constructs maps only from ranked.\n" \
      "old_filter_requires_positive_actual_byte_saving=true\n"
    (ROOT/"candidate_filter_logic.txt").write_text(logic)
    return rows[-1]


def build_new_maps(frames, device):
    old_probe = [r for r in csv.DictReader((ORIG/"block_probe.csv").open()) if r["dataset"]=="fresh_ulong" and r["video_id"]=="0" and r["qp"]=="3" and int(r["frame"])<8]
    old_maps = [r for r in csv.DictReader((ORIG/"frame_candidate_maps.csv").open()) if r["dataset"]=="fresh_ulong" and r["video_id"]=="0" and r["qp"]=="3" and int(r["frame"])<8 and r["candidate_map"]=="all_1x"]
    base_est={int(r["frame"]):float(r["estimated_w1_entropy_bits"]) for r in old_maps}
    by_frame={i:[] for i in range(1,8)}
    for r in old_probe:
        fi=int(r["frame"]); saving=base_est[fi]-float(r["estimated_w1_entropy_bits"])
        if saving>0:
            dmse=float(r["delta_MSE"]); by_frame[fi].append((dmse/saving,int(r["block_index"]),int(r["mode"]),saving,dmse))
    iframe,enc,dec=v13.load_triplet(device); v13.init_sequence(iframe,enc,dec,frames[0],CFG["qp"])
    proposals={}; rows=[]
    for fi in range(1,8):
        aq=enc.shift_qp(CFG["qp"],v13.INDEX_MAP[fi%8]); state=v13.snap_cpu(dec)
        prep=codec.prepare_frame(enc,frames[fi],aq,None,CFG["block_h"],CFG["block_w"])
        candidates=sorted(by_frame[fi]); defs=[("all_1x",0.0),("mild",.025),("medium",.05),("aggressive",.10)]
        proposals[fi]={}
        for name,target in defs:
            modes=[0]*len(prep["modes"]); selected=set(); nominal=0.0
            for ratio,bi,mode,saving,dmse in candidates:
                if target==0 or nominal>=target*base_est[fi]: break
                if bi in selected: continue
                modes[bi]=mode; selected.add(bi); nominal+=saving
            cand=prep if name=="all_1x" else codec.requantize_frame(enc,prep,modes,CFG["block_h"],CFG["block_w"])
            if name=="all_1x": payload,comp=codec.encode_original(enc,cand,aq)
            else: payload,comp=codec.encode_mixed(enc,cand,aq,CFG["block_h"],CFG["block_w"])
            v13.restore_device(dec,state,device); out,cap=codec.decode_mixed(dec,payload,SPS,aq,CFG["block_h"],CFG["block_w"])
            mse,psnr,_=v13.basic_metrics(out,frames[fi]); st=codec.map_stats(modes)
            rows.append({"frame":fi,"candidate_map":name,"proposal_target":target,"number_1x_blocks":st["number_1x_blocks"],
              "number_2x_blocks":st["number_2x_blocks"],"number_4x_blocks":st["number_4x_blocks"],
              "sum_estimated_saved_bits":nominal,"estimated_w1_bits":codec.gaussian_estimated_bits(enc,cand["w1"],cand["s1"]/cand["delta"],cand["s1"]),
              "actual_w1_bytes":comp["w1_bytes"],"actual_mode_map_bytes":comp["mode_map_bytes"],
              "actual_total_frame_bytes":len(v13.packet(payload,aq)),"RGB_MSE":mse,"PSNR":psnr,
              "mode_map_json":json.dumps(modes,separators=(",",":"))})
            proposals[fi][name]=modes
        # advance immutable natural baseline
        bp,bcomp=codec.encode_original(enc,prep,aq); v13.restore_device(dec,state,device); codec.decode_mixed(dec,bp,SPS,aq,CFG["block_h"],CFG["block_w"])
        enc.add_ref_frame(prep["feature"],None)
    write_csv("new_frame_candidate_maps.csv",rows)
    return proposals,rows


def select_beam(states,width):
    unique={}
    for s in states:
        key=(v13.state_hash(s["dec_state"]),s["cum_bytes"],struct.pack(">d",s["cum_mse"]),s["mode_history_hash"])
        if key not in unique or s["cum_mse"]<unique[key]["cum_mse"]: unique[key]=s
    vals=list(unique.values()); pareto=[]
    for i,a in enumerate(vals):
        if not any(i!=j and b["cum_bytes"]<=a["cum_bytes"] and b["cum_mse"]<=a["cum_mse"] and
                   (b["cum_bytes"]<a["cum_bytes"] or b["cum_mse"]<a["cum_mse"]) for j,b in enumerate(vals)): pareto.append(a)
    chosen=[]
    for ordering in (sorted(pareto,key=lambda s:(s["cum_mse"],s["cum_bytes"])),sorted(pareto,key=lambda s:(s["cum_bytes"],s["cum_mse"])),
                     sorted(vals,key=lambda s:(s["cum_mse"],s["cum_bytes"])),sorted(vals,key=lambda s:(s["cum_bytes"],s["cum_mse"]))):
        for s in ordering:
            if len(chosen)>=width: break
            if s not in chosen: chosen.append(s)
    return chosen[:width]


def debug_beam(frames, proposals, baseline_rows, baseline_stream, device):
    iframe,enc,dec=v13.load_triplet(device); initial,rgb0,_=v13.init_sequence(iframe,enc,dec,frames[0],CFG["qp"])
    mse0=v13.basic_metrics(rgb0,frames[0])[0]
    root={"tid":"root","enc_state":v13.snap_cpu(enc),"dec_state":v13.snap_cpu(dec),"stream":initial,"cum_bytes":len(initial),
          "cum_mse":mse0,"cum_est":0.0,"n2":0,"n4":0,"history":[],"mode_history_hash":hashlib.sha256(b"").hexdigest(),"steps":[]}
    root["is_baseline_path"] = True
    baseline_state=root; beam=[]; states_rows=[]; serial=0
    baseline_cum=0
    for fi in range(1,8):
        expanded=[]; aq=enc.shift_qp(CFG["qp"],v13.INDEX_MAP[fi%8])
        for parent in [baseline_state] + beam:
            v13.restore_device(enc,parent["enc_state"],device); v13.restore_device(dec,parent["dec_state"],device)
            prep=codec.prepare_frame(enc,frames[fi],aq,None,CFG["block_h"],CFG["block_w"]); ds=v13.snap_cpu(dec)
            for name,modes in proposals[fi].items():
                cand=prep if name=="all_1x" else codec.requantize_frame(enc,prep,modes,CFG["block_h"],CFG["block_w"])
                if name=="all_1x": payload,comp=codec.encode_original(enc,cand,aq)
                else: payload,comp=codec.encode_mixed(enc,cand,aq,CFG["block_h"],CFG["block_w"])
                v13.restore_device(dec,ds,device); out,cap=codec.decode_mixed(dec,payload,SPS,aq,CFG["block_h"],CFG["block_w"])
                v13.restore_device(enc,parent["enc_state"],device); enc.add_ref_frame(cand["feature"],None)
                pkt=v13.packet(payload,aq); mse=v13.basic_metrics(out,frames[fi])[0]
                est=codec.gaussian_estimated_bits(enc,cand["w1"],cand["s1"]/cand["delta"],cand["s1"])
                history=parent["history"]+[name]; mh=hashlib.sha256(json.dumps(history).encode()).hexdigest(); serial+=1
                expanded.append({"tid":f"T{serial}","parent":parent["tid"],"enc_state":v13.snap_cpu(enc),"dec_state":v13.snap_cpu(dec),
                  "stream":parent["stream"]+pkt,"cum_bytes":parent["cum_bytes"]+len(pkt),"cum_mse":parent["cum_mse"]+mse,
                  "cum_est":parent["cum_est"]+est,"n2":parent["n2"]+modes.count(1),"n4":parent["n4"]+modes.count(2),
                  "history":history,"mode_history_hash":mh,"current_w1_hash":tensor_hash(cap["w1"]),
                  "is_baseline_path":parent["is_baseline_path"] and name=="all_1x",
                  "steps":parent["steps"]+[{"frame":fi,"map":name,"bytes":len(pkt),"mse":mse}]})
        baseline_candidates=[s for s in expanded if s["is_baseline_path"]]
        if len(baseline_candidates)!=1: raise RuntimeError(f"baseline trajectory multiplicity at frame {fi}")
        baseline_state=baseline_candidates[0]
        beam=select_beam([s for s in expanded if not s["is_baseline_path"]],CFG["beam_width"])
        baseline_cum += int(baseline_rows[fi]["frame_bytes"])
        for rank,s in enumerate(beam):
            states_rows.append({"trajectory_id":s["tid"],"parent_trajectory_id":s["parent"],"frame":fi,"beam_rank":rank,
              "cumulative_estimated_bits":s["cum_est"],"cumulative_actual_bytes":s["cum_bytes"],"cumulative_RGB_MSE":s["cum_mse"],
              "number_2x_blocks_used":s["n2"],"number_4x_blocks_used":s["n4"],"bitstream_hash":hashlib.sha256(s["stream"]).hexdigest(),
              "decoded_w1_hash":s["current_w1_hash"],"DPB_state_hash":v13.state_hash(s["dec_state"]),"mode_history_hash":s["mode_history_hash"],
              "is_baseline_path":False})
        states_rows.append({"trajectory_id":"BASELINE_PATH","parent_trajectory_id":"BASELINE_PATH","frame":fi,"beam_rank":-1,
          "cumulative_estimated_bits":baseline_state["cum_est"],"cumulative_actual_bytes":baseline_state["cum_bytes"],
          "cumulative_RGB_MSE":baseline_state["cum_mse"],"number_2x_blocks_used":0,"number_4x_blocks_used":0,
          "bitstream_hash":hashlib.sha256(baseline_state["stream"]).hexdigest(),"decoded_w1_hash":baseline_state["current_w1_hash"],
          "DPB_state_hash":v13.state_hash(baseline_state["dec_state"]),"mode_history_hash":baseline_state["mode_history_hash"],
          "is_baseline_path":True})
    trajectories=[]
    for s in beam:
        path=ROOT/f"bitstreams/debug_beam_{s['tid']}.bin"; path.write_bytes(s["stream"])
        trajectories.append({"trajectory_id":s["tid"],"frames":8,"cumulative_estimated_bits":s["cum_est"],"total_actual_bytes":s["cum_bytes"],
          "cumulative_RGB_MSE":s["cum_mse"],"number_2x_blocks_used":s["n2"],"number_4x_blocks_used":s["n4"],
          "bitstream_hash":sha256(path),"decoded_w1_hash":s["current_w1_hash"],"DPB_state_hash":v13.state_hash(s["dec_state"]),
          "mode_history":json.dumps(s["history"]),"bitstream_path":str(path),"is_baseline_path":False})
    bpath=ROOT/"bitstreams/debug_beam_BASELINE_PATH.bin"; bpath.write_bytes(baseline_stream)
    trajectories.append({"trajectory_id":"BASELINE_PATH","frames":8,"cumulative_estimated_bits":"","total_actual_bytes":len(baseline_stream),
      "cumulative_RGB_MSE":sum(float(r["MSE"]) for r in baseline_rows),"number_2x_blocks_used":0,"number_4x_blocks_used":0,
      "bitstream_hash":sha256(bpath),"decoded_w1_hash":baseline_rows[-1]["w1_hash"],"DPB_state_hash":baseline_rows[-1]["dpb_state_hash"],
      "mode_history":json.dumps(["all_1x"]*7),"bitstream_path":str(bpath),"is_baseline_path":True})
    write_csv("debug_beam_states.csv",states_rows); write_csv("debug_trajectories.csv",trajectories)
    return trajectories


def implementation_trace():
    text = """V13-debug implementation trace (code facts only)\n\nTrue w1 variable\n- experiments/gvcrt_v13_debug_mixed_precision/v13_codec.py:145-168 prepare_frame computes r1=(y_scaled-means1*mask1)*mask1, packs r1w with single_part_for_writing_2x, quantizes q1w=round(r1w/delta), and reconstructs r1hatw=q1w*delta.\n- Baseline delta0 is 1.0. q is the packed centered int8-range residual index. r_hat is delta*q.\n\nProbability and coder\n- experiments/gvcrt_v13_debug_mixed_precision/v13_codec.py:256-280 encode_w1_by_mode selects active symbols by mode and builds indexes using predicted scale/delta.\n- src/models/entropy_models.py:260-295 constructs the original Gaussian CDF table with unit-bin CDF differences and 16-bit integer CDF conversion.\n- experiments/gvcrt_v13_debug_mixed_precision/v13_codec.py:322-345 encode_mixed writes z, w0, w1 RANS streams and the mode-map syntax.\n\nDecoder reconstruction\n- experiments/gvcrt_v13_debug_mixed_precision/v13_codec.py:283-299 decode_w1_by_mode parses each deterministic mode partition with scale/delta indexes.\n- experiments/gvcrt_v13_debug_mixed_precision/v13_codec.py:390-421 decode_mixed reconstructs yhat1 from decoded w1*delta, then decodes the compression feature, B2 codeword and RGB.\n\nGenerator path\n- src/models/video_model_gvcrt.py:300-303 get_recon_and_feature maps compression latent to feature and calls recon_generation_net.\n- src/models/video_model_gvcrt.py:70-77 PretrainedReconWrapper maps feature through pixel_unshuffle and mlp (B2) to an 18-channel codeword, then through the frozen decoder.\n"""
    (ROOT/"implementation_trace.txt").write_text(text)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,default=CFG["gpu"]); args=ap.parse_args()
    device=torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device); torch.manual_seed(CFG["seed"])
    implementation_trace(); frames=read_frames8(VIDEO,device)
    baseline_pass,baseline_rows,baseline_stream=baseline_8(frames,device)
    single_rows,audit_rows=single_block(frames,device)
    forced=forced_fractions(frames,baseline_rows,device)
    pooled=filter_audit(); proposals,new_maps=build_new_maps(frames,device)
    trajectories=debug_beam(frames,proposals,baseline_rows,baseline_stream,device)
    row2=next(r for r in single_rows if r["mode"]==1); row4=next(r for r in single_rows if r["mode"]==2)
    a2=next(r for r in audit_rows if r["mode"]==1); a4=next(r for r in audit_rows if r["mode"]==2)
    f2=next(r for r in forced if r["mode"]==1 and r["fraction"]==1.0); f4=next(r for r in forced if r["mode"]==2 and r["fraction"]==1.0)
    base_bytes=len(baseline_stream); base_mse=sum(float(r["MSE"]) for r in baseline_rows)/8
    checks={"baseline_identity_pass":baseline_pass,
      "single_block_2x_changes_q":not row2["q_equal_1x"],"single_block_4x_changes_q":not row4["q_equal_1x"],
      "single_block_2x_changes_decoded_w1":row2["decoded_w1_max_abs_diff"]>0,"single_block_4x_changes_decoded_w1":row4["decoded_w1_max_abs_diff"]>0,
      "single_block_2x_changes_rgb":row2["final_RGB_max_abs_diff"]>0,"single_block_4x_changes_rgb":row4["final_RGB_max_abs_diff"]>0,
      "independent_decode_2x_pass":a2["decode_pass"],"independent_decode_4x_pass":a4["decode_pass"],
      "forced_100pct_2x_changes_total_bytes":f2["total_bytes"]!=base_bytes,"forced_100pct_4x_changes_total_bytes":f4["total_bytes"]!=base_bytes,
      "forced_100pct_2x_changes_rgb":abs(f2["RGB_MSE"]-base_mse)>0,"forced_100pct_4x_changes_rgb":abs(f4["RGB_MSE"]-base_mse)>0,
      "old_filter_requires_positive_actual_byte_saving":True,
      "candidates_with_estimated_saving_but_zero_byte_saving":int(pooled["estimated_saving_zero_actual_byte_saving"]),
      "new_maps_contain_real_coarse_blocks":all((r["number_2x_blocks"]+r["number_4x_blocks"]>0) for r in new_maps if r["candidate_map"]!="all_1x"),
      "debug_beam_unique_bitstream_count":len({r["bitstream_hash"] for r in trajectories}),
      "debug_beam_unique_dpb_state_count":len({r["DPB_state_hash"] for r in trajectories}),
      "debug_beam_unique_rgb_mse_count":len({struct.pack(">d",float(r["cumulative_RGB_MSE"])) for r in trajectories})}
    (ROOT/"debug_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    (ROOT/"fix.patch").write_text("""--- experiments/gvcrt_v13_generator_aware_mixed_precision/run_v13.py\n+++ experiments/gvcrt_v13_debug_mixed_precision/run_v13_debug.py\n@@ candidate proposal admission @@\n- saved_bits = max(0, -delta_bytes * 8)\n- if saved_bits > 0:\n-     ranked.append((delta_mse / saved_bits, block_index, mode, saved_bits, delta_mse))\n+ estimated_saved_bits = estimated_bits_baseline - estimated_bits_candidate\n+ if estimated_saved_bits > 0:\n+     proposal_pool.append((delta_rgb_mse / estimated_saved_bits, block_index, mode, estimated_saved_bits, delta_rgb_mse))\n""")
    (ROOT/"bug_log.txt").write_text("""Debug execution log\n- First launch stopped before model execution because the independent debug directory did not add the repository root to sys.path. The debug script was corrected; the traceback remains in logs/run.log.\n- Old V13 candidate admission computes saved_bits from whole-frame actual byte delta and appends a candidate only when saved_bits > 0.\n- The debug implementation admits candidates when estimated_saved_bits > 0 and retains actual RANS bytes as measured outputs.\n- The first completed debug beam retained an all-1x optimized state identical to the separately retained baseline path. The debug beam was corrected to expand the baseline path separately and exclude that duplicate from optimized beam slots; the complete debug experiment was rerun.\n- No change was applied to the copied mixed quantizer, RANS, mode-map, or decoder reconstruction implementation.\n- The original V13 directory was not modified.\n""")
    print(str(ROOT)); print("baseline identity:","PASS" if baseline_pass else "FAIL")
    print("q differs 1x/2x/4x:",not row2["q_equal_1x"],not row4["q_equal_1x"])
    print("independent decode 2x/4x:",a2["decode_pass"],a4["decode_pass"])
    print("forced fraction tests: COMPLETE"); print("old candidate filtering audit: COMPLETE")
    print("new sub-byte candidate maps: GENERATED"); print("8-frame beam sanity: COMPLETE")
    print("outputs: baseline_frame_debug.csv, single_block_debug.csv, single_block_tensor_stats.json, single_block_decode_audit.csv, forced_fraction_debug.csv, forced_fraction_frame_debug.csv, candidate_filter_audit.csv, candidate_filter_logic.txt, new_frame_candidate_maps.csv, debug_beam_states.csv, debug_trajectories.csv, debug_checks.json, implementation_trace.txt, fix.patch, bug_log.txt, bitstreams/, tensors/, run.log")


if __name__=="__main__":
    try: main()
    except Exception:
        (ROOT/"FAILED.json").write_text(json.dumps({"status":"FAIL","traceback":traceback.format_exc()},indent=2)+"\n")
        raise
