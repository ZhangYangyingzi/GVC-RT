#!/usr/bin/env python3
import argparse
import csv
import gc
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

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V124 = ROOT.parent / "gvcrt_v12_4_sequence_discrete_beam"
V121 = ROOT.parent / "gvcrt_v12_1_multicontent_budget_oracle"
DBG = REPO / "expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug"
V11 = REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11"
V9 = REPO / "expericent_generation_input/expericent_interface_causal_controls_v9"
for p in (ROOT, REPO, V121, V9 / "src", V11 / "b2_latent_direction_magnitude_v11_7/src", DBG / "src"):
    sys.path.insert(0, str(p))

from common import sha256
from gvc_hooks import load_models
from run_debug import load_b2, unit
from run_v12_1 import read_frames, basic_metrics, init_metric_models, perceptual, INDEX_MAP
from src.models.video_model_gvcrt import RefFrame
from src.utils.stream_helper import write_sps, write_ip, read_header, read_sps_remaining, read_ip_remaining, NalType
from v13_codec import (all_one_modes, block_grid, decode_mixed, decode_probe_latent_cached, encode_mixed, encode_original,
                       gaussian_estimated_bits, map_stats, parse_mixed, prepare_frame,
                       prepare_probe_decode_cache, requantize_frame, tensor_hash)

CFG = json.loads((ROOT / "config.json").read_text())
MAN = json.loads((ROOT / "manifest.json").read_text())
SPS = {"sps_id": 0, "height": 1088, "width": 1920, "ec_part": 1, "use_ada_i": 0}


class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", buffering=1)
    def write(self, value):
        self.stream.write(value); self.file.write(value); return len(value)
    def flush(self): self.stream.flush(); self.file.flush()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open("w", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def tag(video, qp):
    return f"{video['dataset']}_{int(video['video_id']):02d}_qp{qp}"


def packet(payload, qp, is_i=False):
    out = io.BytesIO(); write_ip(out, is_i, 0, qp, payload); return out.getvalue()


def parse_stream(path):
    payloads, cumulative, sps_table = [], [], {}
    path = Path(path)
    with path.open("rb") as f:
        while f.tell() < path.stat().st_size:
            header = read_header(f)
            if header["nal_type"] == NalType.NAL_SPS:
                sps_table[header["sps_id"]] = read_sps_remaining(f, header["sps_id"]); continue
            qp, payload = read_ip_remaining(f)
            payloads.append((header["nal_type"] == NalType.NAL_I, qp, payload))
            cumulative.append(f.tell())
    if len(payloads) != 64 or cumulative[-1] != path.stat().st_size:
        raise RuntimeError("stream parse mismatch")
    return payloads, cumulative


def snap_cpu(model):
    return (model.curr_poc, [(r.poc,
             None if r.frame is None else r.frame.detach().cpu().clone(),
             None if r.feature is None else r.feature.detach().cpu().clone()) for r in model.dpb])


def restore_device(model, state, device):
    model.curr_poc, model.dpb = state[0], []
    for poc, frame, feature in state[1]:
        ref = RefFrame(); ref.poc = poc
        ref.frame = None if frame is None else frame.to(device)
        ref.feature = None if feature is None else feature.to(device)
        model.dpb.append(ref)


def state_hash(state):
    h = hashlib.sha256(str(state[0]).encode())
    for poc, frame, feature in state[1]:
        h.update(str(poc).encode())
        for value in (frame, feature):
            if value is not None: h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def model_hash(*models):
    h = hashlib.sha256()
    for model in models:
        for name, value in model.named_parameters():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frame_metric(rgb, gt, metric_models=None):
    mse, psnr, ssim = basic_metrics(rgb, gt)
    lp, di = ("", "") if metric_models is None else perceptual(rgb, gt, metric_models)
    return {"MSE": mse, "PSNR": psnr, "MS_SSIM": ssim, "LPIPS": lp, "DISTS": di}


def average_metrics(rows):
    out = {}
    for key in ("MSE", "PSNR", "MS_SSIM", "LPIPS", "DISTS"):
        vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
        out[key] = sum(vals) / len(vals) if vals else ""
    return out


def load_triplet(device):
    iframe, _ = load_models(device)
    encoder, _ = load_b2(device)
    decoder, _ = load_b2(device)
    return iframe, encoder, decoder


def init_sequence(iframe, encoder, decoder, first_frame, requested_qp):
    encoder.clear_dpb(); decoder.clear_dpb(); encoder.set_curr_poc(0); decoder.set_curr_poc(0)
    encoded = iframe.compress(first_frame, requested_qp)
    decoded = iframe.decompress(encoded["bit_stream"], SPS, requested_qp)["x_hat"]
    encoder.add_ref_frame(None, encoded["x_hat"]); decoder.add_ref_frame(None, decoded)
    stream = io.BytesIO(); write_sps(stream, SPS); stream.write(packet(encoded["bit_stream"], requested_qp, True))
    return stream.getvalue(), decoded, encoded["bit_stream"]


def baseline_and_identity(video, frames, requested_qp, device, metric_models):
    t = tag(video, requested_qp)
    ref_path = V124 / "bitstreams" / f"{t}_BASELINE.bin"
    ref_payloads, ref_cum = parse_stream(ref_path)
    iframe, encoder, decoder = load_triplet(device)
    before = model_hash(iframe, encoder, decoder)
    stream0, rgb0, i_payload = init_sequence(iframe, encoder, decoder, frames[0], requested_qp)
    stream = bytearray(stream0); metrics = [dict(frame=0, actual_qp=requested_qp,
                                                 actual_bytes=len(packet(i_payload, requested_qp, True)),
                                                 **frame_metric(rgb0, frames[0], metric_models))]
    causal = [state_hash(snap_cpu(decoder))]
    component = {"I_bytes": len(i_payload), "z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0,
                 "mode_map_bytes": 0, "header_bytes": 0}
    identity_symbols, identity_rgb = [], [tensor_hash(rgb0)]
    for fi in range(1, 64):
        aq = encoder.shift_qp(requested_qp, INDEX_MAP[fi % 8])
        prepared = prepare_frame(encoder, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
        payload, comp = encode_original(encoder, prepared, aq)
        if payload != ref_payloads[fi][2]:
            raise RuntimeError(f"identity payload mismatch at frame {fi}")
        out, captured = decode_mixed(decoder, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
        if not torch.equal(out, prepared["rgb"]):
            raise RuntimeError(f"identity RGB mismatch at frame {fi}")
        encoder.add_ref_frame(prepared["feature"], None)
        pkt = packet(payload, aq); stream.extend(pkt)
        met = frame_metric(out, frames[fi], metric_models)
        metrics.append(dict(frame=fi, actual_qp=aq, actual_bytes=len(pkt), **met))
        component["z_bytes"] += comp["z_bytes"]; component["w0_bytes"] += comp["w0_bytes"]
        component["w1_bytes"] += comp["w1_bytes"]
        identity_symbols.append((tensor_hash(prepared["z"]), tensor_hash(prepared["w0"]), tensor_hash(prepared["w1"])))
        identity_rgb.append(tensor_hash(out)); causal.append(state_hash(snap_cpu(decoder)))
    actual = bytes(stream)
    component["header_bytes"] = len(actual) - sum(component[k] for k in ("I_bytes", "z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"))
    ref = ref_path.read_bytes()
    if actual != ref:
        raise RuntimeError("identity full stream is not byte exact")
    if len(actual) != ref_cum[-1]:
        raise RuntimeError("identity stream length mismatch")
    out_path = ROOT / "bitstreams" / f"{t}_baseline.bin"; out_path.write_bytes(actual)
    after = model_hash(iframe, encoder, decoder)
    if before != after: raise RuntimeError("frozen parameter hash changed")
    avg = average_metrics(metrics)
    old = next(r for r in read_csv(V124 / "raw/baseline_cells.csv")
               if r["dataset"] == video["dataset"] and int(r["video_id"]) == int(video["video_id"]) and int(r["qp"]) == requested_qp)
    checks = {"bytes": int(old["baseline_bytes"]) == len(actual),
              "sha": old["bitstream_sha256"] == sha256(out_path),
              "psnr": abs(float(old["baseline_PSNR"]) - float(avg["PSNR"])) < 1e-9,
              "lpips": abs(float(old["baseline_LPIPS"]) - float(avg["LPIPS"])) < 1e-9,
              "dists": abs(float(old["baseline_DISTS"]) - float(avg["DISTS"])) < 1e-9}
    if not all(checks.values()): raise RuntimeError(f"V12.4 baseline mismatch: {checks}")
    row = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
           "qp": requested_qp, "frames": 64, "baseline_bytes": len(actual),
           "baseline_bits": len(actual) * 8, "bpp": len(actual) * 8 / (64 * 1920 * 1080),
           "kbps": len(actual) * 8 * video["fps"] / 64 / 1000, **avg, **component,
           "bitstream_path": str(out_path), "bitstream_sha256": sha256(out_path),
           "v12_4_sha256": old["bitstream_sha256"], "identity_gate_pass": True,
           "decoded_latent_symbol_exact": True, "decoded_rgb_exact": True,
           "causal_state_count": len(causal), "model_hash_before": before, "model_hash_after": after}
    return row, metrics


def evaluate_prepared(encoder, decoder, prepared, modes, qp, gt, decoder_state,
                      original=False, global_mode=None):
    restore_device(decoder, decoder_state, prepared["w1"].device)
    candidate = prepared if modes == prepared["modes"] else requantize_frame(
        encoder, prepared, modes, CFG["block_h"], CFG["block_w"])
    if original:
        payload, comp = encode_original(encoder, candidate, qp)
    else:
        payload, comp = encode_mixed(encoder, candidate, qp, CFG["block_h"], CFG["block_w"], global_mode)
    out, captured = decode_mixed(decoder, payload, SPS, qp, CFG["block_h"], CFG["block_w"])
    if not torch.equal(out, candidate["rgb"]):
        raise RuntimeError("encoder/decoder RGB mismatch")
    if (not torch.equal(captured["z"], candidate["z"]) or
        not torch.equal(captured["w0"], candidate["w0"]) or
        not torch.equal(captured["w1"], candidate["w1"])):
        raise RuntimeError("decoded entropy symbols mismatch")
    if captured["mixed"] and captured["modes"] != list(modes):
        raise RuntimeError("decoded mode-map mismatch")
    if not torch.equal(captured["latent"], candidate["latent"]):
        raise RuntimeError("decoded latent mismatch")
    state_err = float((captured["feature"] - candidate["feature"]).abs().max())
    if state_err != 0: raise RuntimeError(f"causal state mismatch {state_err}")
    mse, psnr, _ = basic_metrics(out, gt)
    est = gaussian_estimated_bits(encoder, candidate["w1"], candidate["s1"] / candidate["delta"], candidate["s1"])
    return candidate, payload, comp, out, {"MSE": mse, "PSNR": psnr}, snap_cpu(decoder), est


@torch.no_grad()
def build_probe_and_maps(video, frames, requested_qp, device):
    t = tag(video, requested_qp)
    iframe, encoder, decoder = load_triplet(device)
    stream0, _, _ = init_sequence(iframe, encoder, decoder, frames[0], requested_qp)
    probe_partial = ROOT / "parts" / f"{t}_block_probe.partial.csv"
    map_partial = ROOT / "parts" / f"{t}_frame_candidate_maps.partial.csv"
    prior_probe = read_csv(probe_partial) if probe_partial.exists() else []
    prior_maps = read_csv(map_partial) if map_partial.exists() else []
    probe_count = {}
    map_count = {}
    for row in prior_probe: probe_count[int(row["frame"])] = probe_count.get(int(row["frame"]), 0) + 1
    for row in prior_maps: map_count[int(row["frame"])] = map_count.get(int(row["frame"]), 0) + 1
    complete = {fi for fi in range(1, 64) if probe_count.get(fi) == 1020 and map_count.get(fi) == 5}
    probe_rows = [r for r in prior_probe if int(r["frame"]) in complete]
    map_rows = [r for r in prior_maps if int(r["frame"]) in complete]
    proposals = {fi: {r["candidate_map"]: json.loads(r["mode_map_json"])
                      for r in map_rows if int(r["frame"]) == fi} for fi in complete}
    exact = len(probe_rows); total_blocks = 0
    for fi in range(1, 64):
        # The native RANS extension retains encoder-side working storage across
        # resets. Renew it once per frame during the 64k-evaluation probe.
        encoder.update(.12); decoder.update(.12)
        encoder.set_use_two_entropy_coders(True); decoder.set_use_two_entropy_coders(True)
        aq = encoder.shift_qp(requested_qp, INDEX_MAP[fi % 8])
        dec_state = snap_cpu(decoder)
        base = prepare_frame(encoder, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
        ones = base["modes"]
        bprep, bpay, bcomp, bout, bmet, _, best = evaluate_prepared(
            encoder, decoder, base, ones, aq, frames[fi], dec_state, original=True)
        gh, gw = block_grid(base["shape"][-2], base["shape"][-1], CFG["block_h"], CFG["block_w"])
        total_blocks += gh * gw
        if fi in complete:
            restore_device(decoder, dec_state, device)
            out, _ = decode_mixed(decoder, bpay, SPS, aq, CFG["block_h"], CFG["block_w"])
            encoder.add_ref_frame(bprep["feature"], None)
            print(t, "reuse_probe_frame", fi, "exact", exact, flush=True)
            continue
        ranked = []
        base_packet_bytes = len(packet(bpay, aq))
        batch, cache = [], None
        def flush_probe_batch():
            nonlocal batch
            if not batch: return
            latents = torch.cat([entry["latent"] for entry in batch], dim=0)
            ctx = cache["ctx"].expand(latents.shape[0], *cache["ctx"].shape[1:])
            features = decoder.dec(latents, ctx, cache["qd"])
            rgbs = decoder.recon_generation_net(features, cache["qr"])
            target = unit(frames[fi]); errors = (unit(rgbs) - target).square().flatten(1).mean(1)
            for ci, entry in enumerate(batch):
                mse = float(errors[ci]); psnr = -10 * math.log10(max(mse, 1e-12))
                delta_bytes = entry["candidate_packet_bytes"] - base_packet_bytes
                dmse = mse - bmet["MSE"]; saved_bits = max(0, -delta_bytes * 8)
                ratio = dmse / saved_bits if saved_bits > 0 else ""
                bi, by, bx, mode, comp, est = (entry[k] for k in ("bi", "by", "bx", "mode", "comp", "est"))
                probe_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                   "qp": requested_qp, "frame": fi, "w1_shape": "x".join(map(str, base["shape"])),
                   "block_index": bi, "block_y": by, "block_x": bx,
                   "pixel_y0": by * CFG["block_h"], "pixel_x0": bx * CFG["block_w"],
                   "mode": mode, "delta": 2 if mode == 1 else 4,
                   "original_bytes": base_packet_bytes, "candidate_bytes": entry["candidate_packet_bytes"],
                   "actual_delta_bytes": delta_bytes, "baseline_RGB_MSE": bmet["MSE"],
                   "candidate_RGB_MSE": mse, "delta_MSE": dmse,
                   "baseline_PSNR": bmet["PSNR"], "candidate_PSNR": psnr,
                   "mode_map_overhead": comp["mode_map_bytes"] + comp["mixed_header_bytes"],
                   "raw_map_bits": comp["raw_map_bits"], "coded_map_bits": comp["coded_map_bits"],
                   "estimated_w1_entropy_bits": est, "actual_w1_RANS_bytes": comp["w1_bytes"],
                   "actual_bytes_saved": saved_bits > 0, "delta_MSE_per_actual_saved_bit": ratio})
                if saved_bits > 0: ranked.append((float(ratio), bi, mode, saved_bits, dmse))
            batch = []
            del latents, ctx, features, rgbs, target, errors
            torch.cuda.empty_cache()
        for bi in range(gh * gw):
            by, bx = divmod(bi, gw)
            for mode in (1, 2):
                modes = list(ones); modes[bi] = mode
                cand = requantize_frame(encoder, base, modes, CFG["block_h"], CFG["block_w"], reconstruct_rgb=False)
                if os.environ.get("V13_DEBUG_PROBE") == "1":
                    print(t, "probe_candidate", fi, bi, mode, flush=True)
                pay, comp = encode_mixed(encoder, cand, aq, CFG["block_h"], CFG["block_w"])
                if cache is None:
                    restore_device(decoder, dec_state, device)
                    cache = prepare_probe_decode_cache(decoder, pay, SPS, aq, CFG["block_h"], CFG["block_w"])
                latent, dz, dw0, dw1, parsed = decode_probe_latent_cached(
                    decoder, pay, cache, CFG["block_h"], CFG["block_w"])
                if (parsed["modes"] != modes or not torch.equal(dz, cand["z"]) or
                    not torch.equal(dw0, cand["w0"]) or not torch.equal(dw1, cand["w1"]) or
                    not torch.equal(latent, cand["latent"])):
                    raise RuntimeError("batched probe entropy decode mismatch")
                est = gaussian_estimated_bits(encoder, cand["w1"], cand["s1"] / cand["delta"], cand["s1"])
                exact += 1
                batch.append({"bi": bi, "by": by, "bx": bx, "mode": mode, "comp": comp, "est": est,
                              "candidate_packet_bytes": len(packet(pay, aq)), "latent": latent})
                if len(batch) >= CFG["probe_batch_size"]: flush_probe_batch()
        flush_probe_batch()
        ranked.sort(key=lambda x: (x[0], -x[3], x[1], x[2]))
        levels = [("all_1x", 0.0), ("mild", .025), ("medium", .05), ("aggressive", .10), ("aggressive_20", .20)]
        frame_maps = {}
        base_w1 = max(1, bcomp["w1_bytes"])
        for name, target in levels:
            modes = list(ones); chosen = set(); nominal = 0
            for _, bi, mode, saved, _ in ranked:
                if target == 0 or nominal >= target * base_w1 * 8: break
                if bi in chosen: continue
                modes[bi] = mode; chosen.add(bi); nominal += saved
            if name == "all_1x":
                cand, pay, comp, out, met, _, est = bprep, bpay, bcomp, bout, bmet, dec_state, best
            else:
                cand, pay, comp, out, met, _, est = evaluate_prepared(
                    encoder, decoder, base, modes, aq, frames[fi], dec_state)
            stats = map_stats(modes)
            actual_packet_bytes = len(packet(pay, aq))
            map_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                             "qp": requested_qp, "frame": fi, "candidate_map": name,
                             "proposal_target": target, "mode_map_json": json.dumps(modes, separators=(",", ":")),
                             "blocks": len(modes), **stats, "actual_frame_bytes": actual_packet_bytes,
                             "baseline_frame_bytes": base_packet_bytes,
                             "actual_delta_bytes": actual_packet_bytes - base_packet_bytes,
                             "RGB_MSE": met["MSE"], "PSNR": met["PSNR"],
                             "estimated_w1_entropy_bits": est, "actual_w1_RANS_bytes": comp["w1_bytes"]})
            frame_maps[name] = modes
        proposals[fi] = frame_maps
        # Advance only the immutable original baseline trajectory.
        restore_device(decoder, dec_state, device)
        out, _ = decode_mixed(decoder, bpay, SPS, aq, CFG["block_h"], CFG["block_w"])
        encoder.add_ref_frame(bprep["feature"], None)
        write_csv(probe_partial, probe_rows)
        write_csv(map_partial, map_rows)
        print(t, "probe_frame", fi, "exact", exact, flush=True)
    write_csv(ROOT / "parts" / f"{t}_block_probe.csv", probe_rows)
    write_csv(ROOT / "parts" / f"{t}_frame_candidate_maps.csv", map_rows)
    (ROOT / "artifacts" / f"{t}_proposal_maps.json").write_text(json.dumps(proposals, separators=(",", ":")))
    return proposals, {"total_spatial_blocks": total_blocks, "probed_blocks": total_blocks,
                       "one_to_two_exact_evaluations": total_blocks,
                       "one_to_four_exact_evaluations": total_blocks,
                       "probe_exact_RANS_encodes": exact}


def deduplicate(states):
    unique, duplicate = {}, 0
    for state in states:
        key = (state_hash(state["dec_state"]), hashlib.sha256(state["stream"]).hexdigest(),
               state["cum_bytes"], struct.pack(">d", state["cum_mse"]))
        if key in unique:
            duplicate += 1
            if state["cum_mse"] < unique[key]["cum_mse"]: unique[key] = state
        else: unique[key] = state
    return list(unique.values()), duplicate


def choose_beam(states, frame, baseline_total):
    states, duplicates = deduplicate(states)
    pareto = []
    for i, a in enumerate(states):
        dominated = any(i != j and b["cum_bytes"] <= a["cum_bytes"] and b["cum_mse"] <= a["cum_mse"]
                        and (b["cum_bytes"] < a["cum_bytes"] or b["cum_mse"] < a["cum_mse"])
                        for j, b in enumerate(states))
        if not dominated: pareto.append(a)
    chosen = []
    progress = frame / 63
    for target in CFG["target_rate_ratio"]:
        desired = 1 - (1 - target) * progress
        candidates = sorted(pareto, key=lambda s: (abs(s["cum_bytes"] / max(1, s["baseline_cum"]) - desired), s["cum_mse"]))
        if candidates and all(candidates[0] is not x for x in chosen): chosen.append(candidates[0])
    for key in (lambda s: (s["cum_mse"], s["cum_bytes"]), lambda s: (s["cum_bytes"], s["cum_mse"])):
        for state in sorted(pareto, key=key):
            if len(chosen) >= CFG["beam_width"]: break
            if all(state is not x for x in chosen): chosen.append(state)
    for state in sorted(states, key=lambda s: (s["cum_mse"], s["cum_bytes"])):
        if len(chosen) >= CFG["beam_width"]: break
        if all(state is not x for x in chosen): chosen.append(state)
    return chosen[:CFG["beam_width"]], duplicates


def causal_search(video, frames, requested_qp, device, proposals, baseline_total, baseline_cumulative):
    t = tag(video, requested_qp)
    iframe, encoder, decoder = load_triplet(device)
    initial_stream, rgb0, _ = init_sequence(iframe, encoder, decoder, frames[0], requested_qp)
    mse0 = basic_metrics(rgb0, frames[0])[0]
    root = {"tid": f"{t}_root", "enc_state": snap_cpu(encoder), "dec_state": snap_cpu(decoder),
            "stream": initial_stream, "cum_bytes": len(initial_stream), "cum_mse": mse0,
            "steps": [], "modified_frames": 0, "mode_counts": [0, 0, 0], "artifacts": []}
    beam = [root]; beam_rows, step_rows = [], []
    expanded_count = exact_encodes = duplicate_count = numerical_failures = 0
    serial = 0
    for fi in range(1, 64):
        aq = encoder.shift_qp(requested_qp, INDEX_MAP[fi % 8]); expanded = []
        for parent in beam:
            expanded_count += 1
            restore_device(encoder, parent["enc_state"], device); restore_device(decoder, parent["dec_state"], device)
            prepared = prepare_frame(encoder, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
            parent_dec = snap_cpu(decoder)
            for map_name, modes in proposals[fi].items():
                try:
                    original = map_name == "all_1x"
                    cand, payload, comp, out, met, new_dec, est = evaluate_prepared(
                        encoder, decoder, prepared, modes, aq, frames[fi], parent_dec, original=original)
                    exact_encodes += 1; serial += 1
                    pkt = packet(payload, aq); new_stream = parent["stream"] + pkt
                    restore_device(encoder, parent["enc_state"], device); encoder.add_ref_frame(cand["feature"], None)
                    counts = [parent["mode_counts"][i] + modes.count(i) for i in range(3)]
                    step = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                            "qp": requested_qp, "trajectory_id": f"{t}_T{serial}", "parent_trajectory_id": parent["tid"],
                            "frame": fi, "candidate_map": map_name, "actual_frame_bytes": len(pkt),
                            "cumulative_actual_bytes": len(new_stream), "baseline_cumulative_bytes": baseline_cumulative[fi],
                            "frame_RGB_MSE": met["MSE"], "cumulative_RGB_MSE": parent["cum_mse"] + met["MSE"],
                            "mode_map_bytes": comp["mode_map_bytes"], "w1_bytes": comp["w1_bytes"],
                            "estimated_w1_entropy_bits": est, "DPB_state_hash": state_hash(new_dec),
                            "bitstream_hash": hashlib.sha256(new_stream).hexdigest()}
                    state = {"tid": step["trajectory_id"], "enc_state": snap_cpu(encoder), "dec_state": new_dec,
                             "stream": new_stream, "cum_bytes": len(new_stream),
                             "cum_mse": parent["cum_mse"] + met["MSE"], "steps": parent["steps"] + [step],
                             "modified_frames": parent["modified_frames"] + (not original), "mode_counts": counts,
                             "baseline_cum": baseline_cumulative[fi],
                             "artifacts": parent["artifacts"] + [{"frame": fi, "modes": modes,
                                "z": cand["z"].cpu(), "w0": cand["w0"].cpu(), "w1": cand["w1"].cpu(),
                                "scales": cand["s1"].cpu(), "latent_hash": tensor_hash(cand["latent"]),
                                "feature_hash": tensor_hash(cand["feature"]), "rgb_hash": tensor_hash(out)}]}
                    expanded.append(state)
                except (RuntimeError, ValueError, FloatingPointError):
                    numerical_failures += 1
        beam, dup = choose_beam(expanded, fi, baseline_total); duplicate_count += dup
        for rank, state in enumerate(beam):
            state["steps"][-1]["entered_beam"] = True
            beam_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                              "qp": requested_qp, "frame": fi, "trajectory_id": state["tid"],
                              "cumulative_actual_bytes": state["cum_bytes"], "baseline_cumulative_bytes": baseline_cumulative[fi],
                              "rate_ratio_to_prefix_baseline": state["cum_bytes"] / baseline_cumulative[fi],
                              "cumulative_RGB_MSE": state["cum_mse"], "beam_rank": rank,
                              "DPB_state_hash": state_hash(state["dec_state"]),
                              "bitstream_hash": hashlib.sha256(state["stream"]).hexdigest()})
        step_rows.extend(state["steps"][-1] for state in beam)
        if fi % 4 == 0:
            write_csv(ROOT / "parts" / f"{t}_beam_states.partial.csv", beam_rows)
            write_csv(ROOT / "parts" / f"{t}_trajectory_steps.partial.csv", step_rows)
            print(t, "search_frame", fi, "beam", len(beam), "expanded", expanded_count, flush=True)
    final_rows, selected = [], {}
    for target in CFG["target_rate_ratio"]:
        target_bytes = math.floor(baseline_total * target)
        feasible = [s for s in beam if s["cum_bytes"] <= target_bytes]
        best = min(feasible, key=lambda s: s["cum_mse"]) if feasible else None
        selected[target] = best
        final_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                           "qp": requested_qp, "target_rate_ratio": target, "target_bytes": target_bytes,
                           "trajectory_id": "" if best is None else best["tid"],
                           "selection_type": "TARGET_NOT_REACHED" if best is None else "MIXED_PRECISION",
                           "selected_total_bytes": "" if best is None else best["cum_bytes"],
                           "cumulative_RGB_MSE": "" if best is None else best["cum_mse"]})
    write_csv(ROOT / "parts" / f"{t}_beam_states.csv", beam_rows)
    write_csv(ROOT / "parts" / f"{t}_trajectory_steps.csv", step_rows)
    write_csv(ROOT / "parts" / f"{t}_final_trajectories.csv", final_rows)
    return selected, {"sequence_candidate_maps_evaluated": exact_encodes,
                      "search_exact_RANS_encodes": exact_encodes,
                      "beam_trajectories_expanded": expanded_count,
                      "duplicated_beam_state_hashes": duplicate_count,
                      "numerical_failures": numerical_failures}


def run_uniform(video, frames, requested_qp, device, mode):
    t = tag(video, requested_qp); iframe, encoder, decoder = load_triplet(device)
    stream, rgb0, ip = init_sequence(iframe, encoder, decoder, frames[0], requested_qp)
    rows = [dict(frame=0, actual_qp=requested_qp, actual_bytes=len(packet(ip, requested_qp, True)),
                 **frame_metric(rgb0, frames[0]))]
    totals = {"I_bytes": len(ip), "z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0,
              "mode_map_bytes": 0, "header_bytes": 0}
    for fi in range(1, 64):
        aq = encoder.shift_qp(requested_qp, INDEX_MAP[fi % 8])
        base = prepare_frame(encoder, frames[fi], aq, None, CFG["block_h"], CFG["block_w"])
        modes = [mode] * len(base["modes"]); cand = requantize_frame(encoder, base, modes, CFG["block_h"], CFG["block_w"])
        payload, comp = encode_mixed(encoder, cand, aq, CFG["block_h"], CFG["block_w"], global_mode=mode)
        out, cap = decode_mixed(decoder, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
        if cap["modes"] != modes or not torch.equal(cap["w1"], cand["w1"]): raise RuntimeError("uniform decode mismatch")
        encoder.add_ref_frame(cand["feature"], None); pkt = packet(payload, aq); stream += pkt
        rows.append(dict(frame=fi, actual_qp=aq, actual_bytes=len(pkt), **frame_metric(out, frames[fi])))
        for key in ("z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"): totals[key] += comp[key]
    totals["header_bytes"] = len(stream) - sum(totals[k] for k in ("I_bytes", "z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"))
    path = ROOT / "bitstreams" / f"{t}_uniform_{2 if mode == 1 else 4}x.bin"; path.write_bytes(stream)
    return {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "qp": requested_qp,
            "control": f"uniform_{2 if mode == 1 else 4}x", "global_delta": 2 if mode == 1 else 4,
            "total_actual_bytes": len(stream), "actual_bits": len(stream) * 8,
            "bpp": len(stream) * 8 / (64 * 1920 * 1080), "kbps": len(stream) * 8 * video["fps"] / 64 / 1000,
            **totals, **average_metrics(rows), "bitstream_path": str(path), "bitstream_sha256": sha256(path),
            "decode_audit_pass": True}, rows


def audit_selected(video, frames, requested_qp, device, target, state):
    if state is None: return None, []
    t = tag(video, requested_qp); path = ROOT / "bitstreams" / f"{t}_target_{target:.2f}.bin"
    path.write_bytes(state["stream"]); payloads, cumulative = parse_stream(path)
    iframe, _, decoder = load_triplet(device); decoder.clear_dpb(); decoder.set_curr_poc(0)
    rows = []; consumed = 0; symbols_ok = modes_ok = latent_ok = state_ok = rgb_ok = True
    expected = {a["frame"]: a for a in state["artifacts"]}
    for fi, (is_i, aq, payload) in enumerate(payloads):
        if is_i:
            out = iframe.decompress(payload, SPS, aq)["x_hat"]; decoder.add_ref_frame(None, out)
        else:
            out, cap = decode_mixed(decoder, payload, SPS, aq, CFG["block_h"], CFG["block_w"])
            exp = expected[fi]
            symbols_ok &= (torch.equal(cap["z"].cpu(), exp["z"]) and torch.equal(cap["w0"].cpu(), exp["w0"])
                           and torch.equal(cap["w1"].cpu(), exp["w1"]))
            modes_ok &= (cap["modes"] == exp["modes"] if cap["mixed"] else all(x == 0 for x in exp["modes"]))
            latent_ok &= tensor_hash(cap["latent"]) == exp["latent_hash"]
            state_ok &= tensor_hash(decoder.dpb[0].feature) == exp["feature_hash"]
            rgb_ok &= tensor_hash(out) == exp["rgb_hash"]
        consumed = cumulative[fi]
        rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
                     "qp": requested_qp, "target_rate_ratio": target, "frame": fi,
                     "actual_qp": aq, "actual_bytes": len(packet(payload, aq, is_i)),
                     **frame_metric(out, frames[fi])})
    passed = len(payloads) == 64 and consumed == path.stat().st_size and symbols_ok and modes_ok and latent_ok and state_ok and rgb_ok
    audit = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "qp": requested_qp,
             "target_rate_ratio": target, "bitstream_path": str(path), "bitstream_sha256": sha256(path),
             "frames": len(payloads), "decoded_quantized_symbols_match": symbols_ok,
             "decoded_precision_map_match": modes_ok, "decoded_latent_match": latent_ok,
             "decoder_causal_state_match": state_ok, "final_RGB_match": rgb_ok,
             "total_consumed_bytes": consumed, "file_bytes": path.stat().st_size,
             "decode_audit_pass": passed}
    torch.save({"target_rate_ratio": target, "frames": state["artifacts"]}, ROOT / "artifacts" / f"{t}_target_{target:.2f}_symbols_contexts.pt")
    return audit, rows


def component_totals(path):
    payloads, _ = parse_stream(path); totals = {"I_bytes": 0, "z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0,
                                                 "mode_map_bytes": 0, "header_bytes": 0}
    for is_i, qp, payload in payloads:
        pkt_len = len(packet(payload, qp, is_i))
        if is_i: totals["I_bytes"] += len(payload); continue
        parsed = parse_mixed(payload, CFG["block_h"], CFG["block_w"])
        if parsed is None:
            totals["w1_bytes"] += len(payload)
        else:
            totals["z_bytes"] += len(parsed["z_stream"]); totals["w0_bytes"] += len(parsed["w0_stream"])
            totals["w1_bytes"] += len(parsed["w1_stream"]); totals["mode_map_bytes"] += len(parsed["map_stream"])
    totals["header_bytes"] = Path(path).stat().st_size - sum(totals[k] for k in ("I_bytes", "z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"))
    return totals


def run_cell(video, requested_qp, device):
    t = tag(video, requested_qp); done = ROOT / "parts" / f"{t}_done.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "PASS": print("reuse", t, flush=True); return
    frames = read_frames(video, device); metric_models = init_metric_models(device)
    baseline, baseline_frames = baseline_and_identity(video, frames, requested_qp, device, metric_models)
    write_csv(ROOT / "parts" / f"{t}_baseline_cells.csv", [baseline])
    write_csv(ROOT / "parts" / f"{t}_baseline_frame_metrics.csv", [dict(dataset=video["dataset"], video=video["name"], video_id=video["video_id"], qp=requested_qp, trajectory="baseline", **r) for r in baseline_frames])
    del metric_models; gc.collect(); torch.cuda.empty_cache()
    proposals, coverage_probe = build_probe_and_maps(video, frames, requested_qp, device)
    _, baseline_cum = parse_stream(baseline["bitstream_path"])
    selected, coverage_search = causal_search(video, frames, requested_qp, device, proposals, baseline["baseline_bytes"], baseline_cum)
    uniform_rows, frame_rows = [], []
    for mode in (1, 2):
        row, fr = run_uniform(video, frames, requested_qp, device, mode); uniform_rows.append(row)
        frame_rows.extend(dict(dataset=video["dataset"], video=video["name"], video_id=video["video_id"], qp=requested_qp, trajectory=row["control"], **x) for x in fr)
    write_csv(ROOT / "parts" / f"{t}_uniform_controls.csv", uniform_rows)
    audits, final_rows = [], []
    for target in CFG["target_rate_ratio"]:
        state = selected[target]; audit, fr = audit_selected(video, frames, requested_qp, device, target, state)
        if audit is not None: audits.append(audit); frame_rows.extend(dict(trajectory=f"target_{target:.2f}", **x) for x in fr)
        target_bytes = math.floor(baseline["baseline_bytes"] * target)
        if state is None:
            final_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "QP": requested_qp,
                               "target_rate_ratio": target, "baseline_total_bytes": baseline["baseline_bytes"],
                               "target_bytes": target_bytes, "selected_total_bytes": "", "selected_rate_ratio": "",
                               "selection_type": "TARGET_NOT_REACHED", "I_bytes": "", "z_bytes": "", "w0_bytes": "",
                               "w1_bytes": "", "mode_map_bytes": "", "modified_frames": "", "number_1x_blocks": "",
                               "number_2x_blocks": "", "number_4x_blocks": "", "PSNR": "", "MS-SSIM": "",
                               "LPIPS": "", "DISTS": "", "decode_audit_pass": ""})
        else:
            met = average_metrics(fr); comp = component_totals(audit["bitstream_path"])
            final_rows.append({"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "QP": requested_qp,
                               "target_rate_ratio": target, "baseline_total_bytes": baseline["baseline_bytes"],
                               "target_bytes": target_bytes, "selected_total_bytes": state["cum_bytes"],
                               "selected_rate_ratio": state["cum_bytes"] / baseline["baseline_bytes"],
                               "selection_type": "MIXED_PRECISION", **comp, "modified_frames": state["modified_frames"],
                               "number_1x_blocks": state["mode_counts"][0], "number_2x_blocks": state["mode_counts"][1],
                               "number_4x_blocks": state["mode_counts"][2], "PSNR": met["PSNR"], "MS-SSIM": met["MS_SSIM"],
                               "LPIPS": met["LPIPS"], "DISTS": met["DISTS"], "decode_audit_pass": audit["decode_audit_pass"]})
    write_csv(ROOT / "parts" / f"{t}_final_streams.csv", final_rows)
    write_csv(ROOT / "parts" / f"{t}_bitstream_audit.csv", audits)
    write_csv(ROOT / "parts" / f"{t}_frame_metrics.csv", frame_rows)
    seq_rows = [dict(dataset=video["dataset"], video=video["name"], video_id=video["video_id"], qp=requested_qp,
                     trajectory=r["control"], total_actual_bytes=r["total_actual_bytes"], bpp=r["bpp"], kbps=r["kbps"],
                     PSNR=r["PSNR"], MS_SSIM=r["MS_SSIM"], LPIPS=r["LPIPS"], DISTS=r["DISTS"]) for r in uniform_rows]
    seq_rows.insert(0, dict(dataset=video["dataset"], video=video["name"], video_id=video["video_id"], qp=requested_qp,
                            trajectory="baseline", total_actual_bytes=baseline["baseline_bytes"], bpp=baseline["bpp"], kbps=baseline["kbps"],
                            PSNR=baseline["PSNR"], MS_SSIM=baseline["MS_SSIM"], LPIPS=baseline["LPIPS"], DISTS=baseline["DISTS"]))
    for r in final_rows:
        if r["selection_type"] == "MIXED_PRECISION":
            seq_rows.append(dict(dataset=r["dataset"], video=r["video"], video_id=r["video_id"], qp=requested_qp,
                                 trajectory=f"mixed_{r['target_rate_ratio']}", total_actual_bytes=r["selected_total_bytes"],
                                 bpp=int(r["selected_total_bytes"]) * 8 / (64 * 1920 * 1080),
                                 kbps=int(r["selected_total_bytes"]) * 8 * video["fps"] / 64 / 1000,
                                 PSNR=r["PSNR"], MS_SSIM=r["MS-SSIM"], LPIPS=r["LPIPS"], DISTS=r["DISTS"]))
    write_csv(ROOT / "parts" / f"{t}_sequence_metrics.csv", seq_rows)
    coverage = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "qp": requested_qp,
                **coverage_probe, **coverage_search,
                "exact_RANS_encodes": coverage_probe["probe_exact_RANS_encodes"] + coverage_search["search_exact_RANS_encodes"],
                "unique_final_bitstream_hashes": len({a["bitstream_sha256"] for a in audits}),
                "decode_failures": sum(not a["decode_audit_pass"] for a in audits)}
    write_csv(ROOT / "parts" / f"{t}_search_coverage.csv", [coverage])
    ok = baseline["identity_gate_pass"] and all(a["decode_audit_pass"] for a in audits)
    done.write_text(json.dumps({"status": "PASS" if ok else "FAIL", "tag": t, "identity_gate_pass": baseline["identity_gate_pass"],
                                "target_searches_complete": 3, "decode_audit_pass": sum(a["decode_audit_pass"] for a in audits)}, indent=2) + "\n")
    if not ok: raise RuntimeError(f"cell failed {t}")


def smoke(device):
    video = next(v for v in MAN["videos"] if v["dataset"] == "fresh_ulong" and int(v["video_id"]) == 10)
    frames = read_frames(video, device); iframe, encoder, decoder = load_triplet(device)
    _, _, _ = init_sequence(iframe, encoder, decoder, frames[0], 1)
    aq = encoder.shift_qp(1, INDEX_MAP[1]); dec_state = snap_cpu(decoder)
    base = prepare_frame(encoder, frames[1], aq, None, CFG["block_h"], CFG["block_w"])
    _, original, _, _, _, _, _ = evaluate_prepared(encoder, decoder, base, base["modes"], aq, frames[1], dec_state, original=True)
    modes = [0] * len(base["modes"]); modes[0] = 1; modes[-1] = 2
    candidate, mixed, comp, _, _, _, _ = evaluate_prepared(encoder, decoder, base, modes, aq, frames[1], dec_state)
    parsed = parse_mixed(mixed, CFG["block_h"], CFG["block_w"])
    restore_device(decoder, dec_state, device); cache = prepare_probe_decode_cache(decoder, mixed, SPS, aq, CFG["block_h"], CFG["block_w"])
    latent, z, w0, w1, _ = decode_probe_latent_cached(decoder, mixed, cache, CFG["block_h"], CFG["block_w"])
    for _ in range(128):
        latent, z, w0, w1, _ = decode_probe_latent_cached(decoder, mixed, cache, CFG["block_h"], CFG["block_w"])
    latents = torch.cat([latent] * CFG["probe_batch_size"], 0)
    ctx = cache["ctx"].expand(latents.shape[0], *cache["ctx"].shape[1:])
    smoke_rgb = decoder.recon_generation_net(decoder.dec(latents, ctx, cache["qd"]), cache["qr"])
    result = {"status": "PASS", "w1_shape": base["shape"], "blocks": len(modes),
              "original_payload_bytes": len(original), "mixed_payload_bytes": len(mixed),
              "mode_map_bytes": comp["mode_map_bytes"], "coded_map_bits": comp["coded_map_bits"],
              "decoded_modes_equal": parsed["modes"] == modes,
              "cached_probe_decode_equal": bool(torch.equal(latent, candidate["latent"]) and torch.equal(z, candidate["z"])
                                                  and torch.equal(w0, candidate["w0"]) and torch.equal(w1, candidate["w1"])),
              "probe_batch_size": CFG["probe_batch_size"], "probe_batch_rgb_shape": list(smoke_rgb.shape),
              "cached_RANS_decode_stress_iterations": 129}
    (ROOT / "logs/smoke.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cells", default=""); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", default=""); args = parser.parse_args()
    suffix = f"_{args.worker}" if args.worker else ""
    sys.stdout = Tee(sys.stdout, ROOT / "logs" / f"gpu{args.gpu}{suffix}.log")
    sys.stderr = Tee(sys.stderr, ROOT / "logs" / f"gpu{args.gpu}{suffix}.log")
    # Address the requested physical GPU directly. Changing CUDA visibility
    # after torch import is driver-version dependent.
    torch.manual_seed(CFG["random_seed"])
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if args.smoke: smoke(device); return
    if not args.cells: parser.error("--cells is required unless --smoke is used")
    lookup = {(v["dataset"], int(v["video_id"])): v for v in MAN["videos"]}
    for spec in args.cells.split(","):
        dataset, video_id, qp = spec.split(":"); video = lookup[(dataset, int(video_id))]
        try: run_cell(video, int(qp), device)
        except Exception as exc:
            fail = ROOT / "parts" / f"{tag(video, int(qp))}_FAILED.json"
            fail.write_text(json.dumps({"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            raise


if __name__ == "__main__": main()
