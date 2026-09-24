#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_v13 import (CFG, MAN, SPS, INDEX_MAP, Tee, average_metrics, frame_metric, init_metric_models,
                     load_triplet, packet, read_frames, sha256, tag, write_csv)
from src.utils.stream_helper import write_sps
import io


def run(video, qp, device):
    t = tag(video, qp); done = ROOT / "parts" / f"{t}_multi_qp_done.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "PASS": print("reuse_multi_qp", t, flush=True); return
    frames = read_frames(video, device); models = init_metric_models(device)
    iframe, encoder, decoder = load_triplet(device)
    encoder.clear_dpb(); decoder.clear_dpb(); encoder.set_curr_poc(0); decoder.set_curr_poc(0)
    stream = io.BytesIO(); write_sps(stream, SPS); rows = []
    for fi, frame in enumerate(frames):
        aq = qp if fi == 0 else encoder.shift_qp(qp, INDEX_MAP[fi % 8])
        if fi == 0:
            encoded = iframe.compress(frame, aq); payload = encoded["bit_stream"]
            out = iframe.decompress(payload, SPS, aq)["x_hat"]
            encoder.add_ref_frame(None, encoded["x_hat"]); decoder.add_ref_frame(None, out); is_i = True
        else:
            encoded = encoder.compress(frame, aq); payload = encoded["bit_stream"]
            out = decoder.decompress(payload, SPS, aq)["x_hat"]; is_i = False
            if not torch.equal(encoder.dpb[0].feature, decoder.dpb[0].feature): raise RuntimeError("causal state mismatch")
        stream.write(packet(payload, aq, is_i)); rows.append(dict(frame=fi, actual_qp=aq,
            actual_bytes=len(packet(payload, aq, is_i)), **frame_metric(out, frame, models)))
    path = ROOT / "bitstreams" / f"{t}_multi_qp.bin"; path.write_bytes(stream.getvalue()); avg = average_metrics(rows)
    result = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "QP": qp,
              "actual_total_bytes": len(stream.getvalue()), "actual_bits": len(stream.getvalue()) * 8,
              "kbps": len(stream.getvalue()) * 8 * video["fps"] / 64 / 1000,
              "bpp": len(stream.getvalue()) * 8 / (64 * 1920 * 1080),
              "PSNR": avg["PSNR"], "MS_SSIM": avg["MS_SSIM"], "LPIPS": avg["LPIPS"], "DISTS": avg["DISTS"],
              "bitstream_path": str(path), "bitstream_sha256": sha256(path), "decode_pass": True}
    write_csv(ROOT / "parts" / f"{t}_multi_qp.csv", [result])
    write_csv(ROOT / "parts" / f"{t}_multi_qp_frames.csv", [dict(dataset=video["dataset"], video=video["name"],
              video_id=video["video_id"], qp=qp, trajectory="multi_qp", **row) for row in rows])
    done.write_text(json.dumps({"status": "PASS", "tag": t}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--gpu", type=int, required=True); parser.add_argument("--cells", required=True)
    args = parser.parse_args(); sys.stdout = Tee(sys.stdout, ROOT / "logs" / f"multi_qp_gpu{args.gpu}.log")
    sys.stderr = Tee(sys.stderr, ROOT / "logs" / f"multi_qp_gpu{args.gpu}.log")
    torch.manual_seed(CFG["random_seed"]); device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)
    lookup = {(v["dataset"], int(v["video_id"])): v for v in MAN["videos"]}
    for spec in args.cells.split(","):
        dataset, video_id, qp = spec.split(":"); video = lookup[(dataset, int(video_id))]
        try: run(video, int(qp), device)
        except Exception as exc:
            (ROOT / "parts" / f"{tag(video, int(qp))}_multi_qp_FAILED.json").write_text(
                json.dumps({"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            raise


if __name__ == "__main__": main()
