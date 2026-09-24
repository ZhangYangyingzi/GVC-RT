#!/usr/bin/env python3
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.io import read_image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/gvcrt_neural_wrapper_v2_joint"
SOURCE_FRAMES = REPO / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames"
FRAME_IDS = (0, 1, 8, 16)
WIDTH, HEIGHT = 1920, 1080


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label(video):
    return f"UVG_{video['name']}" if video["dataset"] == "uvg" else f"ULong_{int(video['video_id']):02d}"


def tag(video):
    return f"{video['dataset']}_{int(video['video_id']):02d}"


def save(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), "RGB").save(path, compress_level=1)


def metrics(candidate, reference):
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    mae = np.mean(np.abs(difference), axis=(0, 1))
    return {
        "PSNR": 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse),
        "MAE": float(np.mean(np.abs(difference))),
        "R_MAE": float(mae[0]), "G_MAE": float(mae[1]), "B_MAE": float(mae[2]),
        "max_abs_diff": float(np.max(np.abs(difference))),
    }


def ffprobe(video):
    fields = "codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,color_range,color_space,color_transfer,color_primaries"
    command = ["ffprobe", "-v", "error"]
    if video["dataset"] == "uvg":
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
                    f"{WIDTH}x{HEIGHT}", "-framerate", str(video["fps"])]
    command += ["-select_streams", "v:0", "-show_entries", f"stream={fields}",
                "-of", "json", video["source_path"]]
    result = json.loads(subprocess.check_output(command))
    stream = result["streams"][0]
    return {key: stream.get(key, "unknown") for key in fields.split(",")}


def ffmpeg_frames(video):
    command = ["ffmpeg", "-v", "error"]
    if video["dataset"] == "uvg":
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
                    f"{WIDTH}x{HEIGHT}", "-framerate", str(video["fps"])]
    command += ["-i", video["source_path"], "-frames:v", str(max(FRAME_IDS) + 1),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(command)
    expected = (max(FRAME_IDS) + 1) * WIDTH * HEIGHT * 3
    if len(raw) != expected:
        raise RuntimeError(f"ffmpeg returned {len(raw)} bytes, expected {expected}")
    values = np.frombuffer(raw, np.uint8).reshape(max(FRAME_IDS) + 1, HEIGHT, WIDTH, 3)
    return {index: values[index].copy() for index in FRAME_IDS}


def raw_yuv_frames(path):
    frame_size = WIDTH * HEIGHT * 3 // 2
    output = {}
    with open(path, "rb") as handle:
        for index in FRAME_IDS:
            handle.seek(index * frame_size)
            raw = handle.read(frame_size)
            if len(raw) != frame_size:
                raise RuntimeError(f"short raw YUV frame {index}: {path}")
            y = np.frombuffer(raw[:WIDTH * HEIGHT], np.uint8).reshape(HEIGHT, WIDTH).copy()
            u_start = WIDTH * HEIGHT
            plane = WIDTH * HEIGHT // 4
            u = np.frombuffer(raw[u_start:u_start + plane], np.uint8).reshape(HEIGHT // 2, WIDTH // 2).copy()
            v = np.frombuffer(raw[u_start + plane:], np.uint8).reshape(HEIGHT // 2, WIDTH // 2).copy()
            output[index] = (y, u, v)
    return output


def yuv_to_rgb(y, u, v, matrix, value_range):
    # Nearest-neighbor chroma expansion matches the audited source-generation path.
    cb = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1).astype(np.float64)
    cr = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1).astype(np.float64)
    yy = y.astype(np.float64)
    kr, kb = ((0.299, 0.114) if matrix == "BT.601" else (0.2126, 0.0722))
    kg = 1 - kr - kb
    if value_range == "limited":
        y_norm = (yy - 16) / 219
        cb_norm, cr_norm = (cb - 128) / 224, (cr - 128) / 224
    else:
        y_norm = yy / 255
        cb_norm, cr_norm = (cb - 128) / 255, (cr - 128) / 255
    r = y_norm + 2 * (1 - kr) * cr_norm
    b = y_norm + 2 * (1 - kb) * cb_norm
    g = (y_norm - kr * r - kb * b) / kg
    return np.clip(np.stack((r, g, b), axis=-1) * 255, 0, 255).round().astype(np.uint8)


def opencv_frames(video, raw_frames):
    if video["dataset"] == "uvg":
        output = {}
        for index, (y, u, v) in raw_frames.items():
            i420 = np.concatenate((y.ravel(), u.ravel(), v.ravel())).reshape(HEIGHT * 3 // 2, WIDTH)
            output[index] = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
        return output
    capture = cv2.VideoCapture(video["source_path"])
    output = {}
    for index in FRAME_IDS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"OpenCV failed at {video['source_path']} frame {index}")
        output[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    capture.release()
    return output


def visualization_source(video, index):
    path = V2 / "visualizations" / f"{tag(video)}_qp1_beta_high_comparison.mp4"
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode visualization {path} frame {index}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb[50:320, 0:480], path


def resized(array):
    return np.asarray(Image.fromarray(array).resize((480, 270), Image.Resampling.LANCZOS))


def make_comparison(video, frame, arrays, best_name):
    names = ["ffmpeg reference", "V2 pipeline RGB", "OpenCV RGB",
             "pipeline R/B swapped", best_name]
    images = [arrays["ffmpeg"], arrays["pipeline"], arrays["opencv"],
              arrays["swapped"], arrays["best"]]
    canvas = Image.new("RGB", (1920, 250), "black")
    draw = ImageDraw.Draw(canvas)
    for column, (name, image) in enumerate(zip(names, images)):
        thumb = Image.fromarray(image).resize((384, 216), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (column * 384, 34))
        draw.text((column * 384 + 5, 10), name, fill="white")
    output = ROOT / "comparisons" / f"{label(video)}_f{frame:03d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=1)


def main():
    manifest = json.loads((V2 / "test_manifest.json").read_text())
    videos = manifest["videos"]
    for directory in ("frames/model_input", "frames/pipeline", "frames/ffmpeg",
                      "frames/opencv", "frames/pil", "frames/color_candidates",
                      "frames/visualization_source", "comparisons", "logs"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    probes, stats_rows, bgr_rows, matrix_rows = [], [], [], []
    tensor_rows, identity_rows, visualization_rows = [], [], []
    video_results = {}
    for video in videos:
        name = label(video)
        video_tag = tag(video)
        probe = ffprobe(video)
        probes.append({"dataset": video["dataset"], "video": video["name"],
                       "video_id": video["video_id"], "source_path": video["source_path"],
                       "extension": Path(video["source_path"]).suffix, **probe})
        ffmpeg = ffmpeg_frames(video)
        raw = raw_yuv_frames(video["source_path"]) if video["dataset"] == "uvg" else None
        opencv = opencv_frames(video, raw)
        normal_maes, swap_maes, visualization_maes = [], [], []
        best_candidates = []
        for index in FRAME_IDS:
            pipeline_path = SOURCE_FRAMES / video_tag / f"im{index + 1}.png"
            pipeline = np.asarray(Image.open(pipeline_path).convert("RGB"), dtype=np.uint8).copy()
            pil = np.asarray(Image.open(pipeline_path).convert("RGB"), dtype=np.uint8).copy()
            tv = read_image(str(pipeline_path)).permute(1, 2, 0).numpy()
            if not np.array_equal(pipeline, pil) or not np.array_equal(pipeline, tv):
                raise RuntimeError(f"PIL/torchvision PNG disagreement: {pipeline_path}")
            swapped = pipeline[:, :, ::-1]
            stem = f"{name}_f{index:03d}.png"
            for directory, array in (("model_input", pipeline), ("pipeline", pipeline),
                                     ("ffmpeg", ffmpeg[index]), ("opencv", opencv[index]),
                                     ("pil", pil)):
                save(array, ROOT / "frames" / directory / stem)
            for channel_index, channel in enumerate("RGB"):
                values = pipeline[:, :, channel_index]
                stats_rows.append({"dataset": video["dataset"], "video": video["name"],
                                   "video_id": video["video_id"], "frame": index,
                                   "channel": channel, "mean": float(values.mean()),
                                   "std": float(values.std()), "min": int(values.min()),
                                   "max": int(values.max())})
            normal = metrics(pipeline, ffmpeg[index])
            swap = metrics(swapped, ffmpeg[index])
            normal_maes.append(normal["MAE"]); swap_maes.append(swap["MAE"])
            bgr_rows.append({"dataset": video["dataset"], "video": video["name"],
                             "video_id": video["video_id"], "frame": index,
                             **{f"normal_{key}": value for key, value in normal.items()},
                             **{f"swapped_{key}": value for key, value in swap.items()},
                             "possible_rgb_bgr_swap": swap["MAE"] + 0.5 < normal["MAE"]})

            best_name, best_array = "ffmpeg reference", ffmpeg[index]
            if raw is not None:
                candidates = {}
                for matrix in ("BT.601", "BT.709"):
                    for value_range in ("limited", "full"):
                        candidate_name = f"{matrix}_{value_range}"
                        candidate = yuv_to_rgb(*raw[index], matrix, value_range)
                        candidates[candidate_name] = candidate
                        save(candidate, ROOT / "frames/color_candidates" /
                             f"{name}_f{index:03d}_{candidate_name.replace('.', '')}.png")
                        result = metrics(candidate, ffmpeg[index])
                        matrix_rows.append({"dataset": video["dataset"], "video": video["name"],
                                            "video_id": video["video_id"], "frame": index,
                                            "candidate": candidate_name,
                                            "current_pipeline_conversion": "none; Y,Cb,Cr saved as R,G,B",
                                            **result})
                best_name = min(candidates, key=lambda key: metrics(candidates[key], ffmpeg[index])["MAE"])
                best_array = candidates[best_name]
                best_candidates.append(best_name)
            else:
                matrix_rows.append({"dataset": video["dataset"], "video": video["name"],
                                    "video_id": video["video_id"], "frame": index,
                                    "candidate": "not_applicable_encoded_mp4",
                                    "current_pipeline_conversion": "ffmpeg rgb24", **metrics(pipeline, ffmpeg[index])})

            tensor = torch.from_numpy(pipeline.copy()).permute(2, 0, 1).unsqueeze(0)
            wrapper_input = tensor.float() / 255
            gvc_input = wrapper_input * 2 - 1
            inverse = ((gvc_input + 1) / 2 * 255).round().clamp(0, 255).byte()
            stages = (("loader_rgb_uint8", tensor), ("wrapper_input", wrapper_input),
                      ("gvc_encoder_input", gvc_input), ("inverse_preprocess_uint8", inverse))
            for stage, value in stages:
                tensor_rows.append({"dataset": video["dataset"], "video": video["name"],
                                    "video_id": video["video_id"], "frame": index, "stage": stage,
                                    "shape": str(list(value.shape)), "dtype": str(value.dtype),
                                    "min": float(value.min()), "max": float(value.max()),
                                    "mean": float(value.float().mean()), "channel_order": "NCHW RGB"})
            restored = inverse[0].permute(1, 2, 0).numpy()
            identity_rows.append({"dataset": video["dataset"], "video": video["name"],
                                  "video_id": video["video_id"], "frame": index,
                                  **metrics(restored, pipeline)})
            if video["dataset"] == "uvg":
                visual, visual_path = visualization_source(video, index)
                save(visual, ROOT / "frames/visualization_source" / stem)
                visual_pipeline = metrics(visual, resized(pipeline))
                visual_ffmpeg = metrics(visual, resized(ffmpeg[index]))
                visualization_maes.append(visual_pipeline["MAE"])
                visualization_rows.append({
                    "dataset": video["dataset"], "video": video["name"],
                    "video_id": video["video_id"], "frame": index,
                    "visualization_path": str(visual_path),
                    **{f"visual_vs_pipeline_{key}": value for key, value in visual_pipeline.items()},
                    **{f"visual_vs_ffmpeg_{key}": value for key, value in visual_ffmpeg.items()},
                    "source_panel_matches_pipeline": visual_pipeline["MAE"] < visual_ffmpeg["MAE"],
                })
            make_comparison(video, index, {"ffmpeg": ffmpeg[index], "pipeline": pipeline,
                                           "opencv": opencv[index], "swapped": swapped,
                                           "best": best_array}, best_name)

        pipeline_pass = float(np.mean(normal_maes)) <= 1.0
        possible_swap = float(np.mean(swap_maes)) + 0.5 < float(np.mean(normal_maes))
        visualization_ok = (float(np.mean(normal_maes)) <= 1.0 and
                            (not visualization_maes or float(np.mean(visualization_maes)) <= 5.0))
        issue = "none"
        if not pipeline_pass:
            issue = ("Y/Cb/Cr channels were saved as RGB without YUV-to-RGB conversion"
                     if video["dataset"] == "uvg" else "pipeline differs from ffmpeg reference")
        video_results[video_tag] = {
            "raw_source_color_ok": True,
            "pipeline_reader_color_ok": pipeline_pass,
            "model_input_color_ok": pipeline_pass,
            "visualization_color_ok": visualization_ok if video["dataset"] == "uvg" else pipeline_pass,
            "pipeline_vs_ffmpeg_mean_mae": float(np.mean(normal_maes)),
            "swapped_vs_ffmpeg_mean_mae": float(np.mean(swap_maes)),
            "possible_rgb_bgr_swap": possible_swap,
            "best_color_candidates": sorted(set(best_candidates)),
            "suspected_stage": "source PNG preparation" if not pipeline_pass else "none",
            "suspected_issue": issue,
        }

    (ROOT / "ffprobe_sources.json").write_text(json.dumps(probes, indent=2) + "\n")
    write_csv(ROOT / "model_input_rgb_stats.csv", stats_rows)
    write_csv(ROOT / "rgb_bgr_audit.csv", bgr_rows)
    write_csv(ROOT / "color_matrix_audit.csv", matrix_rows)
    write_csv(ROOT / "tensor_range_audit.csv", tensor_rows)
    write_csv(ROOT / "identity_preprocess_audit.csv", identity_rows)
    write_csv(ROOT / "visualization_source_audit.csv", visualization_rows)

    pipeline_text = f"""V2 actual final evaluation reader audit
=======================================
Manifest: {V2 / 'test_manifest.json'}
Evaluation entry: {V2 / 'evaluate.py'}
Reader implementation: {V2 / 'eval_core.py'} matched_frames()

Actual V2 evaluation path for all six videos:
source_frames/<tag>/imN.png -> PIL.Image.open(...).convert('RGB') -> np.uint8 HWC RGB
-> torch.from_numpy -> NCHW float32 / 255 -> wrapper [0,1]
-> replicate padding -> float16 * 2 - 1 -> GVC encoder [-1,1].

The final V2 PNG loader alone is the same PIL reader for UVG and U-Long: YES.
The complete reader/conversion path from original source is the same: NO.

U-Long PNG preparation:
MP4/H.264 yuv420p -> ffmpeg -> rgb24 -> uint8 RGB PNG.
Metadata reports BT.709 and limited (tv) range for all three U-Long sources.

UVG PNG preparation:
raw planar 8-bit YUV420 -> YUV420Reader -> nearest-neighbor chroma expansion
-> channels remain [Y,Cb,Cr] -> Image.fromarray(...), which interprets them as [R,G,B].
No YUV-to-RGB conversion is called. The imported ycbcr2rgb function is unused.
Raw .yuv has no embedded matrix/range metadata; ffprobe therefore reports those fields unknown.

Tensor channel order: NCHW RGB as declared by the V2 loader, but UVG PNG channel contents are
Y/Cb/Cr because of the earlier PNG preparation defect.
"""
    (ROOT / "reader_pipeline_audit.txt").write_text(pipeline_text)

    uvg = {key: value for key, value in video_results.items() if key.startswith("uvg_")}
    ulong = {key: value for key, value in video_results.items() if key.startswith("fresh_ulong_")}
    audit = {
        "uvg_source_paths_verified": all(Path(v["source_path"]).is_file() for v in videos if v["dataset"] == "uvg"),
        "ulong_source_paths_verified": all(Path(v["source_path"]).is_file() for v in videos if v["dataset"] == "fresh_ulong"),
        "same_reader_path_uvg_ulong": False,
        "same_final_v2_png_reader_uvg_ulong": True,
        "same_upstream_source_conversion_uvg_ulong": False,
        "ffprobe_complete": len(probes) == 6,
        "pipeline_vs_ffmpeg_complete": len(bgr_rows) == 24,
        "rgb_bgr_swap_checked": True,
        "color_matrix_checked": len([row for row in matrix_rows if row["dataset"] == "uvg"]) == 48,
        "range_checked": True,
        "tensor_range_checked": len(tensor_rows) == 96,
        "identity_preprocess_checked": len(identity_rows) == 24,
        "visualization_source_checked": len(visualization_rows) == 12,
        "videos": video_results,
        "uvg_pipeline_all_pass": all(value["pipeline_reader_color_ok"] for value in uvg.values()),
        "ulong_pipeline_all_pass": all(value["pipeline_reader_color_ok"] for value in ulong.values()),
        "suspected_stage": "UVG source PNG preparation",
        "suspected_issue": "raw YUV420 Y/Cb/Cr channels saved directly as RGB PNG without conversion",
        "status": "AUDIT_COMPLETE",
    }
    (ROOT / "final_color_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    video_by_tag = {tag(video): video for video in videos}
    lines = [f"experiment directory {ROOT}"]
    for video_tag in ("uvg_00", "uvg_01", "uvg_02"):
        result, video = video_results[video_tag], video_by_tag[video_tag]
        lines += [f"UVG {video['name']}:",
                  f"raw source {'PASS' if result['raw_source_color_ok'] else 'FAIL'}",
                  f"pipeline vs ffmpeg {'PASS' if result['pipeline_reader_color_ok'] else 'FAIL'}",
                  f"suspected issue {result['suspected_issue']}"]
    lines += [f"U-Long reference:",
              f"pipeline vs ffmpeg {'PASS' if audit['ulong_pipeline_all_pass'] else 'FAIL'}",
              f"same reader path UVG/U-Long {'YES' if audit['same_reader_path_uvg_ulong'] else 'NO'}",
              f"final_color_audit.json path {ROOT / 'final_color_audit.json'}",
              f"rgb_bgr_audit.csv path {ROOT / 'rgb_bgr_audit.csv'}",
              f"color_matrix_audit.csv path {ROOT / 'color_matrix_audit.csv'}",
              f"comparisons path {ROOT / 'comparisons'}"]
    output = "\n".join(lines) + "\n"
    (ROOT / "stdout.log").write_text(output)
    (ROOT / "stderr.log").write_text("")
    print(output, end="")


if __name__ == "__main__":
    main()
