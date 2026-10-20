"""Read-only four-way latent intervention on the v6.37 verified RANS replay.

All historical modules are imported from their original path for reading.  No
historical output helper is called: every new file is rooted in this directory.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = HERE.parent / "gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500"
sys.path.insert(0, str(OLD))
from receiver37 import (runtime_original, receiver_bank, metadata, sourcepoint,
                        parse, tag, th, difference, sha)  # noqa: E402

METHODS = {
    "M00": ("Y_originalBridge_fullG", "z0", "z0"),
    "M10": ("C_zF_spatial_z0_stats", "zF", "z0"),
    "M01": ("S_z0_spatial_zF_stats", "z0", "zF"),
    "M11": ("F_fullLoss_noP_500", "zF", "zF"),
}
ADA = ("ada1", "ada2", "ada3", "ada4", "ada_final")


def atomic_json(path, obj):
    path = Path(path)
    assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_csv(path, rows):
    path = Path(path)
    assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def small_stats(x):
    import torch
    a = x.detach().float()
    assert torch.isfinite(a).all()
    return dict(shape=list(x.shape), dtype=str(x.dtype), mean=float(a.mean()),
                std=float(a.std(unbiased=False)), rms=float(a.square().mean().sqrt()),
                minimum=float(a.min()), maximum=float(a.max()))


def run(task, gpu, verify_only=False):
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    import numpy as np
    import torch
    from adapter35 import frames_for, backend

    backend()
    torch.set_num_threads(2)
    key = tag(task)
    lockpath = HERE / "tasks" / f"{key}.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = HERE / "tasks" / f"{key}.json"
        if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "PASS":
            return
        oldpath = OLD / "tasks" / f"replay_{key}.json"
        old = json.loads(oldpath.read_text())
        assert old["status"] == "PASS" and old["task"] == task
        archive = Path(old["latent_archive"]["path"])
        assert archive.exists() and sha(archive) == old["latent_archive"]["sha256"]
        lat = np.load(archive)
        assert lat["z0"].shape[0] == lat["zF"].shape[0] == 63
        checkpoint = Path(json.loads((OLD / "evaluation_config.json").read_text())["checkpoints"]["N_oursLoss_noP_500"]["path"])
        assert sha(checkpoint) == "ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc"
        eng, rt = runtime_original()
        video = metadata(task)
        frames = frames_for(video)
        im, pm = rt.models(1088, 1920)
        bank = receiver_bank(pm, ["F_fullLoss_noP_500"])
        decoder = bank["F_fullLoss_noP_500"].decoder
        before = eng.core.module_hash(decoder)
        assert before == "f1c4d2c3e2aa424f1989a9cfe5abce381330435e05fd04a8e1367bbcf512ef8c"
        assert all(not p.requires_grad for p in decoder.parameters()) and not decoder.training
        qbank = pm.q_scale_recon
        before_pm = eng.core.module_hash(pm)
        before_im = eng.core.module_hash(im)
        quality = rt.quality
        flow, perceptual, inception = rt.metric_models
        metric_hashes = [eng.core.module_hash(m) for m in (*quality, flow, perceptual, inception)]
        source = sourcepoint(task, "O_original")
        stream = Path(source["bitstream_path"])
        assert sha(stream) == source["bitstream_sha256"]
        assert stream.stat().st_size == int(source["real_bytes"])
        expected_bpp = stream.stat().st_size * 8 / (64 * 1920 * 1080)
        assert abs(expected_bpp - float(source["actual_bpp"])) < 1e-12

        rows = {m: [] for m in METHODS}
        transitions = {m: [] for m in METHODS}
        features = {m: [] for m in METHODS}
        reconstructed = {m: [] for m in METHODS}
        path_rows = []
        gamma_rows = []
        input_rows = []
        endpoint_errors = []
        q_checks = []
        real = []
        previous = {}
        prevgt = None
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for frame, sp, q, bits, realbits in parse(stream, eng.eco):
                assert q == json.loads((OLD / "qp_semantics_audit.json").read_text())["actual_qps"][str(task["qp"])][frame]
                target = frames[frame].cuda()
                if frame == 0:
                    native = im.decompress(bits, sp, q)["x_hat"]
                    output_raw = {m: native for m in METHODS}
                    for m in METHODS:
                        reconstructed[m].append(eng.eco.tensor_sha(native))
                    i0_hash = eng.eco.tensor_sha(native)
                else:
                    z = {name: torch.from_numpy(lat[name][frame-1]).cuda() for name in ("z0", "zF")}
                    q_recon = qbank[q:q+1]
                    proof = old["tensor_equality"][frame-1]["decoded"]["O_original"]["q"]
                    assert th(q_recon) == proof
                    q_checks.append(dict(frame=frame, actual_qp=q, q_hash=proof["sha256"]))
                    output_raw = {}
                    per_method = {}
                    for method, (full, spatial_name, stats_name) in METHODS.items():
                        zsp, zst = z[spatial_name], z[stats_name]
                        assert zsp.shape == zst.shape and zsp.dtype == zst.dtype
                        seen = []
                        linear = {}
                        conv = []
                        gen = []
                        hooks = []
                        def gen_pre(_module, args):
                            gen.append((th(args[0]), th(args[1])))
                        def conv_out(_module, _args, output):
                            conv.append(th(output))
                        hooks.append(decoder.register_forward_pre_hook(gen_pre))
                        hooks.append(decoder.conv_in.register_forward_hook(conv_out))
                        for layer in ADA:
                            ada = getattr(decoder, layer)
                            def pre(_module, args, layer=layer):
                                assert len(args) == 2 and torch.equal(args[1], zsp)
                                seen.append(layer)
                                path_rows.append(dict(dataset=task["dataset"], video=task["video"], QP=task["qp"],
                                                      frame=frame, method=method, method_full=full, layer=layer,
                                                      spatial_source=spatial_name, stats_source=stats_name,
                                                      hidden=small_stats(args[0]), stats_tensor=th(args[1]),
                                                      hidden_fingerprint=small_stats(args[0][:, :1, ::16, ::16])))
                                return (args[0], zst)
                            hooks.append(ada.register_forward_pre_hook(pre))
                            for kind in ("gamma", "beta"):
                                mod = getattr(ada, kind)
                                def linear_hook(_module, args, output, layer=layer, kind=kind):
                                    assert args[0].shape[-1] == 18
                                    linear[(layer, kind)] = (th(args[0]), th(output), small_stats(output))
                                hooks.append(mod.register_forward_hook(linear_hook))
                        try:
                            out = decoder(zsp, q_recon)
                        finally:
                            for h in hooks:
                                h.remove()
                        assert seen == list(ADA) and len(gen) == len(conv) == 1
                        assert gen[0] == (th(zsp), th(q_recon))
                        assert len(linear) == 10
                        input_rows.append(dict(dataset=task["dataset"], video=task["video"], QP=task["qp"],
                                               frame=frame, method=method, spatial=gen[0][0], q=gen[0][1],
                                               stats=th(zst), conv_out=conv[0]))
                        for layer in ADA:
                            for kind in ("gamma", "beta"):
                                ih, oh, st = linear[(layer, kind)]
                                gamma_rows.append(dict(dataset=task["dataset"], video=task["video"], QP=task["qp"],
                                                       frame=frame, method=method, layer=layer, kind=kind,
                                                       input_hash=ih["sha256"], output_hash=oh["sha256"], **st))
                        output_raw[method] = out
                        reconstructed[method].append(eng.eco.tensor_sha(out))
                        per_method[method] = dict(conv=conv[0]["sha256"], linear={k: (v[0]["sha256"], v[1]["sha256"]) for k,v in linear.items()})
                    assert per_method["M00"]["conv"] == per_method["M01"]["conv"]
                    assert per_method["M10"]["conv"] == per_method["M11"]["conv"]
                    for left, right in (("M00", "M10"), ("M01", "M11")):
                        assert per_method[left]["linear"] == per_method[right]["linear"]
                real.append(inception(target).cpu().numpy())
                gtflow = flow(prevgt, target) if prevgt is not None else None
                for method in METHODS:
                    output = eng.core.unit(output_raw[method], 1080, 1920)
                    assert output.shape == target.shape and torch.isfinite(output).all()
                    rows[method].append(dict(frame=frame, actual_qp=q, real_bits=realbits,
                                             **eng.eco.frame_metrics(output, target, quality)))
                    features[method].append(inception(output).cpu().numpy())
                    if frame:
                        fd = gtflow - flow(previous[method], output)
                        val = float(perceptual(prevgt, previous[method], fd, normalize=True).item())
                        assert math.isfinite(val)
                        transitions[method].append(dict(from_frame=frame-1, to_frame=frame, FloLPIPS=val))
                    previous[method] = output
                    if video["video_index"] == json.loads((OLD / "eval_manifest.json").read_text())[task["dataset"]][0]["video_index"] and frame in (16,32,48,63):
                        eng.eco.save_png(output, HERE / "artifacts" / "examples" / key / method / f"{frame}.png")
                if video["video_index"] == json.loads((OLD / "eval_manifest.json").read_text())[task["dataset"]][0]["video_index"] and frame in (16,32,48,63):
                    eng.eco.save_png(target, HERE / "artifacts" / "examples" / key / "GT" / f"{frame}.png")
                prevgt = target
        assert all(len(rows[m]) == 64 and len(transitions[m]) == 63 for m in METHODS)
        assert len(path_rows) == 63 * 4 * 5
        assert before == eng.core.module_hash(decoder)
        assert before_pm == eng.core.module_hash(pm) and before_im == eng.core.module_hash(im)
        assert metric_hashes == [eng.core.module_hash(m) for m in (*quality, flow, perceptual, inception)]
        outputs = {}
        for method in METHODS:
            old_name = METHODS[method][0]
            rawhash = eng.eco.sha256_bytes("".join(reconstructed[method]).encode())
            fp = HERE / "features" / f"{key}_{method}.npz"
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_name(fp.name + f".{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.savez(f, real=np.concatenate(real), reconstruction=np.concatenate(features[method]))
            tmp.replace(fp)
            out = dict(dataset=task["dataset"], video=task["video"], sequence=video["name"], QP=task["qp"],
                       method=method, method_full=old_name, spatial_source=METHODS[method][1], stats_source=METHODS[method][2],
                       actual_bpp=expected_bpp, real_bytes=stream.stat().st_size, frames=64, transitions=63,
                       LPIPS=sum(x["LPIPS"] for x in rows[method])/64,
                       DISTS=sum(x["DISTS"] for x in rows[method])/64,
                       FloLPIPS=sum(x["FloLPIPS"] for x in transitions[method])/63,
                       reconstruction_sha256=rawhash, feature_path=str(fp), feature_sha256=sha(fp),
                       bitstream_path=str(stream), bitstream_sha256=sha(stream), reuse_bitstream=True,
                       independent_decode_source_receipt=str(oldpath), source_receipt_sha256=sha(oldpath),
                       source_rgb_sha256=video["rgb_sha256"], generator_module_hash=before)
            outputs[method] = out
            if method in ("M00", "M11"):
                historical = old["result"][old_name]
                errors = {k: abs(out[k]-historical[k]) for k in ("LPIPS", "DISTS", "FloLPIPS")}
                oldfeat = np.load(historical["feature_path"])
                feature_exact = np.array_equal(np.concatenate(features[method]), oldfeat["reconstruction"])
                eq = dict(method=method, raw_output_bitwise=rawhash == historical["reconstruction_sha256"],
                          feature_bitwise=feature_exact, spatial_metric_errors=errors)
                endpoint_errors.append(eq)
                assert eq["raw_output_bitwise"] and eq["feature_bitwise"] and max(errors.values()) <= 1e-5, eq
            atomic_csv(HERE / "raw" / f"{key}_{method}_frames.csv", rows[method])
            atomic_csv(HERE / "raw" / f"{key}_{method}_transitions.csv", transitions[method])
        atomic_csv(HERE / "raw" / f"{key}_adagn_paths.csv", path_rows)
        atomic_csv(HERE / "raw" / f"{key}_gamma_beta.csv", gamma_rows)
        atomic_json(receipt_path, dict(status="PASS", task=task, results=outputs, endpoint_equivalence=endpoint_errors,
                                       source_replay_receipt=str(oldpath), source_replay_sha256=sha(oldpath),
                                       latent_archive=str(archive), latent_archive_sha256=sha(archive),
                                       source_stream=str(stream), stream_sha256=sha(stream), i0_hash=i0_hash,
                                       actual_qp_and_q_recon=q_checks, adagn_path_checks=len(path_rows),
                                       input_checks=len(input_rows), generator_module_hash_before=before,
                                       generator_module_hash_after=eng.core.module_hash(decoder),
                                       codec_module_hashes_before=[before_im,before_pm],
                                       codec_module_hashes_after=[eng.core.module_hash(im),eng.core.module_hash(pm)],
                                       metric_module_hashes=metric_hashes, new_optimizer_updates=0, new_encodes=0,
                                       GPU_physical=gpu, seconds=time.time()-t0,
                                       peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
        print("PASS", key, round(time.time()-t0, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", type=int, choices=(4,5,6,7), required=True)
    args = ap.parse_args()
    try:
        run(json.loads(args.task), args.gpu)
    except Exception:
        (HERE / "errors.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with (HERE / "errors.jsonl").open("a") as f:
            f.write(json.dumps(dict(task=json.loads(args.task), traceback=traceback.format_exc(), time=time.time()))+"\n")
        raise
