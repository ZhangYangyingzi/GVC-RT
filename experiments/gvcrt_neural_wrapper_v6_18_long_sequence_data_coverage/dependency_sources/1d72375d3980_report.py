"""Strict validation selection and final audit; CPU only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "gvcrt_neural_wrapper_v3_pb_scale"
BRANCHES = ("beta_low", "beta_mid", "beta_high")
STEPS = (0, 1000, 2000, 5000, 10000)
METRICS = ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "SSIM", "MS_SSIM")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def compare(rows, refs):
    pairs = list(zip(rows, refs))
    return {
        "mean_real_kbps": mean(rows, "kbps"), "mean_bpp": mean(rows, "bpp"),
        "mean_LPIPS": mean(rows, "LPIPS"), "mean_DISTS": mean(rows, "DISTS"),
        "rate_change_percent": 100 * (mean(rows, "kbps") / mean(refs, "kbps") - 1),
        "mean_bitrate_change_percent": sum(100*(float(a["kbps"])/float(b["kbps"])-1) for a,b in pairs)/len(pairs),
        "LPIPS_change": mean(rows, "LPIPS") - mean(refs, "LPIPS"),
        "DISTS_change": mean(rows, "DISTS") - mean(refs, "DISTS"),
        "rate_lower_count": sum(float(a["kbps"]) < float(b["kbps"]) for a,b in pairs),
        "LPIPS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"]) for a,b in pairs),
        "DISTS_nonworse_count": sum(float(a["DISTS"]) <= float(b["DISTS"]) for a,b in pairs),
        "triple_nonworse_count": sum(float(a["kbps"]) < float(b["kbps"]) and
                                    float(a["LPIPS"]) <= float(b["LPIPS"]) and
                                    float(a["DISTS"]) <= float(b["DISTS"]) for a,b in pairs),
        "point_count": len(rows)}


def choose(candidates):
    valid = [r for r in candidates if r["rate_change_percent"] < 0
             and r["LPIPS_change"] <= 0 and r["DISTS_change"] <= 0]
    if not valid:
        raise RuntimeError("no validation candidate meets all three selection gates")
    minimum = min(r["mean_real_kbps"] for r in valid)
    tied = [r for r in valid if r["mean_real_kbps"] / minimum - 1 < .005]
    return min(tied, key=lambda r: (r["mean_LPIPS"], r["mean_DISTS"],
                                    r["mean_real_kbps"], r["branch"], r["step"]))


def check_stream(row):
    p = Path(row["bitstream_path"])
    size = int(row.get("real_bytes", row.get("bytes")))
    assert p.stat().st_size == size and sha(p) == row["bitstream_sha256"], p
    assert int(row["bytes_consumed"]) == size and row["decode_status"] == "PASS", p
    assert str(row["state_sync_pass"]).lower() == "true", p
    assert str(row["independent_decode_pass"]).lower() == "true", p
    assert row["compression_hash_before"] == row["compression_hash_after"], p
    assert all(math.isfinite(float(row[m])) for m in METRICS), p


def select():
    rows = [r for p in sorted((ROOT/"parts").glob("checkpoint_validation_*_gpu*.csv")) for r in read(p)]
    lookup = {}
    for r in rows:
        key = (r["stage"], int(r["step"]), int(r["video_index"]), int(r["qp"]))
        if key in lookup:
            raise RuntimeError(f"duplicate validation key {key}")
        lookup[key] = r
        assert int(r["num_frames"]) == 32
        check_stream(r)
    stages = [(b,s) for b in BRANCHES for s in STEPS] + [("original",0),("v3",20000)]
    expected = {(b,s,v,q) for b,s in stages for v in range(6) for q in range(4)}
    if set(lookup) != expected:
        raise RuntimeError(f"validation incomplete: {len(lookup)}/{len(expected)}")
    cfg = json.loads((ROOT/"config.json").read_text())
    for b,s in stages:
        if b == "original":
            continue
        path = Path(cfg["initial_checkpoint"]) if b == "v3" else ROOT/"checkpoints"/b/f"step_{s:04d}.pt"
        digest = sha(path)
        assert all(lookup[b,s,v,q]["checkpoint_sha256"] == digest for v in range(6) for q in range(4))
    write("checkpoint_validation.csv", rows)
    summaries, vs_v3, candidates = [], [], []
    for b in BRANCHES:
        for s in STEPS:
            for q in [0,1,2,3,"all"]:
                keys = [(v,k) for v in range(6) for k in range(4) if q == "all" or k == q]
                group = [lookup[b,s,v,k] for v,k in keys]
                original = [lookup["original",0,v,k] for v,k in keys]
                v3 = [lookup["v3",20000,v,k] for v,k in keys]
                metadata = {"branch": b, "beta": cfg["branches"][b]["beta"], "step": s, "qp": q}
                row = {**metadata, **compare(group, original)}
                summaries.append(row)
                vs_v3.append({**metadata, **compare(group, v3)})
                if q == "all":
                    eligible = row["rate_change_percent"] < 0 and row["LPIPS_change"] <= 0 and row["DISTS_change"] <= 0
                    candidates.append({**row, "eligible": eligible, "selected": False})
    valid = [r for r in candidates if r["eligible"]]
    if not valid:
        write("checkpoint_selection.csv", candidates)
        raise RuntimeError("no validation candidate meets all three selection gates")
    winner = choose(candidates)
    winner["selected"] = True
    write("validation_summary.csv", summaries)
    write("v4_vs_v3_summary.csv", vs_v3)
    write("checkpoint_selection.csv", candidates)
    dump("selected_checkpoint.json", {
        "stage": winner["branch"], "beta": winner["beta"], "step": winner["step"],
        "checkpoint": str(ROOT/"checkpoints"/winner["branch"]/f"step_{winner['step']:04d}.pt"),
        "checkpoint_sha256": sha(ROOT/"checkpoints"/winner["branch"]/f"step_{winner['step']:04d}.pt"),
        "validation_metrics": winner, "used_final_test": False,
        "selection_rule": "Gate arithmetic mean real kbps < Original, mean LPIPS <= Original, mean DISTS <= Original over 24 validation points. Minimize mean real kbps; candidates within <0.5% of minimum tie by LPIPS, DISTS, then rate. PSNR excluded.",
        "validation_csv_sha256": sha(ROOT/"checkpoint_validation.csv")})
    print(json.dumps({"selected_branch":winner["branch"], "selected_step":winner["step"]}), flush=True)


def audit():
    import torch
    cfg = json.loads((ROOT/"config.json").read_text())
    initial = torch.load(cfg["initial_checkpoint"], map_location="cpu", weights_only=True)
    validation = read(ROOT/"checkpoint_validation.csv")
    lookup = {(r["stage"],int(r["step"]),int(r["video_index"]),int(r["qp"])):r for r in validation}
    init_rows, updates, parameters = [], {}, {}
    for b in BRANCHES:
        cp = torch.load(ROOT/"checkpoints"/b/"step_0000.pt", map_location="cpu", weights_only=True)
        equals = {k: set(cp[k]) == set(initial[k]) and all(torch.equal(cp[k][n], initial[k][n]) for n in initial[k])
                  for k in ("wrapper","bridge","generator")}
        assert cp["step"] == 0 and cp["branch"] == b and cp["optimizer"]["state"] == {}
        for v in range(6):
            for q in range(4):
                a, ref = lookup[b,0,v,q], lookup["v3",20000,v,q]
                bitrate_equal = a["bitstream_sha256"] == ref["bitstream_sha256"] and a["real_bytes"] == ref["real_bytes"]
                reconstruction_equal = (a["reconstruction_sha256"] == ref["reconstruction_sha256"] and
                                        all(float(a[m]) == float(ref[m]) for m in METRICS))
                init_rows.append({"branch":b,"video_index":v,"qp":q,**{k+"_equal":z for k,z in equals.items()},
                                  "bitstream_equal":bitrate_equal,"reconstruction_equal":reconstruction_equal,
                                  "status":"PASS" if all(equals.values()) and bitrate_equal and reconstruction_equal else "FAIL"})
        parameters[b] = json.loads((ROOT/"parts"/f"parameter_audit_initial_{b}.json").read_text())
        done = json.loads((ROOT/"parts"/f"train_{b}_done.json").read_text())
        assert done["status"] == "PASS" and done["steps"] == 10000
        trained = torch.load(ROOT/"checkpoints"/b/"step_10000.pt", map_location="cpu", weights_only=True)
        assert trained["step"] == 10000 and trained["branch"] == b
        assert trained["beta"] == cfg["branches"][b]["beta"]
        for name in ("wrapper", "bridge", "generator"):
            digest = hashlib.sha256()
            for key, tensor in trained[name].items():
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            assert digest.hexdigest() == done["final_hashes"][name], (b,name)
            assert done["initial_hashes"][name] != done["final_hashes"][name], (b,name)
        for group in trained["optimizer"]["param_groups"]:
            assert group["lr"] == cfg["optimizer"][group["name"]+"_lr"]
            assert group["weight_decay"] == cfg["optimizer"]["weight_decay"]
        del trained
        assert done["initial_hashes"]["compression"] == done["final_hashes"]["compression"]
        assert parameters[b]["compression_trainable_parameters"] == 0
        assert all(parameters[b][m+"_trainable_parameters"] > 0 for m in ("wrapper","bridge","generator"))
        assert parameters[b]["optimizer_initialized_independently"]
        logs = read(ROOT/"training_logs"/f"training_{b}.csv")
        assert [int(r["step"]) for r in logs] == list(range(1,10001))
        assert all(math.isfinite(float(r[k])) for r in logs for k in
                   ("loss","LPIPS","DISTS","R_est_bpp","proxy_L1","wrapper_gradient_norm","bridge_gradient_norm","generator_gradient_norm"))
        assert all(int(r["qp"]) in range(4) for r in logs)
        update = json.loads((ROOT/"parts"/f"generator_update_audit_{b}.json").read_text())
        assert update["hash_changed"] and not update["compression_hash_changed"]
        assert update["generator_update_norm"] > 0 and update["generator_gradient_norm_max"] > 0
        updates[b] = update
    assert all(r["status"] == "PASS" for r in init_rows)
    write("initialization_audit.csv", init_rows)
    dump("generator_update_audit.json", updates)
    dump("parameter_audit_initial.json", parameters)


def final():
    selected = json.loads((ROOT/"selected_checkpoint.json").read_text())
    assert selected["used_final_test"] is False
    assert sha(ROOT/"checkpoint_validation.csv") == selected["validation_csv_sha256"]
    audit()
    original = read(V3/"final_rd_points.csv")
    new = [r for p in sorted((ROOT/"parts").glob("final_v4_gpu*.csv")) for r in read(p)]
    digest = sha(selected["checkpoint"])
    assert len(new) == 32 and all(r["checkpoint_sha256"] == digest for r in new)
    methods = ("original","beta_low","v3_beta_low","v4_selected")
    rows = original + new
    manifest = json.loads((ROOT/"test_manifest.json").read_text())["videos"]
    tags = {f"{v['dataset']}_{int(v['video_id']):02d}" for v in manifest}
    keys = {(r["video_tag"],r["method"],int(r["qp"])) for r in rows}
    assert len(rows) == 128 and keys == {(t,m,q) for t in tags for m in methods for q in range(4)}
    for r in rows:
        check_stream(r)
    assert all(Path(r["checkpoint"]).resolve() == Path(json.loads((ROOT/"config.json").read_text())["initial_checkpoint"]).resolve()
               for r in rows if r["method"] == "v3_beta_low")
    write("final_rd_points.csv", rows)
    lookup = {(r["video_tag"],r["method"],int(r["qp"])):r for r in rows}
    for ref, output in (("original","same_qp_summary.csv"),("v3_beta_low","v4_vs_v3_final_summary.csv"),("beta_low","v4_vs_v2_final_summary.csv")):
        summary = []
        for q in [0,1,2,3,"all"]:
            keys = [(t,k) for t in sorted(tags) for k in range(4) if q == "all" or k == q]
            a = [lookup[t,"v4_selected",k] for t,k in keys]
            b = [lookup[t,ref,k] for t,k in keys]
            summary.append({"qp":q,**compare(a,b)})
        write(output, summary)
    write("bitstream_audit.csv", [{"method":r["method"],"video_tag":r["video_tag"],"qp":r["qp"],
          "path":r["bitstream_path"],"sha256":r["bitstream_sha256"],"bytes":r["bytes"],"status":"PASS"} for r in rows])
    write("decode_audit.csv", [{"method":r["method"],"video_tag":r["video_tag"],"qp":r["qp"],
          "bytes_consumed":r["bytes_consumed"],"status":r["decode_status"]} for r in rows])
    rd = [{"method":m,"qp":q,**{f"mean_{k}":mean([r for r in rows if r["method"]==m and int(r["qp"])==q],k) for k in METRICS}}
          for m in methods for q in range(4)]
    write("rd_interpolation_inputs.csv", rd)
    import subprocess
    subprocess.run(["/data1/anaconda3_new/anaconda_program/bin/python", str(ROOT/"plot_results.py")], check=True)
    assert all((ROOT/"rd_curves"/f"rate_{m}.{e}").stat().st_size > 100 for m in ("lpips","dists","psnr","ms_ssim") for e in ("png","pdf"))
    split = json.loads((ROOT/"split_audit.json").read_text())
    assert split["status"] == "PASS"
    snapshot = json.loads((ROOT/"source_snapshot.json").read_text())
    assert all(sha(p) == digest for p,digest in snapshot.items()), "V3 source files changed"
    cfg = json.loads((ROOT/"config.json").read_text())
    assert cfg["clip_length"] == 4 and cfg["online_random_sampling"] and cfg["crop"] == [256,256]
    assert all(int(r["num_frames"]) == 64 for r in rows)
    integrity = {k:True for k in ("initialized_from_v3_step20000","three_beta_branches_independent","clip_length_4",
          "online_random_sampling_used","wrapper_trainable","bridge_trainable","generator_trainable","compression_core_frozen",
          "compression_hash_unchanged","generator_hash_changed","beta_low_completed","beta_mid_completed","beta_high_completed",
          "all_validation_complete","selected_without_final_test","real_rans_final_test_complete","independent_decode_pass",
          "LPIPS_complete","DISTS_complete","rd_curves_created","all_values_finite")}
    integrity.update(train_video_count=256, validation_video_count=6, test_video_count=8, status="PASS")
    dump("final_integrity.json", integrity)
    print("final_integrity PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select","audit","final"))
    args = parser.parse_args()
    {"select":select,"audit":audit,"final":final}[args.action]()
