"""Bounded V1.4 stages; output directories are never overwritten."""
import argparse
import shutil
import subprocess
import traceback

from bridge import *


def gpu_preflight():
    line = subprocess.check_output([
        "nvidia-smi", f"--id={CONFIG['gpu_uuid']}",
        "--query-gpu=uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip()
    uuid, used, total, utilization = [v.strip() for v in line.split(",")]
    free = int(total) - int(used)
    if uuid != CONFIG["gpu_uuid"] or free < CONFIG["minimum_free_gpu_mib"]:
        raise RuntimeError(f"Selected GPU lacks required free memory: {line}")
    return {"uuid": uuid, "memory_used_mib": int(used), "memory_total_mib": int(total),
            "memory_free_mib": free, "utilization_percent": int(utilization),
            "coexistence_allowed": True}


def initialize(out):
    out.mkdir()
    shutil.copy2(HOME / "config.json", out / "config.json")
    shutil.copy2(V13 / "results_v1/manifest.json", out / "manifest.json")
    shutil.copy2(V13 / "results_v1/samples.json", out / "samples.json")
    inputs = [MODEL, V13 / "results_v1/B_lambda_calibration.json", V13 / "results_v1/B_frozen_rule.json",
              V13 / "results_v1/optimization_scale.json", V13 / "results_v1/input_audit.json"]
    for entry in entries("Train2"):
        video = entry["video_id"]
        inputs.extend([base_path(video), OLD / "results_v1/train2_full_CODED_l3" / f"{video}.orc"])
    json_dump(out / "input_audit.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_preflight": gpu_preflight(), "AGENTS": "No applicable AGENTS.md found",
        "input_sha256": {str(path): file_hash(path) for path in inputs},
        "historical_invalid_attempt_policy": "V1.3 B6_lambda0 invalid domain attempt excluded from every candidate and statistic",
        "network_updates": 0})


def verify_inputs(out):
    audit = json.loads((out / "input_audit.json").read_text())
    assert all(file_hash(Path(path)) == digest for path, digest in audit["input_sha256"].items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results_v1")
    p.add_argument("--phase", choices=["diagnose", "resume_diagnose", "optimize", "final", "all"], default="all")
    args = p.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("Output must be a basename")
    out = HOME / args.output
    if args.phase in ["diagnose", "all"]:
        if out.exists():
            raise FileExistsError(out)
        initialize(out)
    else:
        assert json.loads((out / "config.json").read_text()) == CONFIG
        gpu_preflight()
    status = {"phase": args.phase, "status": "RUNNING", "network_updates": 0}
    try:
        setup()
        if args.phase in ["diagnose", "resume_diagnose", "all"]:
            from diagnose import run
            bundle = Bundle(perceptual=False)
            run(out, bundle)
            bundle.check()
        if args.phase in ["optimize", "all"]:
            from optimize_midpoints import run
            run(out)
        if args.phase in ["final", "all"]:
            from finalize import run
            run(out)
            subprocess.run([sys.executable, str(HOME / "final_checks.py"), "--output", args.output], check=True)
        verify_inputs(out)
        status.update(status="COMPLETE", historical_inputs_unchanged=True)
    except Exception:
        status.update(status="BLOCKED/INCONCLUSIVE", exception=traceback.format_exc())
        raise
    finally:
        path = out / f"status_{args.phase}.json"
        if path.exists():
            attempt = 1
            while (out / f"status_{args.phase}_attempt{attempt}.json").exists():
                attempt += 1
            path.rename(out / f"status_{args.phase}_attempt{attempt}.json")
        with path.open("w") as f:
            json.dump(status, f, indent=2)
            f.write("\n")
        print("STATUS", status, flush=True)


if __name__ == "__main__":
    main()
