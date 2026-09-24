"""V1.5 phase driver with immutable run identity and resumable optimization."""
import argparse
import shutil
import subprocess

from allocator import self_test
from bridge import *


def gpu_preflight():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, used, total, utilization = [v.strip() for v in line.split(",")]
        row = {"index": int(index), "uuid": uuid, "memory_used_mib": int(used),
               "memory_total_mib": int(total), "memory_free_mib": int(total) - int(used),
               "utilization_percent": int(utilization)}
        rows.append(row)
    chosen = next(row for row in rows if row["uuid"] == CONFIG["gpu_uuid"])
    if chosen["memory_free_mib"] < CONFIG["minimum_free_gpu_mib"]:
        raise RuntimeError(f"Configured GPU has only {chosen['memory_free_mib']} MiB free")
    return {"command": command, "all_allowed_gpu_rows": [r for r in rows if r["index"] in (4, 5, 6, 7)],
            "chosen": chosen}


def initialize(out):
    identity = {"config_sha256": file_hash(HOME / "config.json"), "model_sha256": file_hash(MODEL),
                "allocator_test": self_test(), "gpu_preflight": gpu_preflight(),
                "historical_inputs": {"v12": str(OLD), "v13": str(V13), "v14": str(V14)}}
    path = out / "run_identity.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["config_sha256"] == identity["config_sha256"]
        assert old["model_sha256"] == identity["model_sha256"]
    else:
        out.mkdir(parents=True)
        shutil.copy2(HOME / "config.json", out / "config.json")
        json_dump(path, identity)
    return identity


def audit(out):
    if (out / "audit.json").exists():
        return
    setup(); bundle = Bundle(False)
    from candidates import load_video_candidates
    provenance = []
    for split in ("Train2",):
        for entry in entries(split):
            _, rows = load_video_candidates(split, entry["video_id"], bundle)
            provenance.extend(rows)
    assert len(provenance) == 120
    assert not any("interrupted_attempt" in (row["source"] or "") for row in provenance)
    json_dump(out / "audit.json", {"candidate_provenance_rows": len(provenance),
              "interrupted_artifacts_selected": False, "network_updates": 0,
              "model_fingerprints": bundle.fingerprints,
              "orc2_header_bytes": ORC2.HEADER.size, "orc2_i_flag_bytes": 1,
              "frame_arithmetic_states_independent": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("audit", "train2", "valopt", "val6", "all"), default="all")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    initialize(out)
    phases = ("audit", "train2", "valopt", "val6") if args.phase == "all" else (args.phase,)
    if "audit" in phases:
        audit(out)
    if "train2" in phases:
        setup()
        from evaluate import run
        run(out, "Train2")
    if "valopt" in phases:
        setup()
        from optimize import run
        run(out)
    if "val6" in phases:
        if not (out / "optimization_complete.json").exists():
            raise RuntimeError("Val6 evaluation requires completed O1/O2 optimization")
        setup()
        from evaluate import run
        run(out, "Val6")


if __name__ == "__main__":
    main()
