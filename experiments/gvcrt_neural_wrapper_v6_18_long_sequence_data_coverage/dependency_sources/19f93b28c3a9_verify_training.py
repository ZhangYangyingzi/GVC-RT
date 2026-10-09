#!/usr/bin/env python3
import json
import math
from collections import Counter

from common import ROOT, config, sha256, torch_load, write_json
from methods import METHOD_DIRS


EXPECTED_PARAMETERS = {
    "m1": (801124, 801124),
    "m2": (19237, 820361),
    "m3": (19545, 19545),
    "m4": (19474, 19474),
    "m5": (19299, 19299),
}


def main():
    cfg = config()
    plan_path = ROOT / "analysis/training_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256(plan_path)
    plan_qps = Counter(int(row["requested_qp"]) for row in plan["updates"])
    plan_checks = {
        "updates_exact": len(plan["updates"]) == cfg["training_updates"],
        "steps_exact": [row["step"] for row in plan["updates"]] == list(range(1, 81)),
        "qp_balance_exact": plan_qps == Counter({qp: 20 for qp in cfg["qps"]}),
    }
    records = []
    for method, directory in METHOD_DIRS.items():
        expected_trainable, expected_stored = EXPECTED_PARAMETERS[method]
        for lr in cfg["learning_rates"]:
            lr_dir = ROOT / "checkpoints" / directory / f"lr_{lr:.0e}"
            for update in cfg["checkpoint_updates"]:
                path = lr_dir / f"update_{update:03d}.pt"
                checks = {"exists": path.exists()}
                if path.exists():
                    payload = torch_load(path, map_location="cpu", weights_only=False)
                    history = payload.get("history", [])
                    qp_counts = Counter(int(row["requested_qp"]) for row in history)
                    expected_prefix = plan["updates"][:update]
                    checks.update({
                        "method_exact": payload.get("method") == method,
                        "lr_exact": payload.get("lr") == lr,
                        "update_exact": payload.get("update") == update,
                        "history_exact": (len(history) == update and
                                          [row["update"] for row in history] == list(range(1, update + 1))),
                        "history_finite": all(
                            math.isfinite(float(value)) for row in history for key, value in row.items()
                            if key not in ("update", "video_index", "requested_qp")),
                        "training_trajectory_exact": all(
                            int(actual["update"]) == int(expected["step"]) and
                            int(actual["video_index"]) == int(expected["video_index"]) and
                            int(actual["requested_qp"]) == int(expected["requested_qp"])
                            for actual, expected in zip(history, expected_prefix)),
                        "final_qp_balance_exact": (update != cfg["training_updates"] or
                                                   qp_counts == Counter(
                                                       {qp: 20 for qp in cfg["qps"]})),
                        "trainable_parameters_exact": payload.get("trainable_parameters") == expected_trainable,
                        "stored_parameters_exact": payload.get("stored_parameters") == expected_stored,
                        "training_plan_hash_exact": payload.get("training_plan_sha256") == plan_hash,
                        "loss_weights_exact": payload.get("loss_weights") == cfg["loss"],
                    })
                    if update == cfg["training_updates"]:
                        log_path = ROOT / "logs" / f"train_{method}_lr_{lr:.0e}.json"
                        log = json.loads(log_path.read_text()) if log_path.exists() else {}
                        checks.update({
                            "formal_log_exists": log_path.exists(),
                            "formal_log_identity_exact": (log.get("method") == method and
                                                          log.get("lr") == lr and
                                                          log.get("updates") == update and
                                                          log.get("smoke") is False),
                            "allowed_physical_gpu": log.get("gpu") in cfg["allowed_physical_gpus"],
                            "formal_log_history_matches": log.get("history") == history,
                        })
                records.append({"method": method, "lr": lr, "update": update,
                                "path": str(path), "sha256": sha256(path) if path.exists() else None,
                                "checks": checks,
                                "status": "PASS" if all(checks.values()) else "FAIL"})
    payload = {"plan_checks": plan_checks, "expected_runs": 15, "expected_checkpoints": 30,
               "records": records,
               "status": ("PASS" if all(plan_checks.values()) and len(records) == 30 and
                          all(row["status"] == "PASS" for row in records) else "FAIL")}
    write_json(ROOT / "analysis/training_integrity.json", payload)
    print(json.dumps({"status": payload["status"], "checkpoints": len(records)}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
