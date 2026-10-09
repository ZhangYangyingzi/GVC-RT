"""Save the independent step-zero tensor and optimizer verification. CPU only."""
import json
import torch
from report import ROOT, BRANCHES, dump, sha

cfg = json.loads((ROOT / "config.json").read_text())
v3 = torch.load(cfg["initial_checkpoint"], map_location="cpu", weights_only=True)
expected_digest = sha(cfg["initial_checkpoint"])
results, audits = {}, {}
for b in BRANCHES:
    cp = torch.load(ROOT/"checkpoints"/b/"step_0000.pt", map_location="cpu", weights_only=True)
    equal = {k: set(cp[k]) == set(v3[k]) and all(torch.equal(cp[k][n], v3[k][n]) for n in v3[k])
             for k in ("wrapper", "bridge", "generator")}
    independent = cp["optimizer"]["state"] == {}
    assert cp["step"] == 0 and cp["branch"] == b and all(equal.values()) and independent
    audit = json.loads((ROOT/"parts"/f"parameter_audit_initial_{b}.json").read_text())
    assert audit["initial_checkpoint_sha256"] == expected_digest
    assert audit["compression_trainable_parameters"] == 0 and audit["generator_trainable_parameters"] > 0
    audits[b] = audit
    results[b] = {"state_equal_to_v3": equal, "optimizer_empty_at_step0": independent,
                  "step0_checkpoint_sha256": sha(ROOT/"checkpoints"/b/"step_0000.pt"), "status": "PASS"}
dump("initialization_weights_audit.json", results)
dump("parameter_audit_initial.json", audits)
print(json.dumps(results))
