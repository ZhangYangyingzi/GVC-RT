import json
from report import ROOT, BRANCHES, read

cfg = json.loads((ROOT/"config.json").read_text())
sel = json.loads((ROOT/"selected_checkpoint.json").read_text())
check = json.loads((ROOT/"final_integrity.json").read_text())
val = sel["validation_metrics"]
final = next(r for r in read(ROOT/"same_qp_summary.csv") if r["qp"] == "all")
v3 = next(r for r in read(ROOT/"v4_vs_v3_final_summary.csv") if r["qp"] == "all")
lines = [f"V4 experiment directory {ROOT}", f"V3 initialization checkpoint {cfg['initial_checkpoint']}"]
for b in BRANCHES:
    lines.append(f"{b} completed {'PASS' if check[b+'_completed'] else 'FAIL'}")
lines += [f"selected beta {sel['stage']} ({sel['beta']})",f"selected step {sel['step']}"]
for prefix, data in (("validation",val),("final test",final)):
    lines += [f"{prefix}: mean rate change vs Original {data['rate_change_percent']}%",
              f"{prefix}: mean LPIPS change vs Original {data['LPIPS_change']}",
              f"{prefix}: mean DISTS change vs Original {data['DISTS_change']}"]
lines += [f"final triple_nonworse count {final['triple_nonworse_count']} / 32",
          f"V4 vs V3: rate change {v3['rate_change_percent']}%",
          f"V4 vs V3: LPIPS change {v3['LPIPS_change']}",f"V4 vs V3: DISTS change {v3['DISTS_change']}",
          f"generator updated {'PASS' if check['generator_hash_changed'] else 'FAIL'}",
          f"compression frozen {'PASS' if check['compression_hash_unchanged'] else 'FAIL'}",
          f"independent decode {'PASS' if check['independent_decode_pass'] else 'FAIL'}",
          f"final_integrity {check['status']}"]
for name in ("checkpoint_selection.csv","same_qp_summary.csv","v4_vs_v3_final_summary.csv","rd_curves"):
    lines.append(f"{name} path {ROOT/name}")
out = "\n".join(lines)+"\n"
(ROOT/"stdout.log").write_text(out)
print(out,end="")
