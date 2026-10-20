"""Independent final count/hash/finite audit for fixed v6.39 outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    with (HERE/path).open(newline="") as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def main():
    p=HERE/"final_integrity.json"
    x=json.loads(p.read_text())
    rd=read("results/all_rd_points.csv")
    fid=read("results/pooled_fid.csv")
    effects=read("results/two_by_two_effects.csv")
    interactions=read("results/interaction_terms.csv")
    frame=read("results/per_frame_quality.csv")
    checks={}
    checks["39_verified_RANS_streams"]=x["reused_streams"]==39
    checks["156_RD_points"]=len(rd)==156 and len({(r["dataset"],r["video"],r["QP"],r["method"]) for r in rd})==156
    checks["36_pooled_FID_points"]=len(fid)==36 and len({(r["dataset"],r["method"],r["QP"]) for r in fid})==36
    checks["9984_per_frame_rows"]=len(frame)==156*64
    checks["49140_AdaGN_paths"]=x["AdaGN_path_checks"]==39*63*4*5
    checks["same_G_and_q"]=all(r["generator_module_hash"]=="f1c4d2c3e2aa424f1989a9cfe5abce381330435e05fd04a8e1367bbcf512ef8c" for r in rd)
    checks["zero_updates_and_encodes"]=x["training_updates"]==x["new_real_encodes"]==0 and all(int(r["new_optimizer_updates"])==int(r["new_real_encodes"])==0 for r in rd)
    checks["all_complete_RANS_bpp"]=all(abs(float(r["actual_bpp"])-int(r["real_bytes"])*8/(64*1920*1080))<1e-14 for r in rd)
    checks["one_stream_per_video_QP"]=all(len({r["bitstream_sha256"] for r in rd if (r["dataset"],r["video"],r["QP"])==k})==1 for k in {(r["dataset"],r["video"],r["QP"]) for r in rd})
    checks["finite_quality"]=all(math.isfinite(float(r[m])) for r in rd for m in ("LPIPS","DISTS","FloLPIPS")) and all(math.isfinite(float(r[m])) for r in fid for m in ("FID","D_mu","D_cov"))
    checks["FID_closure"]=all(abs(float(r["FID"])-float(r["D_mu"])-float(r["D_cov"]))<1e-7 for r in fid)
    checks["FID_sample_counts"]=all(int(r["samples"])=={"ulong":384,"uvg_holdout":128,"hevc_b":320}[r["dataset"]] for r in fid)
    checks["endpoint_bitwise"]=x["endpoint_bitwise"]
    checks["all_effects_and_interactions"]=len(effects)>0 and len(interactions)>0
    checks["frozen_modules"]=x["frozen_modules_pass"]
    source=json.loads((HERE/"source_manifest.json").read_text())
    checks["dependency_hashes"]=all(Path(path).exists() and sha(path)==value["sha256"] for path,value in source["dependencies"].items())
    checks["required_outputs"]=all((HERE/path).exists() for path in ("REPORT.md","RESULTS_SUMMARY.md","INTERVENTION_PROTOCOL.md","METHOD_MAPPING.md","audits/endpoint_equivalence.json","audits/adagn_conditioning_paths.json","results/equal_rate_summary.csv","FIGURE_INDEX.md"))
    x["checks"]=checks
    x["status"]="PASS" if all(checks.values()) else "FAIL"
    tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,indent=2)+"\n");tmp.replace(p)
    print("VERIFY",x["status"],sum(checks.values()),len(checks),flush=True)
    if x["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
