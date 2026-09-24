#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V13=ROOT.parent/"gvcrt_v13_generator_aware_mixed_precision"
sys.path.insert(0,str(ROOT))
from run_v13_1 import CFG,MAN,tag,read_csv,write_csv


def combine(suffix,out):
    rows=[]
    for v in MAN["videos"]:
        for qp in CFG["requested_qps"]:
            path=ROOT/"parts"/f"{tag(v,qp)}_{suffix}.csv"
            if path.exists():rows.extend(read_csv(path))
    write_csv(ROOT/out,rows);return rows


def finite_complete(rows,key):
    try:return bool(rows) and all(math.isfinite(float(r[key])) for r in rows)
    except (ValueError,TypeError,KeyError):return False


def main():
    expected=[(v,q) for v in MAN["videos"] for q in CFG["requested_qps"]]
    done=[];failed=[]
    for v,q in expected:
        d=ROOT/"parts"/f"{tag(v,q)}_done.json";f=ROOT/"parts"/f"{tag(v,q)}_FAILED.json"
        if d.exists() and json.loads(d.read_text()).get("status")=="PASS":done.append(d)
        elif f.exists():failed.append(str(f))
        else:failed.append(f"missing:{tag(v,q)}")
    baseline=combine("baseline_cells","baseline_cells.csv")
    maps=combine("frame_candidate_maps","frame_candidate_maps.csv");modeaudit=combine("mode_map_audit","mode_map_audit.csv")
    beam=combine("beam_states","beam_states.csv");steps=combine("trajectory_steps","trajectory_steps.csv")
    traj=combine("final_trajectories","final_trajectories.csv");final=combine("final_streams","final_streams.csv")
    uniform=combine("uniform_controls","uniform_controls.csv");randoms=combine("random_controls","random_controls.csv")
    frames=combine("frame_metrics","frame_metrics.csv");seq=combine("sequence_metrics","sequence_metrics.csv")
    audits=combine("bitstream_audit","bitstream_audit.csv");coverage=combine("search_coverage","search_coverage.csv")
    multi=[]
    source_hash=hashlib.sha256((V13/"multi_qp_curve.csv").read_bytes()).hexdigest()
    for r in read_csv(V13/"multi_qp_curve.csv"):
        r=dict(r);r["source_path"]=str(V13/"multi_qp_curve.csv");r["source_sha256"]=source_hash;multi.append(r)
    write_csv(ROOT/"multi_qp_curve.csv",multi)
    logs=[]
    for p in sorted((ROOT/"logs").glob("*.log")):logs.append(f"===== {p.name} =====\n"+p.read_text(errors="replace"))
    (ROOT/"run.log").write_text("\n".join(logs))
    manifest_hash=hashlib.sha256((ROOT/"manifest.json").read_bytes()).hexdigest()
    hashes=[hashlib.sha256((x/"manifest.json").read_bytes()).hexdigest() for x in [V13,ROOT.parent/"gvcrt_v13_debug_mixed_precision",ROOT.parent/"gvcrt_v12_3_discrete_symbol_oracle",ROOT.parent/"gvcrt_v12_4_sequence_discrete_beam"]]
    selected=[r for r in final if r.get("selection_type")=="MIXED"]
    integrity={"all_12_cells_completed":len(done)==12,"completed_cells":len(done),"all_manifest_hashes_match":all(x==manifest_hash for x in hashes),
      "all_baseline_identity_pass":len(baseline)==12 and all(str(r["identity_gate_pass"]).lower()=="true" for r in baseline),
      "model_hash_unchanged":len(baseline)==12 and all(r["model_hash_before"]==r["model_hash_after"] for r in baseline),
      "training_false":CFG["training"] is False,"selected_stream_count":len(selected),
      "all_selected_decode_pass":all(str(r["decode_audit_pass"]).lower()=="true" for r in selected),
      "independent_decode_pass_count":sum(str(r.get("decode_audit_pass")).lower()=="true" for r in audits),
      "G4_completed":sum(r["granularity"]=="G4" for r in coverage),"G8_completed":sum(r["granularity"]=="G8" for r in coverage),
      "G16_completed":sum(r["granularity"]=="G16" for r in coverage),"uniform_controls_completed":len(uniform)==24,
      "random_controls_completed":len(randoms)==36,"multi_qp_curve_available":len(multi)==24,
      "lpips_complete":finite_complete(baseline+uniform+randoms+[r for r in final if r.get("selection_type")=="MIXED"],"LPIPS"),
      "dists_complete":finite_complete(baseline+uniform+randoms+[r for r in final if r.get("selection_type")=="MIXED"],"DISTS"),
      "failed_cells":failed}
    integrity["status"]="PASS" if all([integrity["all_12_cells_completed"],integrity["all_manifest_hashes_match"],integrity["all_baseline_identity_pass"],
      integrity["model_hash_unchanged"],integrity["training_false"],integrity["all_selected_decode_pass"],integrity["G4_completed"]==12,
      integrity["G8_completed"]==12,integrity["G16_completed"]==12,integrity["uniform_controls_completed"],integrity["random_controls_completed"],
      integrity["multi_qp_curve_available"],integrity["lpips_complete"],integrity["dists_complete"]]) else "FAIL"
    (ROOT/"final_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    runtime={"python":sys.version,"platform":platform.platform(),"git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,text=True,capture_output=True).stdout.strip(),
      "manifest_sha256":manifest_hash,"random_seed":CFG["random_seed"],"checkpoints":[]}
    for p in sorted((REPO/"checkpoints").glob("*.pt")):runtime["checkpoints"].append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (ROOT/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    artifacts=[]
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name!="artifact_manifest.json":artifacts.append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (ROOT/"artifact_manifest.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    print(str(ROOT));print("12/12 cells complete:",len(done)==12);print("baseline identity pass:",sum(str(r.get("identity_gate_pass")).lower()=="true" for r in baseline),"/12")
    print("G4/G8/G16 completed:",integrity["G4_completed"],integrity["G8_completed"],integrity["G16_completed"])
    print("selected mixed streams:",len(selected));print("independent decode PASS:",integrity["independent_decode_pass_count"],"/",len(audits))
    print("uniform controls complete:",integrity["uniform_controls_completed"]);print("random controls complete:",integrity["random_controls_completed"])
    print("multi-QP curve complete:",integrity["multi_qp_curve_available"])
    print("outputs: baseline_cells.csv, frame_candidate_maps.csv, mode_map_audit.csv, beam_states.csv, trajectory_steps.csv, final_trajectories.csv, final_streams.csv, uniform_controls.csv, random_controls.csv, multi_qp_curve.csv, frame_metrics.csv, sequence_metrics.csv, bitstream_audit.csv, search_coverage.csv, final_integrity.json, run.log, bitstreams/")


if __name__=="__main__":main()
