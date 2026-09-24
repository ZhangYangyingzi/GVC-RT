"""Freeze protocol/sample mappings and inventory before any new model experiment."""
import json
import platform
import subprocess
import sys
from common import HERE,V1,ROOT,CFG,sha,save_json,models,torch


def inventory():
    paths=subprocess.check_output(["rg","--files","--hidden","--no-ignore",str(V1)],text=True).splitlines()
    return {str(p):{"size":__import__("os").stat(p).st_size,"mtime_ns":__import__("os").stat(p).st_mtime_ns} for p in paths}


def main():
    manifest=json.loads((V1/"manifest.json").read_text())
    audit=json.loads((V1/"run_v1/verification.json").read_text())
    assert audit["all_checks_passed"] and sha(V1/"manifest.json")==audit["manifest_sha256"]
    selected={"train2":[v for v in manifest["train"] if v["video_id"] in manifest["train2"]],"val6":manifest["val"]}
    mappings=[]
    for i,entry in enumerate(selected["val6"]):
        donor=selected["val6"][(i+1)%len(selected["val6"])]
        assert donor["video_id"]!=entry["video_id"]
        mappings.append({"target_video":entry["video_id"],"donor_video":donor["video_id"],
                         "frame_pairs":[{"target":f,"donor":f} for f in range(1,entry["frames"])],
                         "levels":[0,1,2],"rule":"fixed cyclic successor in V1 manifest order, same P-frame ordinal"})
    samples=[]
    for entry in selected["train2"]:
        metadata=json.loads((V1/"run_v1/base"/f'{entry["video_id"]}_q0.json').read_text())
        p=[r for r in metadata["decode"] if r["type"]=="P"]
        chosen=[p[0],p[len(p)//2],p[-1]]
        assert [r["frame"] for r in chosen]==CFG["oracle"]["expected_frame_indices"]
        for row in chosen:
            samples.append({"sample":f'{entry["video_id"]}_f{row["frame"]:04d}',"video":entry["video_id"],
                            "frame":row["frame"],"qp":row["qp"],"pts_seconds":entry["pts_seconds"][row["frame"]],
                            "ell_sha256":row["ell_sha256"],"q_recon_sha256":row["q_recon_sha256"],
                            "reference_state":row["dpb"],"zero_checkpoint":CFG["oracle"]["checkpoint"],
                            "zero_delta":2.0,"crop_xyxy":CFG["oracle"]["crop_boxes_xyxy"][entry["video_id"]]})
    checkpoints={}
    for key in ["ablation","oracle"]:
        path=CFG[key]["checkpoint"]
        e,c,entropy,ck=models(path,device="cpu")
        assert ck["base_qp"]==0 and ck["manifest_sha256"]==sha(V1/"manifest.json")
        assert ck["config"]["training"]["lambda_p"]==CFG["oracle"]["lambda_p"]
        checkpoints[key]={"path":str(V1/path),"sha256":sha(V1/path),"stage":ck["stage"],"steps_in_stage":ck["steps"],
                          "enhancement_keys":len(e.state_dict()),"base_only_keys":len(c.state_dict()),
                          "entropy_keys":len(entropy.state_dict()),"strict_match":True}
    used=[V1/"config.json",V1/"manifest.json",V1/"report.md",V1/"enhancement.py",V1/"evaluation.py",V1/"codec.py",
          ROOT/"checkpoints/GVC-RT_I.pt",ROOT/"checkpoints/GVC-RT_P.pt"]
    used += [V1/s["checkpoint"] for s in [CFG["ablation"],CFG["oracle"]]]
    for entry in selected["val6"]:
        used += [V1/"run_v1/base"/f'{entry["video_id"]}_q0.bin']
        used += [V1/f'run_v1/quantized_d_val6_l{level}'/f'{entry["video_id"]}.gle' for level in range(3)]
    info={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "environment":{"python":sys.version,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"platform":platform.platform()},
          "checkpoints":checkpoints,"used_files_sha256":{str(p):sha(p) for p in used},
          "AGENTS":"No applicable AGENTS.md found at repository/ancestors or nested experiment paths",
          "v1_inventory":inventory()}
    save_json(HERE/"fixed_samples.json",samples)
    save_json(HERE/"donor_mapping.json",mappings)
    save_json(HERE/"manifest.json",selected)
    save_json(HERE/"input_audit.json",info)
    with (HERE/"nvidia_smi_before.txt").open("x") as f:
        f.write(subprocess.check_output(["nvidia-smi"],text=True))
    print(json.dumps({"checkpoints":checkpoints,"oracle_samples":samples,"donors":mappings},indent=2))


if __name__=="__main__":
    main()
