"""Final cross-artifact checks against actual bytes and independent decoded tensors."""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def main():
    import torch
    from evaluation import rgb01
    torch.set_num_threads(4)
    run=HERE/os.environ.get("GVCRT_RUN_NAME","run_v1")
    cfg=json.loads((HERE/"config.json").read_text())
    audit=json.loads((HERE/"stage_a_v2/audit.json").read_text())
    manifest=json.loads((HERE/"manifest.json").read_text())
    status=json.loads((run/"status.json").read_text())
    assert status["stage"]=="D-complete"
    result={"all_checks_passed":False,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE.parents[1],text=True).strip()}
    for tag in ["I","P"]:
        item=audit["weights"][tag]
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()==item["sha256"]
    result["original_checkpoints_unchanged"]=True
    checked=0
    for entry in manifest["train"]+manifest["val"]:
        for frame,digest in enumerate(entry["png_sha256"]):
            path=HERE/"data"/entry["video_id"]/f"{frame:04d}.png"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
            checked+=1
    result["source_frames_hash_checked"]=checked
    base_count=0
    for path in sorted((run/"base").glob("*.bin")):
        metadata=json.loads(path.with_suffix(".json").read_text())
        assert sum(r["base_bits"] for r in metadata["encode"])==path.stat().st_size*8
        assert sum(r["base_bits"] for r in metadata["decode"])==path.stat().st_size*8
        base_count+=1
    result["base_bitstreams_checked"]=base_count
    receiver_count=0
    for path in sorted(run.glob("quantized*/*_receiver.pt")):
        r=torch.load(path,map_location="cpu",weights_only=False)
        assert r["dpb_unchanged"] and r["source_access_guard"]
        assert len(r["frames"])==cfg["frames"] and len(r["symbol_hashes"])==cfg["frames"]
        assert r["symbol_hashes"][0] is None and all(r["symbol_hashes"][1:])
        assert all(x.shape==(1,3,1080,1920) and torch.isfinite(x).all() for x in r["frames"])
        video=path.stem.removesuffix("_receiver")
        base=torch.load(run/"base"/f"{video}_q0.pt",map_location="cpu",weights_only=False)
        assert torch.equal(r["frames"][0],rgb01(base[0]["x_base"]))
        receiver_count+=1
    result["independent_receiver_sequences_checked"]=receiver_count
    assert receiver_count==24
    no_enh=torch.load(HERE/"stage_a_v2/receiver_no_enhancement.pt",map_location="cpu",weights_only=False)
    base=torch.load(run/"base"/f'{manifest["train"][0]["video_id"]}_q0.pt',map_location="cpu",weights_only=False)
    assert no_enh["no_enhancement_bypass"] and no_enh["source_access_guard"]
    assert all(torch.equal(x,rgb01(row["x_base"])) for x,row in zip(no_enh["frames"],base[:8]))
    result["no_stream_receiver_exact_frames"]=len(no_enh["frames"])
    for path in sorted(run.glob("*/summary.json")):
        for row in json.loads(path.read_text()):
            base_path=run/"base"/f'{row["video"]}_q{row["qp"]}.bin'
            assert row["base_bits"]==base_path.stat().st_size*8
            if row["method"].startswith("quantized"):
                assert row["enhancement_bits"]==(path.parent/f'{row["video"]}.gle').stat().st_size*8
            assert row["total_bits"]==row["base_bits"]+row["enhancement_bits"]
            assert abs(row["bpp"]-row["total_bits"]/(16*1080*1920))<1e-12
            assert all(math.isfinite(row[k]) for k in ["psnr","ms_ssim","lpips"])
    result["metric_accounting_checked"]=True
    counts=[]
    for stage in ["stage_b","stage_c","stage_d"]:
        data=json.loads((run/stage/"training_summary.json").read_text())
        assert data["generator_unchanged"]
        counts.append(data["steps"])
    assert counts==[500,1000,1500]
    result["training_steps_per_branch"]=counts
    result["all_checks_passed"]=True
    result["config_sha256"]=hashlib.sha256((HERE/"config.json").read_bytes()).hexdigest()
    result["manifest_sha256"]=hashlib.sha256((HERE/"manifest.json").read_bytes()).hexdigest()
    result["final_source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    with (run/"verification.json").open("x") as f:
        f.write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
