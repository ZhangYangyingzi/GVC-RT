"""Single-GPU, bounded V1.1 diagnosis runner. Never trains a neural network."""
import argparse
import json
import os
import subprocess
import traceback
from common import HERE,V1,CFG,sha,save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    args=parser.parse_args()
    if __import__("pathlib").Path(args.output).name!=args.output:
        raise ValueError("Output must be a new basename")
    out=HERE/args.output
    if out.exists():
        raise FileExistsError(out)
    busy=subprocess.check_output(["nvidia-smi",f'--id={CFG["gpu_uuid"]}',"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(s.strip()) for s in busy.split(",")]
    if mem>64 or util:
        raise RuntimeError(f"Configured GPU is not idle: {busy}")
    audit=json.loads((HERE/"input_audit.json").read_text())
    for path,digest in audit["used_files_sha256"].items():
        assert sha(path)==digest,f"V1 input changed: {path}"
    out.mkdir()
    save_json(out/"fixed_config.json",CFG)
    status={"status":"RUNNING","completed":[],"network_training":False}
    try:
        import rate_analysis
        rate_analysis.run(out)
        status["completed"].append("rate_breakdown")
        import ablation
        ablation.run(out)
        status["completed"].append("source_ablation")
        import oracle
        oracle.run(out)
        status["completed"].append("latent_oracle")
        from prepare import inventory
        after=inventory()
        assert after==audit["v1_inventory"],"V1 files were modified"
        for path,digest in audit["used_files_sha256"].items():
            assert sha(path)==digest
        status.update(status="COMPLETE",v1_inventory_unchanged=True,used_input_hashes_unchanged=True)
    except Exception:
        status.update(status="BLOCKED/INCONCLUSIVE",exception=traceback.format_exc())
        raise
    finally:
        save_json(out/"status.json",status)
        print("STATUS",status,flush=True)


if __name__=="__main__":
    main()
