"""One-command reproduction using fixed manifest; fail closed on changed inputs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or (HERE/args.run_name).exists():
        raise ValueError("Choose a new basename for the run; existing results will not be overwritten")
    cfg = json.loads((HERE/"config.json").read_text())
    if not (HERE/"manifest.json").exists():
        subprocess.run([sys.executable,str(HERE/"prepare_data.py")],check=True)
    manifest = json.loads((HERE/"manifest.json").read_text())
    for entry in manifest["train"]+manifest["val"]:
        for frame, expected in enumerate(entry["png_sha256"]):
            path = HERE/"data"/entry["video_id"]/f"{frame:04d}.png"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f"Input changed: {path}")
    audit_path = HERE/"stage_a_v2/audit.json"
    if not audit_path.exists():
        subprocess.run([sys.executable,str(HERE/"run_audit.py")],check=True)
    audit = json.loads(audit_path.read_text())
    assert audit["status"]=="INTERFACE-PASS"
    for tag in ["I","P"]:
        evidence = audit["weights"][tag]
        if hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest()!=evidence["sha256"]:
            raise RuntimeError("Base checkpoint changed")
    gpu = cfg["gpu_uuid"]
    used = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util = [int(x.strip()) for x in used.split(",")]
    if mem>64 or util>0:
        raise RuntimeError(f"Configured single GPU is busy: {used.strip()}")
    env = dict(os.environ,GVCRT_RUN_NAME=args.run_name,CUDA_VISIBLE_DEVICES=gpu)
    subprocess.run([sys.executable,"-u",str(HERE/"run_experiment.py")],cwd=HERE.parents[1],env=env,check=True)
    subprocess.run([sys.executable,"-u",str(HERE/"supplemental.py")],cwd=HERE.parents[1],env=env,check=True)
    if (HERE/args.run_name/"quantized_d_val6_l2/summary.json").exists():
        subprocess.run([sys.executable,str(HERE/"review_val.py")],cwd=HERE.parents[1],env=env,check=True)
    for stage in ["stage_d","stage_c"]:
        checkpoint=HERE/args.run_name/stage/"checkpoint.pt"
        if checkpoint.exists():
            subprocess.run([sys.executable,str(HERE/"benchmark_e2e.py"),"--checkpoint",str(checkpoint),
                            "--output",f"{args.run_name}/e2e","--level","2"],cwd=HERE.parents[1],env=env,check=True)
            break
    subprocess.run([sys.executable,str(HERE/"summarize.py"),"--run-name",args.run_name],cwd=HERE.parents[1],env=env,check=True)
    subprocess.run([sys.executable,str(HERE/"analyze_streams.py"),"--run-name",args.run_name],cwd=HERE.parents[1],env=env,check=True)


if __name__=="__main__":
    main()
