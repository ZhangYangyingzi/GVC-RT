"""Resume V4 training, run validation, select, then evaluate the untouched test split."""
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = "/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python"
BRANCHES = {"beta_low":4, "beta_mid":5, "beta_high":6}


def status(name, **values):
    data = {"state":name,"updated_unix":time.time(),**values}
    temp = ROOT/"pipeline_status.tmp"
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(ROOT/"pipeline_status.json")
    print(json.dumps(data), flush=True)


def run(script, args=(), gpu=None, log=None):
    command = [PYTHON,"-B",str(ROOT/script),*map(str,args)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else ""
    if gpu is not None and gpu not in (4,5,6,7):
        raise RuntimeError("GPU outside authorized allocation")
    log = ROOT/"logs"/(log or (Path(script).stem+".log"))
    with open(log,"a") as f:
        f.write("\nCOMMAND "+" ".join(command)+"\n"); f.flush()
        subprocess.run(command, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def training_active(branch):
    with open(ROOT/"parts"/f"train_{branch}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def branch_worker(branch,gpu):
    done = ROOT/"parts"/f"train_{branch}_done.json"
    while training_active(branch):
        time.sleep(20)
    if not done.exists():
        run("train.py", ("--gpu",gpu,"--branch",branch), gpu, f"train_{branch}.log")
    result = json.loads(done.read_text())
    assert result["status"] == "PASS" and result["steps"] == 10000
    print(f"TRAINING COMPLETE {branch}; starting validation on GPU{gpu}",flush=True)
    run("validate.py", ("--gpu",gpu,"--stage",branch), gpu, f"validate_{branch}.log")


def baselines():
    for stage in ("original","v3"):
        run("validate.py", ("--gpu",7,"--stage",stage), 7, f"validate_{stage}.log")


def main():
    with open(ROOT/"pipeline.lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT/"final_integrity.json").exists():
            result = json.loads((ROOT/"final_integrity.json").read_text())
            if result.get("status") == "PASS":
                print("ALREADY COMPLETE",flush=True)
                return
        try:
            run("prepare_audit.py")
            status("TRAINING_AND_VALIDATION", allocation={**BRANCHES,"original_v3_validation":7})
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(branch_worker,b,g) for b,g in BRANCHES.items()]
                futures.append(pool.submit(baselines))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            status("VALIDATION_SELECTION")
            run("report.py", ("select",), log="selection.log")
            run("report.py", ("audit",), log="initialization_audit.log")
            status("FINAL_TEST")
            test = json.loads((ROOT/"test_manifest.json").read_text())["videos"]
            tags = [f"{v['dataset']}_{int(v['video_id']):02d}" for v in test]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run,"final_evaluate.py",("--gpu",gpu,"--tags",",".join(tags[i::4])),gpu,f"final_gpu{gpu}.log") for i,gpu in enumerate((4,5,6,7))]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            status("FINAL_AUDIT")
            run("report.py", ("final",), log="finalize.log")
            run("print_status.py", log="final_status.log")
            status("PASS")
        except Exception as exc:
            status("FAILED", error=str(exc),traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
