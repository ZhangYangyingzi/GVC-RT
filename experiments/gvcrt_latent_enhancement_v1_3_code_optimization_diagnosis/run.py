"""Q1/Q2 independent staged diagnosis; existing directories/results never overwritten."""
import argparse
import subprocess
import traceback
from bridge import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    parser.add_argument("--phase",choices=["prepare","calibrate","calibrate_B","evaluate","all"],default="all")
    args=parser.parse_args()
    out=HOME/args.output
    if Path(args.output).name!=args.output:
        raise ValueError("Output must be a new basename")
    busy=subprocess.check_output(["nvidia-smi",f'--id={CONFIG["gpu_uuid"]}',"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(v.strip()) for v in busy.split(",")]
    if mem>64 or util:
        raise RuntimeError(f"GPU busy: {busy}")
    setup()
    if args.phase in ["prepare","all"]:
        out.mkdir()
        save_json(out/"config.json",CONFIG)
        save_json(out/"input_audit.json",{"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  "python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_uuid":CONFIG["gpu_uuid"],
                  "historical_inventory":inventory(),"model_path":str(MODEL),"model_sha256":sha(MODEL),
                  "AGENTS":"No applicable AGENTS.md found in repository, parents or experiment paths"})
        (out/"nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"],text=True))
    else:
        assert json.loads((out/"config.json").read_text())==CONFIG
    status={"phase":args.phase,"status":"RUNNING","network_updates":0}
    try:
        bundle=Bundle()
        if args.phase in ["prepare","all"]:
            import prepare
            prepare.run(out,bundle)
        if args.phase in ["calibrate","all"]:
            import stages
            stages.calibrate(out,bundle)
        if args.phase=="calibrate_B":
            import stages
            stages.calibrate(out,bundle,resume_B=True)
        if args.phase in ["evaluate","all"]:
            import stages
            status.update(stages.evaluate(out,bundle))
        bundle.check()
        audit=json.loads((out/"input_audit.json").read_text())
        assert inventory()==audit["historical_inventory"] and sha(MODEL)==audit["model_sha256"]
        status.update(status="COMPLETE",historical_files_unchanged=True)
    except Exception:
        status.update(status="BLOCKED/INCONCLUSIVE",exception=traceback.format_exc())
        raise
    finally:
        save_json(out/f"status_{args.phase}.json",status)
        print("STATUS",status,flush=True)


if __name__=="__main__":
    main()
