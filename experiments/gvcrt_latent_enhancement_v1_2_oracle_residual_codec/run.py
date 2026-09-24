"""Bounded stage runner; --phase supports progressing without overwriting earlier artifacts."""
import argparse
import platform
import subprocess
import traceback
from support import *


def inventory():
    result={}
    for root in [V1,V11]:
        for path in subprocess.check_output(["rg","--files","--hidden","--no-ignore",str(root)],text=True).splitlines():
            s=os.stat(path)
            result[path]={"size":s.st_size,"mtime_ns":s.st_mtime_ns}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    parser.add_argument("--phase",choices=["prepare","direct","train","all"],default="all")
    args=parser.parse_args()
    out=HERE/args.output
    if Path(args.output).name!=args.output:
        raise ValueError("Use an output basename")
    busy=subprocess.check_output(["nvidia-smi",f'--id={CFG["gpu_uuid"]}',"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(v.strip()) for v in busy.split(",")]
    if mem>64 or util:
        raise RuntimeError(f"Configured single GPU busy: {busy}")
    configure()
    if args.phase in ["prepare","all"]:
        out.mkdir()
        save_json(out/"config.json",CFG)
        save_json(out/"manifest.json",manifest())
        save_json(out/"fixed_samples.json",json.loads((V11/"fixed_samples.json").read_text()))
        audit={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "python":sys.version,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,
               "platform":platform.platform(),"gpu_uuid":CFG["gpu_uuid"],"history_inventory":inventory(),
               "AGENTS":"No applicable AGENTS.md found in repository, experiment paths or ancestors",
               "checkpoint_sha256":{str(p):sha(p) for p in [V1/CFG["baseline_checkpoint"],V1/CFG["control_C_checkpoint"],ROOT/"checkpoints/GVC-RT_I.pt",ROOT/"checkpoints/GVC-RT_P.pt"]}}
        save_json(out/"input_audit.json",audit)
        (out/"nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"],text=True))
    else:
        assert json.loads((out/"config.json").read_text())==CFG
    status={"phase":args.phase,"status":"RUNNING"}
    try:
        fixed=FixedModels()
        lp=lpips_model()
        if args.phase in ["prepare","all"]:
            import teachers
            tensors,calibration=teachers.prepare(out,fixed,lp)
        if args.phase in ["direct","all"]:
            import experiment
            experiment.direct_phase(out,fixed,lp)
        if args.phase in ["train","all"]:
            import experiment
            status.update(experiment.train_phases(out,fixed,lp))
        fixed.assert_frozen()
        initial=json.loads((out/"input_audit.json").read_text())
        assert inventory()==initial["history_inventory"]
        for p,digest in initial["checkpoint_sha256"].items():
            assert sha(p)==digest
        status.setdefault("outcome","PHASE_COMPLETE")
        status.update(status="COMPLETE",historical_files_unchanged=True)
    except Exception:
        status.update(status="BLOCKED/INCONCLUSIVE",exception=traceback.format_exc())
        raise
    finally:
        save_json(out/f"status_{args.phase}.json",status)
        print("STATUS",status,flush=True)


if __name__=="__main__":
    main()
