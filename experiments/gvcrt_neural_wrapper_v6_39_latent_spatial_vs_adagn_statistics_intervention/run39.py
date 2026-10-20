"""Bounded, resumable read-only task scheduler for physical GPUs 6 and 7."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PYTHON="/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python"
TASKS=json.loads((HERE/"evaluation/task_manifest.json").read_text())


def write(obj):
    p=HERE/"run_status.json";t=p.with_name(p.name+".tmp")
    t.write_text(json.dumps(obj,indent=2)+"\n");t.replace(p)


def key(t):
    t=t["task"]
    return f"{t['dataset']}_{t['video']}_q{t['qp']}"


def done(t):
    p=HERE/"tasks"/f"{key(t)}.json"
    if not p.exists():return False
    try:
        r=json.loads(p.read_text())
        return r.get("status")=="PASS" and r["task"]==t["task"] and r["adagn_path_checks"]==1260 and r["new_optimizer_updates"]==r["new_encodes"]==0
    except Exception:return False


def main(pilot):
    lock=HERE/"driver.lock"
    with lock.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidates=TASKS
        if pilot:
            ev=json.loads((HERE.parent/"gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/eval_manifest.json").read_text())
            first={d:rows[0]["video_index"] for d,rows in ev.items()}
            candidates=[t for t in TASKS if t["task"]["video"]==first[t["task"]["dataset"]]]
        pending=[t for t in candidates if not done(t)]
        running={}
        failures={}
        retries={}
        gpus=[6,7]
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending:continue
                task=pending.pop(0)
                k=key(task)
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
                logpath=HERE/"logs"/f"{k}.attempt{retries.get(k,0)+1}.log"
                logpath.parent.mkdir(parents=True,exist_ok=True)
                stream=logpath.open("w")
                cmd=[PYTHON,str(HERE/"intervene39.py"),"--task",json.dumps(task["task"]),"--gpu",str(gpu)]
                p=subprocess.Popen(cmd,env=env,cwd=HERE.parents[1],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(p,task,stream,logpath,time.time())
            write(dict(status="RUNNING",pilot=pilot,complete=sum(done(t) for t in candidates),planned=len(candidates),
                       running=[dict(task=key(x[1]),gpu=g,pid=x[0].pid,log=str(x[3]),start=x[4]) for g,x in running.items()],
                       pending=len(pending),failures=failures,retries=retries,updated=time.time()))
            time.sleep(15)
            for gpu,x in list(running.items()):
                p,task,stream,logpath,start=x
                rc=p.poll()
                if rc is None:continue
                stream.close();del running[gpu]
                k=key(task)
                if rc==0 and done(task):continue
                retries[k]=retries.get(k,0)+1
                tail=logpath.read_text(errors="replace")[-6000:]
                if "endpoint" in tail.lower() and "mismatch" in tail.lower():
                    failures[k]=dict(returncode=rc,error=tail,blocking=True)
                    pending.clear()
                elif retries[k]<3:
                    pending.append(task)
                else:failures[k]=dict(returncode=rc,error=tail)
        write(dict(status="PASS" if not failures and all(done(t) for t in candidates) else "PARTIAL",
                   pilot=pilot,complete=sum(done(t) for t in candidates),planned=len(candidates),
                   failures=failures,retries=retries,updated=time.time()))
        print("RUN", "PASS" if not failures else "PARTIAL",sum(done(t) for t in candidates),len(candidates),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--pilot",action="store_true");main(a.parse_args().pilot)
