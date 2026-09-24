"""Reproduce the fixed diagnostics in a new output directory; no network training."""
import argparse
import subprocess
import sys
from common import HERE,ROOT,CFG


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if (HERE/args.output).exists():
        raise FileExistsError("Choose a new output basename; results are never overwritten")
    if not (HERE/"input_audit.json").exists():
        subprocess.run([sys.executable,str(HERE/"prepare.py")],cwd=ROOT,check=True)
    if not (HERE/"source_identity_checks.json").exists():
        subprocess.run([sys.executable,str(HERE/"check_inputs.py")],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u",str(HERE/"run.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"postprocess.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"verify.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"finalize_artifacts.py"),"--output",args.output],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
