"""Run the fixed staged prototype once in a fresh directory, then aggregate and verify."""
import argparse
import subprocess
from support import HERE,ROOT,sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if (HERE/args.output).exists():
        raise FileExistsError("Choose a new output basename")
    subprocess.run([sys.executable,"-u",str(HERE/"run.py"),"--output",args.output,"--phase","all"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"postprocess.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"verify.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/"visualize.py"),"--output",args.output],cwd=ROOT,check=True)
    if (HERE/args.output/"val6_CODED_l3/summary.json").exists():
        subprocess.run([sys.executable,str(HERE/"final_data.py"),"--output",args.output],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
