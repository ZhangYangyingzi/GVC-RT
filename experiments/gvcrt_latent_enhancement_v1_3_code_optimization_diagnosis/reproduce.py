"""One-command, fresh-directory reproduction of the bounded diagnosis."""
import argparse
import subprocess
from bridge import HOME,ROOT,sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if (HOME/args.output).exists():
        raise FileExistsError("Use a new output basename")
    subprocess.run([sys.executable,"-u",str(HOME/"run.py"),"--output",args.output,"--phase","all"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HOME/"postprocess.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HOME/"verify.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HOME/"visualize.py"),"--output",args.output],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HOME/"final_facts.py"),"--output",args.output],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
