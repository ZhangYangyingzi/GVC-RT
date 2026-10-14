"""One supplemental persistent evaluation scheduler, same formal task locks."""
from io32 import *
import subprocess
def main():
 p=ROOT/'parallel_evaluation_status.json'
 if p.exists():
  r=load(p)
  try:os.kill(r['PID'],0);assert str(ROOT/'parallel_eval32.py').encode() in Path(f'/proc/{r["PID"]}/cmdline').read_bytes();print('ALREADY RUNNING',r['PID']);return
  except ProcessLookupError:pass
 session=f'gvcrt_v632_parallel_{int(time.time())}';argv=[PYTHON,'-B','-u',str(ROOT/'parallel_eval32.py')];log=ROOT/'logs/parallel_driver.log'
 dump(ROOT/'parallel_launch.json',dict(session=session,argv=argv,log=str(log),started_unix=time.time(),additional_task_budget=0))
 subprocess.run(['screen','-dmS',session,'-L','-Logfile',str(log),*argv],cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
 print('PARALLEL EVALUATION SCREEN',session,flush=True)
if __name__=='__main__':main()
