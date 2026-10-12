from io21 import *
import fcntl

def main():
 lock=(ROOT/'diagnostic_postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  plan=load(ROOT/'diagnostic_plan.json')['updates'];paths=[ROOT/'diagnostics/points'/f'{r["panel_id"]}_s{s}.json' for r in plan for s in (0,250,500,1000)]
  if all(p.exists() and load(p).get('status')=='PASS' for p in paths):break
  time.sleep(20)
 from summarize import diagnostics
 diagnostics();subprocess.run([PYTHON,'-B','-u',str(ROOT/'plots.py')],check=True)
 env=os.environ.copy();env['V627_PUBLICATION_PHASE']='supervision_diagnostics_complete';subprocess.run([PYTHON,'-B','-u',str(ROOT/'publish_github.py')],env=env,cwd=REPO,check=True);print('DIAGNOSTIC POSTPROCESS PASS',flush=True)
if __name__=='__main__':main()
