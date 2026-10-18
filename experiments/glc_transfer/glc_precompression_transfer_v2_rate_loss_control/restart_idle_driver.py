"""Bounded recovery: restart only an idle, verified driver of this experiment."""
from preflight import *
from data import load
import argparse,signal
def main(pid):
 s=load(R/'status.json');e=load(R/'evaluation_scheduler.json')
 assert s['PID']==pid and s['phase']=='evaluation' and not e['running']
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 assert str(R/'run_pipeline.py').encode() in cmd
 for _ in range(10):
  if not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip():break
  time.sleep(1)
 else:raise RuntimeError('Driver has active children; no restart performed')
 assert not load(R/'evaluation_scheduler.json')['running']
 save(f'audits/recovery/idle_driver_restart_{time.time_ns()}.json',dict(status='PLANNED',PID=pid,previous_status=s,previous_scheduler=e,reason='bounded recovery of extra GPU volatility margin; standard reserve and full measured-peak admission unchanged',policy_checks=load(R/'audits/stable_margin_policy_checks.json'),training_repeated=False,other_processes_untouched=True))
 os.kill(pid,signal.SIGTERM)
 for _ in range(20):
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(.5)
 else:raise RuntimeError('Owned driver did not exit; no duplicate launched')
 subprocess.run([sys.executable,'-B',str(R/'launch.py')],cwd=R,check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);main(p.parse_args().pid)
