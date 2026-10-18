"""Drain only owned CPU plan preparation before loading updated resource guards."""
from preflight import *
from data import load
import signal
def procstate(pid):
 p=Path(f'/proc/{pid}/status')
 return next(x.split()[1] for x in p.read_text().splitlines() if x.startswith('State:')) if p.exists() else None
def main():
 s=load(R/'status.json');pid=s['PID'];assert s['phase']=='data' and str(R/'run_pipeline.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 paused=False
 try:
  os.kill(pid,signal.SIGSTOP);paused=True;limit=time.time()+10
  while procstate(pid)!='T':assert time.time()<limit;time.sleep(.1)
  children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
  for child in children:assert str(R/'data.py').encode() in Path(f'/proc/{child}/cmdline').read_bytes() or procstate(child)=='Z'
  save('audits/preparation_handoff.json',dict(status='DRAINING_CPU_PREPARATION',old_driver=pid,children=children,data_work_interrupted=False,GPU_work_interrupted=False))
  limit=time.time()+1800
  while any(procstate(c) not in (None,'Z') for c in children):assert time.time()<limit;time.sleep(15)
  plan=load(R/'plans/shared.json');assert len(plan['updates'])==1000 and plan['seed']==20261016 and load(R/'audits/plan_integrity.json')['status']=='PASS'
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);paused=False;limit=time.time()+60
  while Path(f'/proc/{pid}').exists():assert time.time()<limit;time.sleep(1)
  subprocess.run([sys.executable,'-B',str(R/'launch.py')],cwd=GLC,check=True)
  save('audits/preparation_handoff.json',dict(status='PASS',old_driver=pid,plan_sha256=sha(R/'plans/shared.json'),data_work_interrupted=False,GPU_work_interrupted=False,numerical_changes='v2 categorical forward alignment, audited separately; unchanged real codec',resource_guards_loaded=True,time=time.time()))
 except Exception:
  if paused and procstate(pid) is not None:os.kill(pid,signal.SIGCONT)
  raise
if __name__=='__main__':main()
