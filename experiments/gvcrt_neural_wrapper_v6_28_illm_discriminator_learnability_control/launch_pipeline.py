from io21 import *
import fcntl
(ROOT/'logs').mkdir(exist_ok=True)
f=(ROOT/'launch.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=ROOT/'scheduler_identity.json'
if p.exists():
 s=load(p)
 try:
  st=Path('/proc')/str(s['PID'])/'stat';assert st.read_text().split()[21]==s['start_ticks'];print('Existing live scheduler preserved',s['PID']);sys.exit(0)
 except (FileNotFoundError,AssertionError):pass
log=(ROOT/'logs/pipeline.log').open('a');proc=subprocess.Popen([PYTHON,str(ROOT/'pipeline.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(p,dict(PID=proc.pid,start_ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().split()[21],command=[PYTHON,str(ROOT/'pipeline.py')],started_unix=time.time()));print('Scheduler PID',proc.pid)
