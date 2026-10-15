"""Single detached screen driver; lock and process identity prevent duplicate launch."""
import subprocess,fcntl,time
from audit_io import *
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
def main():
 with (ROOT/'launcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (ROOT/'launch.json').exists():
   prior=load(ROOT/'launch.json');pid=prior.get('PID')
   if pid and Path(f'/proc/{pid}/cmdline').exists():
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(ROOT/'run_all.py') in cmd:print('EXISTING_DRIVER',pid);return
  for gate in ('preflight.json','selftest.json','sanity/sanity.json','analysis_selftest.json'):assert load(ROOT/gate)['status']=='PASS'
  code={p.name:sha(p) for p in ROOT.glob('*.py')}
  if (ROOT/'implementation_hashes.json').exists():assert load(ROOT/'implementation_hashes.json')==code,'Code changed since launch; review dependency receipts before resume'
  else:dump('implementation_hashes.json',code)
  session='gvcrt_v633_'+str(int(time.time()));argv=[PYTHON,'-B','-u',str(ROOT/'run_all.py')];log=ROOT/'logs/driver.log'
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  if subprocess.run(['which','screen'],stdout=subprocess.DEVNULL).returncode==0:
   # screen logging preserves the exact argv; no shell interpolation.
   subprocess.run(['screen','-L','-Logfile',str(log),'-dmS',session,*argv],cwd=REPO,env=env,check=True)
   time.sleep(2)
   ps=subprocess.check_output(['ps','-eo','pid,args'],text=True);rows=[x for x in ps.splitlines() if str(ROOT/'run_all.py') in x and len(x.split())>1 and x.split()[1]==PYTHON];assert len(rows)==1,rows;pid=int(rows[0].split()[0])
  else:
   with log.open('a') as out:p=subprocess.Popen(['nohup',*argv],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);pid=p.pid
   session='nohup'
  dump('launch.json',dict(PID=pid,session=session,argv=argv,log=str(log),protocol_sha256=sha(ROOT/'protocol.json'),implementation_hashes=code,started=time.time()))
  print('DETACHED_DRIVER_STARTED',pid,session)
if __name__=='__main__':main()
