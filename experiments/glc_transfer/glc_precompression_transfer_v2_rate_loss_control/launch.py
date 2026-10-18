from preflight import *
def main():
 p=R/'status.json'
 if p.exists():
  s=json.loads(p.read_text());pid=s.get('PID')
  if s.get('status')=='COMPLETE' and (R/'final_integrity.json').exists() and json.loads((R/'final_integrity.json').read_text()).get('status')=='PASS':print('ALREADY COMPLETE; no duplicate launch');return
  if pid:
   try:os.kill(pid,0);print('EXISTING DRIVER',pid);return
   except ProcessLookupError:pass
 session='glc_transfer_v2_'+str(int(time.time()));log=R/'logs/driver.log';log.parent.mkdir(exist_ok=True)
 argv=['screen','-L','-Logfile',str(log),'-dmS',session,'env','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2',sys.executable,'-B','-u',str(R/'run_pipeline.py')]
 subprocess.run(argv,cwd=GLC,check=True);save('launch.json',dict(session=session,command=argv,time=time.time(),log=str(log)))
 print('LAUNCHED',session)
if __name__=='__main__':main()
