"""Persistent screen launch; driver writes PID and per-GPU worker PIDs."""
import subprocess
import time
from audit_io import *

if __name__=='__main__':
    assert load(ROOT/'sanity/proxy_sanity.json')['status']=='PASS'
    session='gvcrt_v631a_'+time.strftime('%Y%m%d_%H%M%S')
    argv=[sys.executable,'-B','-u',str(ROOT/'run_all.py'),'--resume']
    subprocess.run(['screen','-L','-Logfile',str(ROOT/'logs/driver.log'),'-dmS',session,*argv],cwd=REPO,check=True)
    dump('launch.json',dict(screen_session=session,argv=argv,cwd=str(REPO),started=time.time(),persistent=True))
    print('LAUNCHED',session,flush=True)
