"""Wait for inherited lifecycle leases, then exec the one durable coordinator."""
from common_s2 import *
def main():
    env();p=ROOT/'runner.lock'
    with p.open('a') as lease:
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                status('waiting_for_existing_development_workers',PID=os.getpid(),committed_router_updates=0,
                    reason='Controlled coordinator-only metadata repair; existing step0 evaluation is retained',time_of_snapshot=time.time())
                time.sleep(20)
        fcntl.flock(lease,fcntl.LOCK_UN)
    os.execv(PYTHON,[PYTHON,'-B','-u',str(ROOT/'runner.py')])
if __name__=='__main__':main()
