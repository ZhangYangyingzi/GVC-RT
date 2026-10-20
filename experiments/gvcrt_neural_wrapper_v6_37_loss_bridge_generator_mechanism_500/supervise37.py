from io32 import *
import subprocess,fcntl

def main(job):
 lock=(ROOT/'runtime'/f'{job["key"]}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 attempt=job['key']+'_'+str(time.time_ns());p=None
 with (ROOT/'logs'/f'{job["key"]}.log').open('a') as f:
  p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/job['script']),*map(str,job.get('args',[])),'--gpu',str(job['gpu'])],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
  launch=dict(job,attempt=attempt,worker_PID=p.pid,wrapper_PID=os.getpid(),started=time.time());dump(ROOT/'runtime/attempts'/f'{attempt}_launch.json',launch);dump(ROOT/'runtime/active'/f'{job["key"]}.json',launch)
  rc=p.wait()
 dump(ROOT/'runtime/attempts'/f'{attempt}_exit.json',dict(launch,exit_code=rc,finished=time.time()))
 dump(ROOT/'runtime/exits'/f'{job["key"]}.json',dict(launch,exit_code=rc,finished=time.time()))
 print('EXIT',job['key'],rc,flush=True)
if __name__=='__main__':main(json.loads(sys.argv[1]))
