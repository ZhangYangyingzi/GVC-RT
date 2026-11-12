from util import *
with lock('auxiliary.lock'):
 jobs=[('extended_checks',2,['extended_checks.py']),('extra_development_cache',1,['cache_data.py','--shard','0','--dev','--extra'])];ps=[]
 for name,gpu,args in jobs:
  h=(ROOT/'logs'/f'{name}.log').open('a');p=subprocess.Popen([PYTHON,'-B',*args],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=h,stderr=subprocess.STDOUT);ps.append((name,p,h,gpu));event('AUXILIARY_STARTED',task=name,child_pid=p.pid,physical_gpu=gpu)
 while any(p.poll() is None for name,p,h,gpu in ps):
  heartbeat('auxiliary','CHECKS_AND_FIXED_UVG_CACHE',children=[dict(task=name,pid=p.pid,gpu=gpu,returncode=p.poll()) for name,p,h,gpu in ps]);time.sleep(5)
 for name,p,h,gpu in ps:h.close();assert p.returncode==0,(name,p.returncode)
 heartbeat('auxiliary','COMPLETE')
