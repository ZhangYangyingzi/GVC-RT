from io21 import *
def main():
 state=load(ROOT/'pipeline_status.json');pid=state.get('PID');cmd=Path(f'/proc/{pid}/cmdline') if pid else None
 if cmd and cmd.exists() and str(ROOT/'run_pipeline.py').encode() in cmd.read_bytes():print('ACTIVE SCHEDULER',pid);return
 out=(ROOT/'logs/run_pipeline.log').open('ab');p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
 dump(ROOT/'launch.json',dict(PID=p.pid,argv=p.args,unix=time.time(),persistent_new_session=True));print('STARTED SCHEDULER',p.pid)
if __name__=='__main__':main()
