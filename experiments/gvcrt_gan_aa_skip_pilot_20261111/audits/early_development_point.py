"""Execute the first prescribed development point while other groups train.

Only this experiment's GPU0 trainer is temporarily suspended and always resumed
in finally. The evaluation uses independent step0 receiver states.
"""
from pathlib import Path
import sys
ROOT0=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT0))
from common_s2 import *
import signal,traceback


def main():
    env()
    with lock('locks/early_development_point.lock'):
        matches=[]
        while not matches:
            for p in (ROOT/'receipts/processes').glob('*.json'):
                a=load(p);proc=Path(f'/proc/{a["PID"]}/cmdline')
                if a['args'][:3]==['train.py','--group','J0'] and proc.exists() and Path(f'/proc/{a["PID"]}/cwd').resolve()==ROOT:
                    if b'train.py' in proc.read_bytes():matches.append(a)
            if not matches:
                dump(ROOT/'audits/early_development_control.json',dict(status='WAITING_TO_ACQUIRE_NEXT_J0_TRAINER',PID=os.getpid(),time=time.time()))
                time.sleep(2)
        owner=max(matches,key=lambda a:a['start_time']);pid=owner['PID']
        dump(ROOT/'audits/early_development_control.json',dict(status='RUNNING',PID=os.getpid(),suspended_own_trainer_PID=pid,physical_GPU=0,point=dict(video_index=0,QP=0),time=time.time(),driver_SHA256=sha(Path(__file__))))
        os.kill(pid,signal.SIGSTOP)
        try:
            from cache_codec import cache_paths
            video=load(ROOT/'manifests/development_videos.json')[0];q=0
            request=ROOT/'cache/requests/early_development_point.json';dump(request,[dict(row=video,qp=q,training=False)])
            subprocess.run([PYTHON,'cache_job.py',str(request)],cwd=ROOT,check=True)
            key,path,rp=cache_paths(video,q,False);assert sha(path)==load(rp)['cache_sha256']
            original=ROOT/'cache/original_FP32'/f'{key}.pt'
            subprocess.run([PYTHON,'original_reconstruction.py',str(path),str(original)],cwd=ROOT,check=True)
            import torch
            from evaluate import Scorer
            scorer=Scorer();data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);orig=torch.load(original,map_location='cpu',weights_only=True,mmap=True)
            for model in ['Original','B300',*GROUPS]:scorer.point(data,model,0,orig)
            dump(ROOT/'audits/early_development_result.json',dict(status='PASS',video=video['sequence'],QP=0,points_per_state=1,models=['Original','B300',*GROUPS],frames=64,phase='prescribed development step0',time=time.time()))
        except Exception as e:
            dump(ROOT/'audits/early_development_failure.json',dict(status='FAILED_EVALUATION_ONLY',error=repr(e),traceback=traceback.format_exc(),time=time.time()))
            raise
        finally:
            proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'train.py' in proc.read_bytes() and Path(f'/proc/{pid}/cwd').resolve()==ROOT:os.kill(pid,signal.SIGCONT)
            dump(ROOT/'audits/early_development_control.json',dict(status='TRAINER_RESUMED',PID=os.getpid(),own_trainer_PID=pid,time=time.time(),driver_SHA256=sha(Path(__file__))))
            subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
