"""One experiment process, with two workers restricted to its assigned physical GPUs."""
import argparse,fcntl,time
from parallel_utils import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=('A','B'),required=True);args=p.parse_args()
    root=exp(args.experiment);config=load(root/'config.json')
    lock=(root/'parts/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};waiting=list(enumerate(config['gpus']));started=time.time()
    try:
        while waiting or active:
            for shard,gpu in waiting[:]:
                free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<(7000 if args.experiment=='A' else 14000):continue
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                with (root/f'logs/worker_gpu{gpu}.log').open('ab') as log:
                    child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'worker.py'),'--experiment',args.experiment,'--gpu',str(gpu),'--shard',str(shard)],
                        cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=child;waiting.remove((shard,gpu));print('WORKER START',args.experiment,gpu,child.pid,flush=True)
            for gpu,child in list(active.items()):
                if child.poll() is not None:
                    assert child.returncode==0,f'GPU {gpu} worker exited {child.returncode}'
                    assert load(root/f'parts/worker_gpu{gpu}_done.json')['status']=='PASS'
                    del active[gpu]
            count=len(list((root/'parts').glob('*/*/video_*_qp*.json')))
            dump(root/'pipeline_status.json',dict(status='RUNNING',phase='evaluation',pid=os.getpid(),experiment=args.experiment,
                gpus=config['gpus'],worker_pids={g:p.pid for g,p in active.items()},waiting_gpus=[g for _,g in waiting],
                completed_points=count,expected_points=config['expected_points'],start_unix=started,updated_unix=time.time()))
            if active or waiting:time.sleep(10)
        assert count==config['expected_points']
        if args.experiment=='A':
            dump(root/'pipeline_status.json',dict(status='RUNNING',phase='visualizations',completed_points=count))
            with (root/'logs/visualizations.log').open('ab') as log:
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
                subprocess.run([PYTHON,'-B','-u',str(ROOT/'visualize.py')],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(root/'pipeline_status.json',dict(status='COMPUTE_COMPLETE',phase='awaiting_joint_reports_and_integrity',completed_points=count,
            expected_points=config['expected_points'],start_unix=started,end_unix=time.time(),gpus=config['gpus']))
    except BaseException as exc:
        for child in active.values():
            if child.poll() is None:child.terminate()
        dump(root/'pipeline_status.json',dict(status='FAIL',error=repr(exc)));raise
if __name__=='__main__':main()
