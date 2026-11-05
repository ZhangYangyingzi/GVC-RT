"""Resume 135 independent original-code points, validate, plot, and publish."""
from common import *
import argparse
import fcntl
import subprocess
import sys
import time
import traceback


def cpu_preflight():
    from runtime import initialize,load_models,verify_codec_imports,model_sha
    rt=initialize('cpu');im,pm=load_models(rt)
    p=load(ROOT/'protocol.json')
    dump(ROOT/'cpu_preflight.json',dict(status='PASS',original_commit=COMMIT,TCG_parameters=0,
                                      I_state_tensors=len(im.state_dict()),P_state_tensors=len(pm.state_dict()),
                                      model_hashes=[model_sha(m) for m in (im,pm)],imports=verify_codec_imports(),
                                      checkpoint_sha256=p['checkpoint_sha256'],protocol_sha256=sha(ROOT/'protocol.json')))
    del im,pm
    print('CPU_PREFLIGHT_PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-preflight',action='store_true');ap.add_argument('--skip-publish',action='store_true')
    args=ap.parse_args()
    for d in ('logs','cache/mpl','cache/cuda','tmp','points','progress'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',MPLCONFIGDIR=str(ROOT/'cache/mpl'),
                      CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),TMPDIR=str(ROOT/'tmp'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (ROOT/'runner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_protocol()
        assert load(ROOT/'source_frame_audit.json')['status']=='PASS'
        if args.cpu_preflight:
            cpu_preflight();return
        files={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
        if (ROOT/'execution_sources.json').exists():
            assert load(ROOT/'execution_sources.json')==files,'Evaluation source changed after run freeze'
        else:dump(ROOT/'execution_sources.json',files)
        os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=4:
            dump(ROOT/'run_status.json',dict(status='BLOCKED_GPU_ACCESS',completed_points=len(list((ROOT/'points').glob('*.json'))),expected_points=135,
                                            reason='This execution context cannot access the four authorized CUDA GPUs',time=time.time()))
            raise RuntimeError('GPU execution access is required; no original metric points were fabricated')
        for gpu in range(4):
            torch.empty(1,device=f'cuda:{gpu}')
        started=time.time();processes=[]
        for rank in range(4):
            log=(ROOT/f'logs/worker{rank}.log').open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(4+rank))
            process=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'evaluate_original.py'),'--rank',str(rank)],cwd=ROOT,
                                     env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
            processes.append((process,log))
        try:
            while any(p.poll() is None for p,_ in processes):
                failures=[p.returncode for p,_ in processes if p.poll() not in (None,0)]
                count=len(list((ROOT/'points').glob('*.json')))
                state=dict(status='RUNNING',completed_points=count,expected_points=135,elapsed_seconds=time.time()-started,
                           worker_pids=[p.pid for p,_ in processes],worker_exit_codes=[p.poll() for p,_ in processes],time=time.time())
                dump(ROOT/'run_status.json',state)
                if failures:
                    for p,_ in processes:
                        if p.poll() is None:p.terminate()
                    raise RuntimeError('An original evaluation worker failed; inspect logs and resume only after fixing the cause')
                print(json.dumps(state),flush=True)
                time.sleep(20)
            assert all(p.returncode==0 for p,_ in processes)
        finally:
            for p,log in processes:
                if p.poll() is None:p.terminate()
                p.wait();log.close()
        from finalize import finalize
        finalize()
        dump(ROOT/'run_status.json',dict(status='COMPLETE_LOCAL',completed_points=135,expected_points=135,elapsed_seconds=time.time()-started))
        if not args.skip_publish:
            from publish import publish
            publish()


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        old=load(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {}
        if old.get('status') not in ('BLOCKED_GPU_ACCESS','COMPLETE_LOCAL'):
            dump(ROOT/'run_status.json',dict(old,status='FAILED',error=traceback.format_exc()))
        sys.exit(1)
