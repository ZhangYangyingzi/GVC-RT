"""Start both experiments together and deliver each independently when ready."""
import fcntl,time
from parallel_utils import *
def merge(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]:result[-1][1]=max(hi,result[-1][1])
        else:result.append([lo,hi])
    return result

def compute_overlap():
    intervals=[]
    for root in (A,B):
        intervals.append(merge([(r['start_unix'],r['end_unix']) for p in (root/'parts').glob('*/*/video_*_qp*.json') for r in [load(p)]]))
    return sum(max(0,min(a1,b1)-max(a0,b0)) for a0,a1 in intervals[0] for b0,b1 in intervals[1])

def cpu(script,args,log,python=PYTHON):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (ROOT/'logs'/log).open('ab') as output:
        subprocess.run([python,'-B','-u',str(ROOT/script),*args],cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)

def main():
    lock=(ROOT/'parts/parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={};starts={};ends={};codes={};delivered=set()
    dump(ROOT/'parallel_run_integrity.json',dict(status='PENDING',gpu_A=[4,5],gpu_B=[6,7],reason='Concurrent run in progress; final audits pending'))
    try:
        for which,root in [('A',A),('B',B)]:
            assert load(root/'config.json')['evaluation_only']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,load(root/'config.json')['gpus'])),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            starts[which]=time.time()
            with (ROOT/f'logs/experiment_{which}.log').open('ab') as log:
                children[which]=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'run_experiment.py'),'--experiment',which],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        dump(ROOT/'parallel_processes.json',dict(pids={k:p.pid for k,p in children.items()},started_unix=starts,gpu_A=[4,5],gpu_B=[6,7]))
        while len(codes)<2:
            for which,child in children.items():
                if which not in codes and child.poll() is not None:codes[which]=child.returncode;ends[which]=time.time()
            for which in tuple(codes):
                if codes[which]==0 and which not in delivered:
                    cpu('report.py',['--experiment',which],f'report_{which}.log')
                    if which=='A':cpu('plot.py',[],'plots_A.log',PLOT_PYTHON)
                    cpu('finalize.py',['--experiment',which],f'finalize_{which}.log')
                    assert load(exp(which)/'final_integrity.json')['status']=='PASS'
                    status=load(exp(which)/'pipeline_status.json')
                    status.update(status='PASS',phase='complete',independent_delivery=True)
                    dump(exp(which)/'pipeline_status.json',status)
                    delivered.add(which)
                    print('EXPERIMENT READY',which,'R-D curves ready' if which=='A' else '',flush=True)
            dump(ROOT/'parallel_status.json',dict(status='RUNNING' if not any(codes.values()) else 'FAIL_WAITING_FOR_OTHER_EXPERIMENT',
                pids={k:p.pid for k,p in children.items()},exit_codes=codes,experiments={k:load(exp(k)/'pipeline_status.json') for k in children}))
            if len(codes)<2:time.sleep(10)
        overlap=max(0,min(ends.values())-max(starts.values()))
        base=dict(experiment_A_exit_code=codes['A'],experiment_B_exit_code=codes['B'],gpu_A=[4,5],gpu_B=[6,7],
            started_in_parallel=abs(starts['A']-starts['B'])<5 and overlap>0,overlap_runtime_seconds=overlap,
            compute_overlap_seconds=compute_overlap(),pids={k:p.pid for k,p in children.items()},started_unix=starts,ended_unix=ends)
        assert all(v==0 for v in codes.values()),codes
        # Only the combined parallel-run audit waits for both experiments.
        assert delivered=={'A','B'}
        base.update(experiment_A_integrity=load(A/'final_integrity.json')['status'],experiment_B_integrity=load(B/'final_integrity.json')['status'])
        assert base['started_in_parallel'] and base['compute_overlap_seconds']>0
        assert base['experiment_A_integrity']==base['experiment_B_integrity']=='PASS'
        dump(ROOT/'parallel_run_integrity.json',dict(base,status='PASS'))
        cpu('print_status.py',[],'final_stdout.log')
        for root in (A,B):dump(root/'pipeline_status.json',dict(status='PASS',phase='complete',expected_points=load(root/'config.json')['expected_points']))
        dump(ROOT/'parallel_status.json',dict(status='PASS',phase='complete',integrity=str(ROOT/'parallel_run_integrity.json')))
    except BaseException as exc:
        dump(ROOT/'parallel_run_integrity.json',dict(status='FAIL',error=repr(exc),experiment_A_exit_code=codes.get('A'),experiment_B_exit_code=codes.get('B'),
              experiment_A_integrity=load(A/'final_integrity.json')['status'],experiment_B_integrity=load(B/'final_integrity.json')['status'],
              gpu_A=[4,5],gpu_B=[6,7],compute_overlap_seconds=compute_overlap()))
        raise
if __name__=='__main__':main()
