"""Detached resumable evaluation -> statistics -> report -> integrity -> publication."""
import argparse
import fcntl
import gzip
import shutil
import subprocess
import time
import traceback
from audit_io import *


def run(name,args=None):
    with (ROOT/'logs'/f'{name}.log').open('a') as f:
        python='/data1/anaconda3_new/anaconda_program/bin/python' if name=='analysis' else sys.executable
        subprocess.run([python,'-B','-u',str(ROOT/(name+'.py')),*(args or [])],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)


def publish():
    assert load(ROOT/'integrity.json')['status']=='PASS'
    relative=str(ROOT.relative_to(REPO))
    # Preserve local full CSVs; compress large text artifacts for ordinary GitHub blobs.
    ignored=[];compressed=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.log','.txt') and p.stat().st_size>40_000_000:
            dst=p.with_suffix(p.suffix+'.gz')
            with p.open('rb') as src,gzip.open(dst,'wb',compresslevel=6) as out:shutil.copyfileobj(src,out)
            ignored.append('/'+str(p.relative_to(ROOT)))
            compressed.append(dict(original=str(p.relative_to(ROOT)),sha256=sha(p),compressed=str(dst.relative_to(ROOT)),compressed_sha256=sha(dst)))
    if ignored:
        with (ROOT/'.gitignore').open('a') as f:f.write('\n'+'\n'.join(ignored)+'\n')
    dump('publication_artifacts.json',dict(large_text_gzip=compressed,restore='python restore_large_text.py',binary_cache_omitted=True))
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    def git(*args):return subprocess.run(['git',*args],cwd=REPO,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    add=git('add','--',relative)
    if add.returncode:raise RuntimeError(add.stdout)
    commit=git('commit','--only','-m','Add v6.31a latent fidelity vs generative utility audit','--',relative)
    if commit.returncode and 'nothing to commit' not in commit.stdout:raise RuntimeError(commit.stdout)
    head=git('rev-parse','HEAD').stdout.strip();branch=git('branch','--show-current').stdout.strip()
    push=git('push','origin',f'HEAD:refs/heads/{branch}')
    (ROOT/'logs/push.log').write_text(push.stdout)
    dump('publication_status.json',dict(status='PASS' if push.returncode==0 else 'PUSH_FAILED_LOCAL_RESULTS_PRESERVED',
         commit=head,branch=branch,push_returncode=push.returncode,force_push=False,remote_conflict_not_overwritten=True,log='logs/push.log'))
    # The publication receipt is local post-push state; no speculative success is committed.
    return push.returncode==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert load(ROOT/'sanity/proxy_sanity.json')['status']=='PASS'
    assert load(ROOT/'selftest.json')['status']=='PASS'
    assert all(sha(ROOT/p)==h for p,h in load(ROOT/'implementation_hashes.json').items())
    workers=[]
    for gpu in (4,5,6,7):
        f=(ROOT/'logs'/f'gpu{gpu}.log').open('a')
        p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'worker.py'),'--gpu',str(gpu)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        workers.append((gpu,p,f))
    dump('status.json',dict(status='RUNNING',audit_scope='paper-faithful proxy audit',background_pid=os.getpid(),
         sanity='PASS',full_experiment_launched=True,gpu_workers=[dict(gpu=g,pid=p.pid) for g,p,f in workers],
         log_directory=str(ROOT/'logs'),started=time.time()))
    print('ALL_GPU_WORKERS_STARTED',[(g,p.pid) for g,p,f in workers],flush=True)
    # Waiting happens in detached driver, never in the interactive Codex session.
    while any(p.poll() is None for g,p,f in workers):
        failed=[(g,p.returncode) for g,p,f in workers if p.poll() not in (None,0)]
        if failed:
            dump('status.json',dict(status='WORKER_FAILURE',failed=failed,background_pid=os.getpid(),
                 gpu_workers=[dict(gpu=g,pid=p.pid,returncode=p.poll()) for g,p,f in workers],resume='reproduce.sh --resume'))
        time.sleep(20)
    assert all(p.returncode==0 for g,p,f in workers),[(g,p.returncode) for g,p,f in workers]
    for g,p,f in workers:f.close()
    dump('status.json',dict(status='AGGREGATING',background_pid=os.getpid(),sanity='PASS'))
    run('analysis');run('render_examples')
    from worker import valid
    tasks=load(ROOT/'task_manifest.json')['tasks']
    assert all(valid(t) for t in tasks)
    changed=[p for p,h in load(ROOT/'source_hashes.json').items() if sha(p)!=h]
    assert not changed,('source/model dependency changed',changed)
    assert all(sha(ROOT/p)==h for p,h in load(ROOT/'implementation_hashes.json').items())
    required=['all_rd_points.csv','latent_fidelity_per_frame.csv','latent_fidelity_per_video_qp.csv',
              'all_matched_pairs.csv','primary_matched_pairs.csv','inversion_pairs.csv','inversion_summary.csv',
              'correlation_by_video.csv','rate_controlled_correlation.csv','bootstrap_summary.csv','fid_summary.csv','REPORT.md']
    assert all((ROOT/f).exists() for f in required)
    assert not load(ROOT/'counterexample_selection.json')['render_pending']
    dump('integrity.json',dict(status='PASS',audit_scope='paper-faithful proxy audit',sanity='PASS',
         points=len(tasks),expected_points=1380,real_RANS=True,independent_decode=True,causal_DPB=True,
         checkpoints_and_sources_unchanged=True,no_training=True,noise_repeat_points=10,
         exact_original_teacher_verified=False,artifacts={f:sha(ROOT/f) for f in required}))
    dump('status.json',dict(status='EVALUATION_COMPLETE_PUBLISHING',background_pid=os.getpid(),completed_points=len(tasks),sanity='PASS'))
    published=publish()
    dump('status.json',dict(status='COMPLETE' if published else 'COMPLETE_LOCAL_PUSH_FAILED',completed_points=len(tasks),
         sanity='PASS',integrity='PASS',publication='publication_status.json',background_pid=os.getpid()))
    print('PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        dump('status.json',dict(status='FAILED',background_pid=os.getpid(),traceback=traceback.format_exc(),resume='reproduce.sh --resume'))
        raise
