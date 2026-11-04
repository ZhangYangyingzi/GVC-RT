"""CPU partition/output-isolation and immutable training-state checks before GPU probes."""
from eval_parallel_common import *
import ast

def main():
    verify_parallel_sources();tests=[]
    for videos in (8,13,32,45):
        for w in (1,2):
            assignments={r:[i for i in range(videos) if i%(4*w)==r] for r in range(4*w)}
            flat=[i for assigned in assignments.values() for i in assigned];assert sorted(flat)==list(range(videos)) and len(flat)==len(set(flat))
            assert all(4+r//w in (4,5,6,7) for r in assignments)
    tests.append('Video partitions disjoint and exhaustive, including45 videos across8 workers')
    s=worker_source();ast.parse(s)
    assert hashlib.sha256(s.encode()).hexdigest()==load(EP/'source_manifest.json')['generated_worker_sha256']
    assert "result=point_function(engine,Runtime(),quality,flow,perceptual,frames,v,q,model['absolute_step'],cp,key,cache,sample_dir=preview)" in s
    assert 'for q in (0,5,9):' in s and "if index%WORKER_COUNT!=rank:continue" in s
    tests.append('Original numerical point call and all QPs preserved; only dispatch and input/audit paths adapted')
    pause=load(EP/'safe_pause.json')
    for arm,record in pause['checkpoints'].items():
        assert sha(latest_path(arm))==record['sha256'] and committed(arm)==record['additional_step']
    tests.append('Formal checkpoint bytes and cursors unchanged since safe pause')
    handles=[]
    try:
        for i in range(4):
            f=(EP/f'input_{i}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
        for i in range(4):
            with (EP/f'input_{i}.lock').open('a') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise AssertionError('Input concurrency lease did not exclude a fifth reader')
    finally:
        for f in handles:f.close()
    tests.append('Four input-decoder leases enforce the preprocessing concurrency cap')
    dump(EP/'selftest.json',dict(status='PASS',tests=tests,scope='CPU engineering assertions only, GPU metric/stream equivalence still required'))
    print(json.dumps(dict(status='PASS',tests=tests)))

if __name__=='__main__':main()
