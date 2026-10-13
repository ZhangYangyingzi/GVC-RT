"""Exercise the actual queue, signatures, and failure persistence without a GPU."""
from audit_io import *
def perform(t):
    if valid(t):return
    if t['method']=='forced_failure':raise RuntimeError('V630_EXPECTED_SELFTEST_FAILURE')
    p=ROOT/'audits/selftest_executions.json';state=load(p) if p.exists() else {};key=task_id(t);state[key]=state.get(key,0)+1;dump(p,state)
    artifact=ROOT/'audits/selftest'/f'{key}.json';dump(artifact,dict(status='SELFTEST_ONLY',updates=0))
    dump(output(t),dict(task=t,signature=signature(t),execution_complete=True,status='PASS',artifacts={str(artifact):sha(artifact)},peak_RSS_MiB=64,selftest_not_formal_result=True))
def main():
    from run_all import Queue
    import tempfile
    q=Queue()
    a=dict(kind='selftest',dataset='none',video=0,method='success',qp=-1)
    b=dict(kind='selftest',dataset='none',video=1,method='forced_failure',qp=-1)
    c=dict(kind='selftest',dataset='none',video=2,method='independent_success',qp=-1)
    q.run('selftest_resume',[a],cpu=True)
    before=load(ROOT/'audits/selftest_executions.json')[task_id(a)]
    q.run('selftest_resume',[a],cpu=True)
    assert load(ROOT/'audits/selftest_executions.json')[task_id(a)]==before, 'duplicate successful execution on resume'
    assert valid(a)
    r=load(output(a));r['signature']='deliberately_wrong';dump(output(a),r);assert not valid(a)
    q.run('selftest_repair_signature',[a],cpu=True);assert valid(a)
    q.run('selftest_failure_independent',[b,c],cpu=True)
    assert valid(c) and not valid(b) and any('V630_EXPECTED_SELFTEST_FAILURE' in '\n'.join(e['last_80_lines']) for e in q.failures)
    dump(ROOT/'audits/scheduler_selftest.json',dict(status='PASS',resume_did_not_recompute=True,invalid_signature_recomputed=True,worker_exception_persisted=True,independent_task_after_failure_completed=True,expected_selftest_failure=True,formal_updates=0))
if __name__=='__main__':main()
