"""Audited metadata-only signature migration; no decode/features/FID are rerun."""
from audit_io import *
import concurrent.futures
def main():
    old=load(ROOT/'audits/monitor_repair/old_code_hashes.json')
    before=(ROOT/'audits/monitor_repair/old_audit_io.py').read_text()
    assert hashlib.sha256(before.encode()).hexdigest()==old['audit_io.py']
    expected=before.replace("p.with_name(p.name + f'.{os.getpid()}.tmp')", "p.with_name(p.name + f'.{os.getpid()}.{time.time_ns()}.tmp')").replace('def signature(t):','def signature(t, code_override=None):').replace("code={p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}","code=code_override if code_override is not None else {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}")
    assert expected==(ROOT/'audit_io.py').read_text(), 'unexpected IO modification; stop migration'
    now={p.name:sha(p) for p in ROOT.glob('*.py')}
    assert set(now)==set(old)|{'repair_monitor.py'}
    assert all(now[n]==h for n,h in old.items() if n!='audit_io.py'), 'model, feature, statistics, or other code changed'
    def stress(i):
        for j in range(100):dump(ROOT/'audits/monitor_repair/atomic_stress.json',dict(i=i,j=j))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(stress,range(4)))
    assert load(ROOT/'audits/monitor_repair/atomic_stress.json')['j']==99
    migrated=[];already=[]
    for p in sorted((ROOT/'tasks').glob('*.json')):
        r=load(p);t=r['task']
        if valid(t):already.append(task_id(t));continue
        assert r['signature']==signature(t,old),('old signature does not match frozen inputs and original code',p)
        assert r['execution_complete']
        assert all(Path(a).is_file() and sha(a)==h for a,h in r['artifacts'].items()),('artifact changed',p)
        receipt_sha=sha(p);original=r['signature'];r['signature']=signature(t)
        r['metadata_only_signature_migration']=dict(original_receipt_sha256=receipt_sha,original_signature=original,audit='audits/monitor_repair/migration.json',numerical_code_unchanged=True)
        dump(p,r);assert valid(t)
        migrated.append(dict(task=task_id(t),old_receipt_sha256=receipt_sha,old_signature=original,new_signature=r['signature'],status=r['status'],artifacts=r['artifacts']))
    dump(ROOT/'audits/monitor_repair/migration.json',dict(status='PASS',cause='heartbeat and scheduler used same PID-only temporary name',fix='unique PID plus nanosecond temporary names',atomic_writers=4,writes_each=100,stress_test='PASS',old_code_hashes=old,new_code_hashes=now,changed_numerical_code=[],input_and_output_hashes_verified=True,migrated=migrated,already_valid=already,computations_repeated=0))
    print('METADATA MIGRATION PASS',len(migrated),'already valid',len(already),flush=True)
if __name__=='__main__':main()
