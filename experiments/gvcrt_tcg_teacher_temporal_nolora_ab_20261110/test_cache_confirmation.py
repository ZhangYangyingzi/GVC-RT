"""A completed cache must verify after transient reader-lock contention."""
from common_s2 import *

def main():
    env();import pipeline
    base=ROOT/'tmp/cache_confirmation_test';base.mkdir(exist_ok=True);key='SYNTHETIC_confirmation_lock';p=base/'tensor.pt';rp=base/'receipt.json';signal=base/'lock_held.json'
    if signal.exists():signal.unlink()
    p.write_bytes(b'complete');dump(rp,dict(status='PASS',cache_identity=key,cache_bytes=8,cache_sha256=sha(p)))
    pipeline.cache_paths=lambda row,qp,training:(key,p,rp)
    script="""from common_s2 import *
env()
with lock('locks/derived_SYNTHETIC_confirmation_lock.lock',blocking=True):
 dump(ROOT/'tmp/cache_confirmation_test/lock_held.json',dict(status='SYNTHETIC_LOCK_HELD'))
 time.sleep(1)
"""
    child=subprocess.Popen([PYTHON,'-B','-c',script],cwd=ROOT);start=time.time()
    while not signal.exists():assert time.time()-start<10;time.sleep(.01)
    assert not pipeline.ready({},0,False), 'POLL_SHOULD_REPORT_BUSY'
    begin=time.time();ok=pipeline.ready({},0,False,True);elapsed=time.time()-begin;assert child.wait()==0
    assert ok,'COMPLETED_CACHE_WRONGLY_FAILED_DUE_TO_TRANSIENT_LOCK'
    dump(ROOT/'audits/cache_confirmation_regression.json',dict(status='PASS',synthetic_only=True,nonblocking_poll_reports_busy=True,strict_confirmation_waits_for_lock_and_checks_SHA=True,confirmation_seconds=elapsed,hash_checks_not_relaxed=True,time=time.time()))
    print('CACHE_CONFIRMATION_REGRESSION_PASS',elapsed)
if __name__=='__main__':main()
