"""Synthetic cache generations: a delayed receipt must not become READY."""
from common_s2 import *

def main():
    env();import pipeline
    base=ROOT/'tmp/cache_publication_test';base.mkdir(exist_ok=True);key='SYNTHETIC_publication_lease';p=base/'tensor.pt';rp=base/'receipt.json';signal=base/'tensor_published.json'
    if signal.exists():signal.unlink()
    p.write_bytes(b'oldcache');dump(rp,dict(status='PASS',cache_identity=key,cache_bytes=8,cache_sha256=sha(p)))
    pipeline.cache_paths=lambda row,qp,training:(key,p,rp)
    script="""from common_s2 import *
env();key='SYNTHETIC_publication_lease';base=ROOT/'tmp/cache_publication_test';p=base/'tensor.pt';rp=base/'receipt.json'
with lock('locks/lease_'+key+'.lock',blocking=True),lock('locks/derived_'+key+'.lock',blocking=True):
    temp=base/'new.tmp';temp.write_bytes(b'newcache');temp.replace(p)
    dump(base/'tensor_published.json',dict(status='SYNTHETIC_TENSOR_ONLY'))
    time.sleep(2)
    dump(rp,dict(status='PASS',cache_identity=key,cache_bytes=8,cache_sha256=sha(p)))
"""
    w=subprocess.Popen([PYTHON,'-B','-c',script],cwd=ROOT)
    begin=time.time()
    while not signal.exists():assert time.time()-begin<10;time.sleep(.02)
    assert not pipeline.ready({},0,False),'INCOMPLETE_PAIR_VISIBLE_AS_READY'
    try:
        with lock('locks/lease_'+key+'.lock'):raise AssertionError('CONSUMER_OR_EVICTOR_ENTERED_DURING_PUBLICATION')
    except BlockingIOError:pass
    assert w.wait()==0
    assert pipeline.ready({},0,False,True)
    with lock('locks/lease_'+key+'.lock'):assert sha(p)==load(rp)['cache_sha256']
    dump(ROOT/'audits/cache_publication_regression.json',dict(status='PASS',synthetic_only=True,delayed_receipt_not_READY=True,consumer_and_eviction_excluded_during_publish=True,completed_pair_SHA_verified=True,hash_checks_preserved=True,time=time.time()))
    print('CACHE_PUBLICATION_REGRESSION_PASS')
if __name__=='__main__':main()
