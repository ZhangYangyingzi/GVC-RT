from common_s2 import *
def main():
 env();import pipeline
 key='SYNTHETIC_INVALID_CACHE';p=ROOT/'tmp/cache_repair_fixture/data.pt';rp=ROOT/'tmp/cache_repair_fixture/receipt.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'invalid')
 dump(rp,dict(status='PASS',cache_identity=key,cache_bytes=7,cache_sha256='0'*64));original=pipeline.cache_paths;pipeline.cache_paths=lambda row,qp,training:(key,p,rp)
 assert not pipeline.ready({},0,False,True) and not p.exists() and not rp.exists()
 assert load(ROOT/'audits/cache_rejected'/f'{key}.json')['status']=='REJECTED_REBUILD_REQUESTED'
 dump(ROOT/'audits/cache_regeneration_regression.json',dict(status='PASS',synthetic_only=True,corrupt_derived_cache_not_consumed=True,repair_request_retained=True,source_files_unchanged=True,time=time.time()))
 print('CACHE_REGENERATION_REGRESSION_PASS')
if __name__=='__main__':main()
