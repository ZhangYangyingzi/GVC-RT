"""Verify every frozen historical dependency, without editing any source."""
from io32 import *
import concurrent.futures

def main():
 h=load(ROOT/'source_hashes.json')['hashes']
 def one(x):
  p,expected=x;actual=sha(p);return dict(path=p,expected_sha256=expected,actual_sha256=actual,PASS=actual==expected)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,h.items()))
 bad=[r for r in rows if not r['PASS']];dump(ROOT/'audits/frozen_dependencies.json',dict(status='PASS' if not bad else 'FAIL',checked=len(rows),mismatches=bad,dependencies=rows));assert not bad,bad
 print('FROZEN DEPENDENCIES PASS',len(rows),flush=True)
if __name__=='__main__':main()
