from common_s2 import *
from cache_codec import ensure
if __name__=='__main__':
 env();a=load(sys.argv[1])
 try:ensure(a['row'],a['qp'],a['training'],0)
 except Exception as error:
  from gpu_resources import failure_record
  dump(ROOT/'audits/cache_attempts'/f'{Path(sys.argv[1]).stem}_{os.getpid()}.json',failure_record(error,key=Path(sys.argv[1]).stem))
  raise
