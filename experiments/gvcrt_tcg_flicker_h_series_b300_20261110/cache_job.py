from common_s2 import *
from cache_codec import ensure
if __name__=='__main__':
 env();a=load(sys.argv[1]);ensure(a['row'],a['qp'],a['training'],0)
