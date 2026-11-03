"""Same fixed train/evaluate orchestration with parallel frozen-flow workers."""
import inspect,sys
import runner as original
from common_expanded import *
if __name__=='__main__':
 source=inspect.getsource(original.main)
 assert source.count("str(ROOT/'train.py')")==1
 source=source.replace("str(ROOT/'train.py')","str(ROOT/'train_parallel.py')")
 namespace=dict(original.__dict__)
 exec(compile(source,str(ROOT/'runner_parallel.py'),'exec'),namespace)
 sys.exit(namespace['main']())
