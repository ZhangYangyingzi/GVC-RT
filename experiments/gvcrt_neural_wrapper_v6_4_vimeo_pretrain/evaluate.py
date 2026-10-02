"""Invoke the frozen V6.2-B evaluator with output paths confined to V6.4."""
import traceback
from v64_io import *
if __name__=='__main__':
    try:eval_adapter().main()
    except Exception:dump(ROOT/'logs'/f'evaluation_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
