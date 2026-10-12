from runtime import *
import argparse,traceback
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--keys',nargs='+',required=True);a=p.parse_args();device=init(a.gpu)
for key in a.keys:
 try:torch.cuda.reset_peak_memory_stats();buildcache(key,device);print('CACHE PASS',key,flush=True)
 except Exception:dump(ROOT/'logs'/f'cache_failure_{key}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
