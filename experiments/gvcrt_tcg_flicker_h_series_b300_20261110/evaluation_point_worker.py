"""Resource-only wrapper: immutable states share one frozen source/QP cache."""
from common_s2 import *
import gc

def main(spec_path):
 env()
 import torch
 from evaluation import Evaluator
 from cache_codec import cache_paths
 from pipeline import ready
 spec=load(spec_path);phase=spec['phase'];v=spec['video'];q=spec['QP'];row=v.get('source_row',v)
 if phase=='final':assert load(ROOT/'selection.json')['status']=='FROZEN'
 assert len(spec['tasks'])==len({(t['model'],t['step']) for t in spec['tasks']})
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0);key,path,rp=cache_paths(row,q,False)
 assert os.environ.get('H_CACHE_LEASE_HELD')==key and ready(row,q,False,True)
 proof=load(rp);data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);evaluator=Evaluator(0);records=[];begin=time.time()
 for task in spec['tasks']:
  a=evaluator.point(v,q,task['model'],task['step'],phase,data,proof);records.append(dict(model=task['model'],step=a['identity']['step'],metric_id=a['identity']['metric_id'],checkpoint_SHA256=a['identity']['checkpoint_SHA256'],stream_SHA256=a['identity']['stream_SHA256'],seconds=a['seconds']))
  evaluator.contexts.clear();gc.collect();torch.cuda.empty_cache()
  dump(ROOT/'evaluation_queue/worker_progress'/Path(spec_path).name,dict(status='RUNNING',physical_GPU=physical_gpu(0),completed=len(records),expected=len(spec['tasks']),last=records[-1],time=time.time()))
 if path.is_relative_to(ROOT):os.utime(path,None)
 dump(ROOT/'audits/evaluation_bundles'/Path(spec_path).name,dict(status='PASS',physical_GPU=physical_gpu(0),PID=os.getpid(),phase=phase,point=spec['index'],records=records,numerical_evaluation_SHA256=sha(ROOT/'evaluation.py'),seconds=time.time()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),cache_SHA256=proof['cache_sha256'],time=time.time()))

if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception as error:
  from gpu_resources import failure_record
  dump(ROOT/'audits/evaluation_bundles'/Path(sys.argv[1]).name,failure_record(error,spec=load(sys.argv[1])))
  raise
