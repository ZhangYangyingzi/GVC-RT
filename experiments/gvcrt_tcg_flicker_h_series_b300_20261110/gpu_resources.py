"""Scheduling only: observe free VRAM; never signal another task's processes."""
from common_s2 import *

def free_MiB(gpu):
 assert gpu in [0,1,2,3]
 return int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())

def memory_fits(gpu,kind,cfg):
 minimum=cfg.get('cache_min_free_MiB',26000) if kind=='cache' else cfg.get('evaluation_min_free_MiB',12000)
 return free_MiB(gpu)>=minimum

def failure_record(error,**context):
 import torch,traceback
 return dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),retryable_gpu_OOM=isinstance(error,torch.OutOfMemoryError),physical_GPU=int(os.environ['CUDA_VISIBLE_DEVICES']),PID=os.getpid(),time=time.time(),**context)
