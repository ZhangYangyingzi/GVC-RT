from preflight import *
import argparse
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from runtime import Sender
 rows=[]
 for kind,shape in [('B_rgb_P',(1,3,256,256)),('C_latent_P',(1,256,16,16)),('B_rgb_P',(1,3,1080,1920)),('C_latent_P',(1,256,68,120))]:
  torch.manual_seed(20261015);m=Sender(kind).cuda().eval();x=torch.ones(shape,device='cuda');torch.cuda.reset_peak_memory_stats();elapsed=[]
  with torch.inference_mode():
   for i in range(13):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();m(x);b.record();b.synchronize()
    if i>=3:elapsed.append(a.elapsed_time(b))
  rows.append(dict(branch=kind,input_shape=shape,parameters=sum(p.numel() for p in m.parameters()),FLOPs_conv_only=sum(2*t.weight.numel()*shape[-2]*shape[-1] for t in m.modules() if isinstance(t,torch.nn.Conv2d)),median_ms=float(np.median(elapsed)),samples=10,warmup=3,precision='FP32',peak_gpu_MiB=torch.cuda.max_memory_reserved()/2**20));del m,x;torch.cuda.empty_cache()
 save('audits/complexity.json',dict(rows=rows,FLOPs_convention='2 per convolution MAC; excludes bias, residual adds, ReLU/clamp; standalone module only; timings do not imply equal whole-system cost',gpu=gpu,optimizer_updates=0))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);main(a.parse_args().gpu)
