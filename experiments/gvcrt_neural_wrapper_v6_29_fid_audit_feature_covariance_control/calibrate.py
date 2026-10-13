"""Independent training-only calibration, no optimizer updates."""
from io21 import *
import argparse,random,statistics,math

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 import torch,numpy as np
 from window_training import replay,make_runner
 from stat_loss import covariance,feature_loss
 torch.set_num_threads(2);frozen();random.seed(20261029);np.random.seed(20261029);torch.manual_seed(20261029)
 v4,im,pm,models,quality=load_training(torch.device('cuda:0'));before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
 run=make_runner(v4,im,pm,models,quality);rows=[]
 out=ROOT/'calibration/samples.json';rows=load(out) if out.exists() else []
 for i,row in enumerate(load(ROOT/'plans/calibration.json')['updates']):
  if i<len(rows):continue
  frames=replay(row,'B_covstats',torch.device('cuda:0'));models['bridge'].zero_grad(set_to_none=True);models['generator'].zero_grad(set_to_none=True)
  sums,qps,trace,warmup,grads=run(frames,row['external_qp'],'B_covstats',mode='calibrate')
  g0,gs=grads['base'],grads['adv'];n0=sums['B1_gradient_norm'];ns=sums['stat_gradient_norm'];assert n0>1e-12 and ns>1e-12
  assert all(bool(torch.isfinite(g).all()) for g in g0+gs)
  off=0;modules={}
  for k,m in models.items():
   n=sum(p.requires_grad for p in m.parameters());modules[k]=dict(g0=sum(float(g.double().square().sum()) for g in g0[off:off+n])**.5,g_stat=sum(float(g.double().square().sum()) for g in gs[off:off+n])**.5);off+=n
  assert all(modules[k]['g_stat']>0 for k in ('bridge','generator'))
  rows.append(dict(index=i,plan_row=row,losses=sums,modules=modules,g0=n0,g_stat=ns,cosine=sums['stat_cosine'],lambda_candidate=.10*n0/(ns+1e-12),actual_qps=qps))
  dump(out,rows);write(ROOT/'calibration/samples.csv',[dict(index=x['index'],domain=x['plan_row']['domain'],QP=x['plan_row']['external_qp'],g0=x['g0'],g_stat=x['g_stat'],cosine=x['cosine'],lambda_candidate=x['lambda_candidate']) for x in rows]);print('CALIBRATION',len(rows),'/40',flush=True)
  assert before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
 assert len(rows)==40
 lam=statistics.median(x['lambda_candidate'] for x in rows)
 dump(ROOT/'calibration/final.json',dict(status='PASS',lambda_stat=lam,samples=40,source_sha256=EXPECTED,plan_sha256=sha(ROOT/'plans/calibration.json'),no_optimizer_update=True,parameters_buffers_unchanged=True,physical_gpu=a.gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('CALIBRATION PASS lambda_stat',lam,flush=True)
if __name__=='__main__':main()
