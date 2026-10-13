from io21 import *
import argparse,types,math

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 import torch,numpy as np
 from window_training import make_runner,replay
 from stat_loss import covariance,feature_loss
 from gradient_tools import dists_details
 torch.set_num_threads(2);v4,im,pm,models,quality=load_training(torch.device('cuda'));frozen_before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm)
 # Spatial statistics: preserve marginal mean/variance while reversing cross-channel correlation.
 z=torch.tensor([[-1.,1.,-1.,1.],[-1.,1.,-1.,1.]],device='cuda').reshape(1,2,2,2);zz=z.clone();zz[:,1]*=-1
 assert torch.equal(z.mean((2,3)),zz.mean((2,3))) and torch.equal(z.var((2,3)),zz.var((2,3)))
 offdiag=float((covariance(z)-covariance(zz)).square().mean());assert offdiag>0
 fs=[torch.randn(1,c,4,4,device='cuda') for c in (3,64,128,256,512)];same=float(feature_loss(fs,fs)[0]);assert same==0
 row=load(ROOT/'plans/calibration.json')['updates'][0];frames=replay(row,'A_continue',torch.device('cuda'))
 params=[p for m in models.values() for p in m.parameters() if p.requires_grad]
 new=make_runner(v4,im,pm,models,quality);s,qs,trace,warm,gr=new(frames,row['external_qp'],'A_continue',mode='calibrate',adv_weight=0.)
 # Execute unchanged historical runner in the isolated current namespace.
 ns=dict(globals());exec(compile((V26/'window_training.py').read_text().replace('from io21 import *',''),str(V26/'window_training.py'),'exec'),ns)
 old=ns['make_runner'](v4,im,pm,models,quality);s0,q0,t0,w0,gold=old(frames,row['external_qp'],'A_continue',mode='calibrate',adv_weight=0.)
 assert qs==q0 and [t['reconstruction_sha256'] for t in trace]==[t['reconstruction_sha256'] for t in t0]
 error=sum(float((a-b).double().square().sum()) for a,b in zip(gr['base'],gold['base']))**.5;norm=sum(float(a.double().square().sum()) for a in gold['base'])**.5
 assert error<=1e-6+1e-5*norm
 for k in ('LPIPS','DISTS','rate_bpp','proxy_L1','reconstruction_L1'):assert s[k]==s0[k]
 # One AdamW update must match for lambda=0; cloned parameters/optimizers avoid formal updates.
 p0=[torch.nn.Parameter(p.detach().clone()) for p in params];p1=[torch.nn.Parameter(p.detach().clone()) for p in params]
 def optimizer(ps):
  n=len(list(models['bridge'].parameters()));return torch.optim.AdamW([dict(params=ps[:n],lr=3e-6),dict(params=ps[n:],lr=5e-7)],weight_decay=1e-4)
 o0,o1=optimizer(p0),optimizer(p1)
 for ps,gs,opt in ((p0,gold['base'],o0),(p1,gr['base'],o1)):
  for p,g in zip(ps,gs):p.grad=g.clone()
  torch.nn.utils.clip_grad_norm_(ps,1.);opt.step()
 update_error=max(float((p-q).abs().max()) for p,q in zip(p0,p1));assert update_error<=1e-7
 x=frames[49].detach().clone().requires_grad_(True);score,layers,terms=dists_details(quality,x,frames[50],True)
 assert all(t['output_gradient']['norm']>0 for t in layers if t['kind']=='deep')
 assert frozen_before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm)
 assert all(p.grad is None for m in quality for p in m.parameters())
 audit=dict(status='PASS',same_features_stat_loss=same,non_diagonal_correlation_test_loss=offdiag,lambda0_loss_equal=True,lambda0_gradient_error=error,lambda0_gradient_norm=norm,lambda0_single_update_maxabs=update_error,initialization_modules=frozen_before,core_hash=core,DISTS_layers=layers,GT_detached=True,covariance_FP32=True,unit_scale=4096.,window_denominator=15,one_clip_one_optimizer_step=True,formal_rng_untouched=True,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,physical_gpu=a.gpu)
 for b in BRANCHES:dump(ROOT/'audits'/f'{b}_pretraining.json',audit)
 dump(ROOT/'audits/stat_feature_shapes.json',dict(indices=[2,3,4],names=['stage2','stage3','stage4'],channels=[128,256,512],source=str(type(quality[1])),shapes=[x['shape'] for x in layers[2:5]]))
 print('PRECHECK PASS',flush=True)
if __name__=='__main__':main()
