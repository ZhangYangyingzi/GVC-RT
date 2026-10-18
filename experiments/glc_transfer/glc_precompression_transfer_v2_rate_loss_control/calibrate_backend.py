from preflight import *
from data import load
def main():
 import numpy as np
 from run_pipeline import parallel
 parallel([(f'calibrate_{i:02d}',['calibrate_backend.py','--index',str(i)]) for i in range(40) if not (R/f'calibration/rows/{i:02d}.json').exists()],need=7000,max_workers=4,per_gpu=2)
 rows=[load(R/f'calibration/rows/{i:02d}.json') for i in range(40)]
 beta={str(q):.1*float(np.median([r['ratio'] for r in rows if r['q']==q])) for q in load(R/'protocol.json')['q_grid']}
 save('calibration/final.json',dict(status='PASS',beta=beta,optimizer_updates=0,training_windows=40,measurements=40,rule='.1 * median(||g_LPIPS+.5DISTS+lambda_recL1|| / ||g_common_rate||), same source C sender, training only',frozen_unchanged=True,common_to_all_four_branches=True))
def worker(index,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from runtime import models,quality,Sender
 from train import gradients,norm,seed
 from rate_backend import install
 from src.entropy_coder import GaussianRansCodec
 from bootstrap import V1
 seed(20261016);im,pm=models('cuda:0');ql=quality('cuda:0');fixed=[state_hash(m.state_dict()) for m in (im,pm,*ql)];p=Sender('C_latent_P').cuda();p.load_state_dict(torch.load(V1/'final_P/C_latent_P_1000.pt',map_location='cpu',weights_only=False)['P'],strict=True);before=state_hash(p.state_dict());install(pm,load(R/'audits/rate_backend_decision.json')['backend'],GaussianRansCodec())
 r=load(R/'plans/rate_audit.json')['rows'][index];values,trace,a,b=gradients(im,pm,p,ql,r)
 assert norm(a)>0 and norm(b)>0 and all(torch.isfinite(g).all() for g in a+b)
 assert before==state_hash(p.state_dict()) and fixed==[state_hash(m.state_dict()) for m in (im,pm,*ql)]
 save(f'calibration/rows/{index:02d}.json',dict(status='PASS',index=index,domain=r['domain'],q=r['q'],quality_gradient_norm=norm(a),rate_gradient_norm=norm(b),ratio=norm(a)/norm(b),values=values,optimizer_updates=0))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--index',type=int);a.add_argument('--gpu',type=int,choices=[4,5,6,7]);x=a.parse_args();main() if x.index is None else worker(x.index,x.gpu)
