from io21 import *
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 import torch
 from window_training import replay
 from gradient_tools import dists_details,stats
 torch.set_num_threads(2);v4,im,pm,models,quality=load_training(torch.device('cuda'));common=module('v629_deep_probe_qp',V62/'fullqp_common.py');row=load(ROOT/'plans/calibration.json')['updates'][0];frames=replay(row,'B_covstats',torch.device('cuda'));wrapper=models['wrapper'];q=row['external_qp'];caps=[];hook=pm.dec.register_forward_hook(lambda m,a,o:caps.append(o));core=v4.compression_hash(im,pm)
 pm.clear_dpb();pm.set_curr_poc(0)
 with torch.no_grad():
  pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),q)['x_hat'])
  for pos in range(1,49):
   rec,_,_=v4.joint_ste_forward(pm,wrapper(frames[pos]),common.frame_qp(pm,q,pos));pm.add_ref_frame(caps.pop().detach(),(rec.detach()*2-1).half())
 rec,_,_=v4.joint_ste_forward(pm,wrapper(frames[49]),common.frame_qp(pm,q,49))
 from stat_loss import dists_and_stat
 target=frames[49].detach().clone().requires_grad_(True)
 score,loss,terms,shapes=dists_and_stat(quality[1],rec,target)
 gx,gt=torch.autograd.grad(loss*(4096/15),[rec,target],retain_graph=True,allow_unused=True)
 assert gx is not None and bool(torch.isfinite(gx).all()) and bool((gx!=0).any()) and gt is None
 restored=gx.detach().float()/4096
 rows={}
 for k in ('bridge','generator'):
  ps=list(models[k].parameters());gs=torch.autograd.grad(loss*(4096/15),ps,retain_graph=True,allow_unused=True)
  gs=[g.float()/4096 if g is not None else None for g in gs];st=stats(gs,ps);assert st['finite'] and st['norm']>0;rows[k]=st
 assert core==v4.compression_hash(im,pm) and all(p.grad is None for m in quality for p in m.parameters());hook.remove()
 dump(ROOT/'audits/stat_image_GT_gradient_probe.json',dict(status='PASS',GT_gradient_is_None=True,GT_covariance_and_denominator_detached=True,rec_gradient_norm=float(restored.double().square().sum().sqrt()),rec_gradient_dtype=str(restored.dtype),rec_gradient_finite=True,modules=rows,unit_scale=4096,frame_denominator=15,shapes=shapes,physical_gpu=a.gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,no_optimizer_update=True))
 print('STAT IMAGE AND GT GRADIENT PASS',flush=True)
if __name__=='__main__':main()
