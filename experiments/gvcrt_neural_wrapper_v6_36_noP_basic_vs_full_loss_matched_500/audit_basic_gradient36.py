import os,argparse,json,hashlib,math,time
from pathlib import Path
from io32 import *
def main(gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,torch.nn.functional as F
 from training32 import setup,restore_rng,buffer_state,restore_buffers,groups,norm,seed
 from frames32 import PlanFrames
 ctx=setup('N_noP_BG_FD');v4,im,pm,models,quality,common,params,frozen=ctx
 source_path=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
 source=torch.load(source_path,map_location='cpu',weights_only=True)
 for k in ('wrapper','bridge','generator'):models[k].load_state_dict(source[k],strict=True)
 allmods=dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1]);restore_buffers(allmods,{k:v for k,v in source['buffers'].items() if k in allmods});restore_rng(source['crop_rng'])
 before={k:v4.module_hash(m) for k,m in models.items()};frozen_before=frozen
 item=load(ROOT/'plans/shared.json')['updates'][0];row=item['source_row'];frames=PlanFrames(row,'cuda:0');params=[p for p in params if p.requires_grad]
 lp_acc=[torch.zeros_like(p) for p in params];rec_acc=[torch.zeros_like(p) for p in params];rate_acc=[torch.zeros_like(p) for p in params];total_acc=[torch.zeros_like(p) for p in params]
 cap=[];hook=pm.dec.register_forward_hook(lambda _m,_a,o:cap.append(o));trace=[]
 try:
  pm.clear_dpb();pm.set_curr_poc(0)
  with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),row['external_qp'])['x_hat'])
  beta=common.beta_q('schedule_s1p0',row['external_qp'])
  for pos in range(1,64):
   gt=frames[pos];qa=common.frame_qp(pm,row['external_qp'],pos)
   with torch.enable_grad() if pos>=49 else torch.no_grad():
    proxy=models['wrapper'](gt);rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
    if pos>=49:
     lp=quality[0](rec,gt,normalize=True).mean();rl=F.l1_loss(rec,gt);pl=F.l1_loss(proxy,gt);rb=rate/65536
     scalars={'lpips':lp/15,'reconstruction_weighted':rl*(4096/15),'rate_weighted':beta*rb/15}
     grads={}
     for name,val in scalars.items():
      if val.requires_grad:grads[name]=torch.autograd.grad(val,params,retain_graph=True,allow_unused=True)
      else:grads[name]=tuple(None for _ in params)
     for j,p in enumerate(params):
      gl=grads['lpips'][j];gr=grads['reconstruction_weighted'][j];gt_rate=grads['rate_weighted'][j]
      if gl is not None:lp_acc[j].add_(gl.detach().float())
      if gr is not None:rec_acc[j].add_(gr.detach().float()/4096*.1383589248954108)
      if gt_rate is not None:rate_acc[j].add_(gt_rate.detach().float())
      if gl is not None:total_acc[j].add_(gl.detach().float())
      if gr is not None:total_acc[j].add_(gr.detach().float()/4096*.1383589248954108)
      if gt_rate is not None:total_acc[j].add_(gt_rate.detach().float())
     trace.append({'position':pos,'actual_qp':qa,'lpips':float(lp.detach()),'reconstruction_l1':float(rl.detach()),'rate_bpp':float(rb.detach()),'proxy_l1':float(pl.detach()),'rate_requires_grad':bool(rb.requires_grad)})
   assert len(cap)==1
   with torch.no_grad():pm.add_ref_frame(cap.pop().detach(),(rec.detach()*2-1).half())
 finally:hook.remove();frames.close()
 def mod_norm(x):return groups(models,x)
 after={k:v4.module_hash(m) for k,m in models.items()};assert before==after
 check_frozen=__import__('training32').check_frozen;check_frozen(ctx)
 data={'status':'PASS','purpose':'read-only step0 component-gradient and rate-path audit; no optimizer update','source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),'plan_row_sha256':item['source_row_canonical_sha256'],'plan_source_rgb_hashes':row['full_RGB_sha256'],'supervised_frames':len(trace),'LPIPS_gradient_norm':norm(lp_acc),'weighted_reconstruction_gradient_norm':norm(rec_acc),'weighted_rate_gradient_norm':norm(rate_acc),'combined_gradient_norm':norm(total_acc),'modules':{'LPIPS':mod_norm(lp_acc),'weighted_reconstruction':mod_norm(rec_acc),'weighted_rate':mod_norm(rate_acc),'combined':mod_norm(total_acc)},'rate_requires_grad_per_frame':[x['rate_requires_grad'] for x in trace],'rate_module_gradient_nonzero':{k:v['norm'] for k,v in mod_norm(rate_acc).items()},'proxy_L1_values':[x['proxy_l1'] for x in trace],'initial_module_hashes':before,'final_module_hashes':after,'frozen_hashes':frozen_before,'freeze_unchanged':True,'optimizer_updates':0,'trace':trace,'physical_GPU':gpu,'completed_unix':time.time()}
 dump(ROOT/'audits/basic_objective_gradient_audit.json',data);print('BASIC_GRADIENT_AUDIT_PASS',data['LPIPS_gradient_norm'],data['weighted_reconstruction_gradient_norm'],data['weighted_rate_gradient_norm'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);main(a.parse_args().gpu)
