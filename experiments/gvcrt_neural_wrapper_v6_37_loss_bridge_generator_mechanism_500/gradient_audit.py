from io32 import *
import argparse

def main(gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 import training32 as tr
 from frames32 import PlanFrames
 from train import basic_crop,PARENT_N
 ctx=tr.setup('N_noP_BG_FD');v4,im,pm,models,quality,common,params,frozen=ctx
 source=torch.load(PARENT_N/'checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True);allmods=dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1])
 def reset():
  for k in ('wrapper','bridge','generator'):models[k].load_state_dict(source[k],strict=True)
  tr.restore_buffers(allmods,{k:v for k,v in source['buffers'].items() if k in allmods});tr.restore_rng(source['crop_rng'])
 reset();before={k:v4.module_hash(m) for k,m in models.items()};row=load(ROOT/'plans/shared.json')['updates'][0]['source_row']
 frames=PlanFrames(row,'cuda:0');grads,d=basic_crop(ctx,frames,row['external_qp'],audit=True);frames.close()
 for comp in ('LPIPS','weighted_reconstruction','weighted_DISTS'):
  assert all(d['component_gradient_modules'][comp][m]['finite'] and d['component_gradient_modules'][comp][m]['norm']>0 for m in ('bridge','generator'))
 assert all(d['component_gradient_modules']['rate_proxy'][m]['norm']==0 for m in ('bridge','generator'))
 assert len(d['deep_gradient_checks'])==15 and all(x['proxy_L1']==0 for x in d['trace'])
 reset();basic=module('v637_readonly_B_crop',V36/'basic_train.py');frames=PlanFrames(row,'cuda:0');_,b=basic.basic_crop(ctx,frames,row['external_qp']);frames.close()
 common_keys=('LPIPS','rate_bpp','beta_rate','proxy_L1','weighted_proxy_L1','reconstruction_L1','weighted_reconstruction_L1')
 bc={k:abs(d['losses'][k]-b['losses'][k]) for k in common_keys};assert max(bc.values())<=1e-7,bc
 reset();frames=PlanFrames(row,'cuda:0');_,f,ft=tr.crop_c1(ctx,frames,row['external_qp']);frames.close()
 fc={k:abs(d['losses'][k]-f[k]) for k in common_keys+('DISTS','weighted_DISTS')};assert max(fc.values())<=1e-7,fc
 reset();assert before=={k:v4.module_hash(m) for k,m in models.items()};tr.check_frozen(ctx)
 dump(ROOT/'audits/step0_gradient.json',dict(status='PASS',new_optimizer_updates=0,scope='separate read-only process; formal training restores step0 again',source_sha256=sha(PARENT_N/'checkpoints/adaptation_step_0.pt'),modules=d['component_gradient_modules'],combined_modules=tr.groups(models,grads),combined_norm=tr.norm(grads),DISTS_deep_RGB=d['deep_gradient_checks'],common_B_abs_error=bc,DISTS_F_and_common_abs_error=fc,D_loss=d['losses'],B_loss=b['losses'],F_C1_loss=f,quality_frozen=True,frozen_hashes=frozen,initial_module_hashes=before,rate_gradient_note='receiver-only rate/proxy have zero BG gradient; original formula retained',physical_GPU=gpu))
 print('STEP0_GRADIENT_PASS',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);main(a.parse_args().gpu)
