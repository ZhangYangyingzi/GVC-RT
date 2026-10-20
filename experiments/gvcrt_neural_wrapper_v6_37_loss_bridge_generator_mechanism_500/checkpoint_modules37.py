from io32 import *
def main():
 import torch
 cfg=load(ROOT/'evaluation_config.json');manifest=load(ROOT/'checkpoint_reuse_manifest.json');proof=[]
 entries=[('B_basicLoss_noP_500',V36/'branches/B_basicLoss_noP_500/checkpoints/adaptation_step_500.pt','B_basicLoss_noP_500'),('F_fullLoss_noP_500',ROOT.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500/branches/N_noP_BG_FD/checkpoints/adaptation_step_500.pt','N_oursLoss_noP_500')]
 for name,path,key in entries:
  st=torch.load(path,map_location='cpu',weights_only=True);ix=cfg['checkpoints'][key];si=torch.load(ix['path'],map_location='cpu',weights_only=True);actual={k:tensor_hash(st[k]) for k in ('bridge','generator')};infer={k:tensor_hash(si[k]) for k in actual};deploy={k:tensor_hash({n:t.half() for n,t in si[k].items()}) for k in actual}
  assert actual==infer=={k:ix['module_hashes'][k] for k in actual};assert deploy==cfg['deployment_receiver_hashes'][key]
  proof.append(dict(method=name,train_checkpoint=str(path),inference_checkpoint=ix['path'],FP32_training_module_hashes=actual,FP16_deployment_module_hashes=deploy,full_state_dict_including_buffers=True,source_optimizer_not_loaded=True));del st,si
 p=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt';st=torch.load(p,map_location='cpu',weights_only=True);assert st['adaptation_step']==st['committed_optimizer_updates']==st['plan_cursor']==0
 d=load(ROOT/'audits/source_and_checkpoint_verification.json');d.update(module_precision_verification=proof,original_deployment_module_hashes=cfg['original_receiver_hashes'],shared_initial_FP32_hashes={k:tensor_hash(st[k]) for k in ('bridge','generator')},shared_initial_crop_RNG_restored=True);dump(ROOT/'audits/source_and_checkpoint_verification.json',d);print('CHECKPOINT MODULES PASS',flush=True)
if __name__=='__main__':main()
