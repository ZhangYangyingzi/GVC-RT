"""Pre-training checks: first two updates count toward the fixed 300-update budget."""
from util import *
import shutil,time

def main(gpu):
 if (P/'preflight.json').exists():
  old=load(P/'preflight.json')
  if old['status']=='PASS' and old['signature']==signature():
   print('PREFLIGHT_ALREADY_PASS; preserve committed updates and step2 audit',flush=True);return
 gpu_env(gpu);sys.path.insert(0,str(P));import train
 train.setup_path()
 from preflight import gradient_checks
 gradient_checks(gpu)
 import evaluate,torch
 from model_io import load_checkpoint,phi_hash
 initial=[];post=[];behavior=[]
 for branch,baseline in [('N1','N0'),('Q1','Q0')]:
  train.train_branch(branch,gpu,2,True)
  # Step0 stays the original common initialization; first two updates are provisional until these checks pass.
  fresh=evaluate.evaluate(branch,0,'UVal0',2,gpu)
  old=evaluate.evaluate(baseline,0,'UVal0',2,gpu,task_dir='initial_baselines')
  assert fresh['stream_sha256']==old['stream_sha256'] and fresh['real_total_bytes']==old['real_total_bytes']
  assert all(abs(fresh[k]-old[k])<1e-8 for k in ['LPIPS','DISTS','FloLPIPS','PSNR'])
  # Also compare to the already audited historical v2 point.
  name={'N0':'A','Q0':'H11'}[baseline]
  history=load(V2/'server_artifacts/evaluation'/f'{name}_s0_UVal0_q2.json')
  assert old['stream_sha256']==history['stream_sha256'] and old['real_total_bytes']==history['real_total_bytes']
  initial.append(dict(method=branch,reference=baseline,all64_initial_actual_bytes_reconstruction_references_exact=True,stream_sha256=fresh['stream_sha256'],historical_task_sha256=sha(V2/'server_artifacts/evaluation'/f'{name}_s0_UVal0_q2.json')))
  p=P/'server_artifacts/preflight_checkpoints'/f'{branch}_step2.pt';p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/'server_artifacts'/branch/'latest.pt',p)
  result=evaluate.evaluate(branch,2,'UVal0',2,gpu,task_dir='post_update_checks',checkpoint_path=p)
  st0=torch.load(P/'server_artifacts'/branch/'step_0.pt',map_location='cpu',weights_only=False);st2=torch.load(p,map_location='cpu',weights_only=False)
  assert all(state_hash(st0['Phi'][n])!=state_hash(st2['Phi'][n]) for n in st0['Phi'])
  assert state_hash(st0['G'])!=state_hash(st2['G'])
  if branch=='Q1':assert state_hash(st0['H'])!=state_hash(st2['H'])
  assert result['probability_encoder']==result['probability_decoder']
  post.append(dict(method=branch,checkpoint_path=str(p),checkpoint_sha256=sha(p),optimizer_updates=2,first_two_updates_count_toward_300=True,all64_reference_symbols_latents_qmaps_synchronized=True,initial_probability= fresh['probability_encoder'],updated_probability=result['probability_encoder'],checkpoint_complete_Phi=True))
  del st0,st2;torch.cuda.empty_cache()
 # A preset global diagnostic multiplier, not per-video transmitted side information or a trained branch.
 for m in [.95,1.,1.05]:
  rr=evaluate.evaluate('Q1',0,'UVal0',2,gpu,task_dir='fixed_multiplier_check',multiplier=m)
  behavior.append(dict(multiplier=m,actual_bpp=rr['actual_bpp'],actual_P_payload_bits=rr['actual_P_payload_bits'],estimated_P_bits=rr['estimated_P_total_bits'],LPIPS=rr['LPIPS'],DISTS=rr['DISTS'],PSNR=rr['PSNR'],stream_sha256=rr['stream_sha256'],symbols_and_full_sequence_reference_checks=True))
 write(P/'multiplier_behavior.csv',behavior)
 dump(P/'audits/initial_full_sequence_identity.json',dict(status='PASS',conditions=initial))
 dump(P/'cdf_consistency_audit.json',dict(status='PASS',post_update_checkpoint_reloads=post,small_multiplier_behavior_check='multiplier_behavior.csv',not_a_finite_difference_STE_derivative_test=True,no_monotonicity_requirement=True))
 dump(P/'preflight.json',dict(status='PASS',signature=signature(),v2_preflight_reused=True,new_probability_gradient_checks='audits/probability_gradient_initial.json',all64_initial_identity=True,post_update_full_Phi_CDF_real_RANS_reload=True,first_two_updates_committed_per_new_group=2,extra_effective_optimizer_updates=0,time=time.time()))
 print('PREFLIGHT_PASS; first 2 updates accepted into each fixed 300-update budget',flush=True)
if __name__=='__main__':main(6)
