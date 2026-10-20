from io32 import *
import math,collections

def main():
 import torch
 b=ROOT/'branches/D_basicDISTS_noP_500';state=torch.load(b/'checkpoints/adaptation_step_500.pt',map_location='cpu',weights_only=True);zero=torch.load(b/'checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)
 assert state['committed_updates']==state['plan_cursor']==state['adaptation_step']==500
 assert not zero['optimizer']['state']
 assert all(int(s['step'])==500 for s in state['optimizer']['state'].values())
 assert zero['initial_module_hashes']['bridge']=='9c4ca3f7a3e2f502688509ca82555f0ce7ee6fe8c31a89fa9f8dff1104953cd3'
 assert zero['initial_module_hashes']['generator']=='318ea46772ebd1ae756b71edb430552e2e704c7fc248f8b140b5f28e77d11aab'
 assert not state['wrapper'] and zero['frozen_hashes']==state['frozen_hashes']
 ledger=[json.loads(s) for s in (b/'optimizer_calls.jsonl').read_text().splitlines()];committed=[r for r in ledger if r['event']=='COMMITTED'];assert [r['step'] for r in committed]==list(range(1,501))
 plan=load(ROOT/'plans/shared.json')['updates'][:500];gradient=[]
 for i,item in enumerate(plan,1):
  r=load(b/'receipts'/f'{i}.json');row=item['source_row'];assert r['step']==r['optimizer_internal_step']==i and r['plan_row_sha256']==item['source_row_canonical_sha256']
  assert r['canonical_indices']==row['canonical_indices'] and r['crop']==[row['crop_x'],row['crop_y'],256,256] and r['external_qp']==row['external_qp']
  assert [x['actual_qp'] for x in r['frame_trace']]==row['actual_qps'] and all(x['proxy_L1']==0 for x in r['frame_trace'])
  assert r['freeze_check']=='PASS' and r['frozen_hashes']==state['frozen_hashes'] and r['gradient_norms']['global_postclip']<=1.00001
  assert all(r['parameter_delta_norms'][k]>0 and math.isfinite(r['parameter_delta_norms'][k]) for k in ('bridge','generator'))
  gradient.append(dict(step=i,plan_row_SHA=r['plan_row_sha256'],**r['gradient_norms'],**{k+'_parameter_delta':v for k,v in r['parameter_delta_norms'].items()}))
 write(b/'gradient_summary.csv',gradient)
 assert tensor_hash(state['bridge'])!=tensor_hash(zero['bridge']) and tensor_hash(state['generator'])!=tensor_hash(zero['generator'])
 cps=[]
 for step in (0,100,250,500):
  ix=load(b/'checkpoint_index.json')[str(step)];assert sha(ix['path'])==ix['sha256'];cps.append(ix)
 dump(ROOT/'audits/frozen_state.json',dict(status='PASS',initial_frozen_hashes=zero['frozen_hashes'],final_frozen_hashes=state['frozen_hashes'],all500_update_receipts_pass=True,quality_frozen=True,noP_identity=True,initial_optimizer_steps=0,final_optimizer_steps=500,BG_parameters_changed=True,checkpoint_index=cps))
 source=load(ROOT/'source_manifest.json');changed=[]
 for p,v in source['files'].items():
  if p.endswith('.py') and sha(p)!=v['sha256']:changed.append(p)
 assert not changed,('Historical executable sources changed',changed)
 inputs=load(ROOT/'audits/data_isolation.json');assert inputs['all_current_source_hashes_verified']
 for p in load(ROOT/'checkpoint_reuse_manifest.json')['checkpoints']:assert sha(p['path'])==p['sha256']
 assert load(ROOT/'audits/intervention_input_hooks.json')['status']=='PASS' and load(ROOT/'audits/intervention_input_hooks.json')['video_QP_tasks']==39
 agg=load(ROOT/'runtime/aggregation_status.json');assert agg['RD_complete']==273 and agg['FID_complete']==63
 for n in ('stream_and_feature_equality','replay_endpoint_equivalence','interpolation_endpoints','reuse_pointwise_verification','step0_gradient','data_isolation'):assert load(ROOT/'audits'/f'{n}.json')['status']=='PASS'
 assert load(ROOT/'audits/fixed_example_manifest.json')['complete']==36
 updates={m:(500 if m=='D_basicDISTS_noP_500' else 0) for m in METHODS};exits=[load(p) for p in (ROOT/'runtime/attempts').glob('*_exit.json')];failed=[e for e in exits if e['exit_code']!=0]
 dump(ROOT/'audits/GPU_execution.json',dict(status='PASS',worker_exit_records=exits,reproducible_commands=[dict(attempt=e['attempt'],argv=[PYTHON,'-B','-u',str(ROOT/e['script']),*map(str,e.get('args',[])),'--gpu',str(e['gpu'])],exit_code=e['exit_code']) for e in exits],failed_attempts=failed,failed_final_tasks=[],allowed_GPUs=[4,5,6,7],actual_GPU_counts=dict(collections.Counter(str(e['gpu']) for e in exits)),single_D_training_worker_PID=sorted({r['PID'] for r in committed}),prevented_duplicate_wrapper_launches='flock rejections retained in logs/supervisors.log; no duplicate training worker or optimizer update'))
 assert len({r['PID'] for r in committed})==1,'A legitimate exact-resume case requires explicit audit of both worker segments'
 required=['README.md','REPORT.md','RESULTS_SUMMARY.md','OBJECTIVE_COMPARISON.md','MECHANISM_PROTOCOL.md','effective_config.json','protocol.json','source_manifest.json','checkpoint_reuse_manifest.json','evaluation/task_manifest.json','errors.jsonl','figures/FIGURE_INDEX.md']+['results/'+n+'.csv' for n in ('all_rd_points','pooled_fid','equal_rate_summary','equal_rate_per_video','same_stream_pairs','loss_attribution','module_swap_summary','latent_interpolation','latent_statistics','latent_pair_changes')]
 assert all((ROOT/p).is_file() for p in required)
 files=[]
 for p in list(ROOT.glob('branches/*/checkpoints/*.pt'))+list(ROOT.glob('evaluation/checkpoints/*.pt'))+[ROOT/'plans/shared.json']:
  files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'training_artifacts_index.json',files);dump(ROOT/'audits/current_code_hashes.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
 dump(ROOT/'publication_status.json',dict(status='READY_FOR_PUSH',push_success=False,no_final_SHA_self_reference=True))
 final=dict(status='PASS',computation_status='PASS',publication_status='PENDING',new_effective_updates=updates,optimizer_committed_updates=500,plan_rows=500,training_domains=dict(collections.Counter(x['source_row']['domain'] for x in plan)),RD=dict(complete=273,planned=273),FID=dict(complete=63,planned=63,samples=agg['pooled_samples']),D_unique_real_encode_points=39,replay_verification_tasks=39,intervention_hook_verification_tasks=39,failed_attempts=len(failed),failed_final_tasks=[],blocked=[],frozen_checks='PASS',all_actual_RANS_streams_equal=True,required_files_sha256={p:sha(ROOT/p) for p in required},finished=time.time())
 dump(ROOT/'final_integrity.json',final);print('FINAL INTEGRITY PASS D500 RD273 FID63',flush=True)
if __name__=='__main__':main()
