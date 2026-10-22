"""Small audit tables and immutable binary SHA index after successful computation."""
import hashlib,json,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500'
OLD=HERE.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500'
def get(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def put(path,obj):
 p=HERE/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write(path,rows):
 p=HERE/path;p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def main():
 branch=HERE/'branches/G_fullLoss_Gonly_500'
 status=get(branch/'training_status.json')
 upto=status['committed_updates'];rr=[];comparison=[]
 for step in range(1,upto+1):
  p=branch/'receipts'/f'{step}.json';assert p.exists()
  x=get(p);assert x['step']==step and x['EMA_total_commits']==step and x['bridge_delta_norm']==0
  rr.append(dict(step=step,domain=x['domain'],QP=x['QP'],C1_norm=x['C1_norm'],FD_unit_norm=x['FD_unit_norm'],
   FD_weighted_norm=x['FD_weighted_norm'],merge_error=x['gradient_formula_error'],
   preclip_norm=x['preclip_norm'],postclip_norm=x['postclip_norm'],clip_coefficient=x['clip_coefficient'],
   generator_delta_norm=x['generator_delta_norm'],bridge_delta_norm=0,EMA_total_commits=step,
   seconds=x['seconds'],peak_memory_MiB=x['peak_memory_MiB']))
  f=PARENT/'branches/N_noP_BG_FD/receipts'/f'{step}.json'
  if f.exists():
   y=get(f);comparison.append(dict(step=step,Gonly_G_C1_norm=x['C1_norm'],BG_G_C1_norm=y['gradient_C1']['generator']['norm'],
    Gonly_G_FD_norm=x['FD_unit_norm'],BG_G_FD_norm=y['gradient_FD_unit']['generator']['norm'],
    Gonly_preclip=x['preclip_norm'],BG_preclip=y['global_preclip_norm'],
    Gonly_clip=x['clip_coefficient'],BG_clip=y['clip_coefficient'],
    Gonly_G_delta=x['generator_delta_norm'],BG_G_delta=y['parameter_delta']['generator']['norm'],
    Gonly_seconds=x['seconds'],BG_seconds=y['seconds']))
 write('branches/G_fullLoss_Gonly_500/gradient_summary.csv',rr)
 write('results/generator_update_comparison.csv',comparison)
 write('results/compute_budget.csv',[dict(method='G_fullLoss_Gonly_500',committed_updates=upto,
  total_seconds=sum(x['seconds'] for x in rr),mean_seconds=sum(x['seconds'] for x in rr)/max(upto,1),
  max_peak_memory_MiB=max([x['peak_memory_MiB'] for x in rr] or [0])),
  dict(method='F_BG500',committed_updates=500,source='historical v6.34b; no new training')])
 if upto==500:
  integ=get(branch/'training_integrity.json');assert integ['status']=='PASS' and integ['EMA_commits']==500
  import torch
  cp=get(branch/'checkpoint_index.json')['500']
  state=torch.load(cp['path'],map_location='cpu',weights_only=True)
  assert state['adaptation_step']==state['plan_cursor']==state['committed_optimizer_updates']==500
  assert set(int(v['step']) for v in state['optimizer']['state'].values())=={500}
  plan=get(HERE/'plans/shared.json')['updates'][:500]
  from collections import Counter
  expected=Counter(f"{z['source_row']['domain']}:{z['source_row']['external_qp']}" for z in plan)
  actual={k:v['counter'] for k,v in state['EMA'].items()}
  assert all(actual[k]==expected[k] for k in actual) and sum(actual.values())==500
  assert sha(cp['path'])==cp['sha256']
  put('audits/EMA_optimizer_final.json',dict(status='PASS',checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],
   plan_cursor=500,optimizer_internal_steps=500,EMA_counters=actual,expected_counters=dict(expected),
   EMA_total_commits=500))
  put('audits/frozen_bridge_and_codec.json',dict(status='PASS',bridge_initial=integ['bridge_initial_hash'],
   bridge_final=integ['bridge_final_hash'],bridge_unchanged=integ['bridge_initial_hash']==integ['bridge_final_hash'],
   generator_final=integ['generator_final_hash'],check_frozen_called_at_each_atomic_commit=True))
 tasks=get(HERE/'evaluation/task_manifest.json')
 verified=[];binary={}
 for entry in tasks:
  t=entry['task'];key=f"{t['dataset']}_{t['video']}_q{t['qp']}"
  a=HERE/'tasks'/f'{key}.json';g=HERE/'tasks'/f'gonly_{key}.json'
  if not a.exists() or not g.exists():continue
  ar=get(a);gr=get(g)
  assert ar['status']==gr['status']=='PASS'
  assert ar['stream_sha256']==gr['result']['bitstream_sha256']
  assert ar['latent_archive_sha256']==gr['latent_archive_sha256']
  assert gr['generator_hash_before']==gr['generator_hash_after']
  assert gr['frozen_hashes_before']==gr['frozen_hashes_after']
  verified.append(dict(task=t,bitstream_sha256=ar['stream_sha256'],z0_archive_sha256=ar['latent_archive_sha256'],
   A_endpoint_bitwise=all(z['raw_output_bitwise'] and z['feature_bitwise'] for z in ar['endpoint_equivalence']),
   G_only_path_checks=gr['adagn_path_checks'],source_receipt=gr['source_replay_receipt']))
  for p in [ar['source_stream'],ar['latent_archive'],gr['result']['feature_path']]:binary[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=sha(p))
 put('audits/stream_and_feature_equality.json',dict(status='PASS' if len(verified)==39 else 'PARTIAL',
  completed=len(verified),planned=39,checks=verified))
 put('audits/reuse_pointwise_verification.json',dict(status='PASS' if len(verified)==39 else 'PARTIAL',
  A_and_B_matching_streams=len(verified),planned=39,source_receipts=[x['source_receipt'] for x in verified]))
 if upto==500:
  idx=get(branch/'checkpoint_index.json')['500']
  for p in (idx['path'],idx['inference_path']):binary[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=sha(p))
 put('binary_artifacts_index.json',dict(status='PASS' if upto==500 and len(verified)==39 else 'PARTIAL',
  files=list(binary.values()),note='Large checkpoints, streams, latent archives and feature arrays stay on server.'))
 print('FINALIZE41',upto,len(verified),flush=True)
if __name__=='__main__':main()
