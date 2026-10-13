"""Explicit immutable A provenance and per-point verified reuse."""
from io21 import *
def main():
 import torch
 from adapter import cache_key,validate
 old=load(V26/'config.json');cfg=load(ROOT/'config.json');idx=load(V26/'branches/A_decoder_continue/checkpoint_index.json')
 cp=idx['1000'];assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True)
 for k in ('seed','optimizer','source_checkpoint','batch','crop','lambda_dists'):
  assert old[k]==cfg[k],k
 assert s['source_checkpoint_sha256']==EXPECTED and s['adaptation_step']==s['optimizer_update_count']==s['replay_cursor']==1000
 assert s['plan_sha256']==sha(ROOT/'plans/shared.json') and s['config_sha256']==sha(V26/'config.json')
 for k,h in s['code_hashes'].items():assert sha(V26/k)==h
 rows=[json.loads(x) for x in (V26/'branches/A_decoder_continue/training_logs/train.jsonl').read_text().splitlines()]
 plan=load(ROOT/'plans/shared.json')['updates'];assert len(rows)==1000
 for r,p in zip(rows,plan):
  assert all(r[k]==p[k] for k in p) and r['supervision_frames']==15 and r['warmup_frames']==48 and r['reference_resets']==[0]
  assert r['lambda_rec']==.1383589248954108 and r['reconstruction_unit_gradient_scale']==4096 and r['D_optimizer_updates']==0 and r['all_finite']
  assert r['learning_rates']=={'bridge':3e-6,'generator':5e-7} and r['PBG_gradients']['wrapper']['norm']==0
  assert r['lambda_adv']==0 and r['global_clip_coefficient']==min(1.,1./(r['global_preclip_norm']+1e-6))
 for k,m in (('bridge','bridge'),('generator','generator')):
  assert cp['module_hashes'][k]==tensor_hash(s[k])
 ec=evalcfg();ec['checkpoints']['A_continue_1000']=load(V26/'evaluation/config.json')['checkpoints']['A_decoder_continue_1000'];ec['deployment_receiver_hashes']['A_continue_1000']=load(V26/'evaluation/config.json')['deployment_receiver_hashes']['A_decoder_continue_1000'];dump(ROOT/'evaluation/config.json',ec)
 dump(ROOT/'audits/A_reuse_equivalence.json',dict(status='PASS',source_config=str(V26/'config.json'),source_config_sha256=sha(V26/'config.json'),seed=20261024,optimizer=cfg['optimizer'],checkpoint=cp,source_code_hashes=s['code_hashes'],shared_plan_sha256=s['plan_sha256'],valid_updates=1000,loss='complete B1, no adversarial term',reference_detached=True,native_I_bypass=True,reconstruction_unit_scale=4096,global_clip_once=1.,metadata_note='Retains original checkpoint bytes and branch label. No claim of byte equality to a newly serialized V6.29 checkpoint.',historical_code_unmodified=True))
 branch=ROOT/'branches/A_continue';dump(branch/'checkpoint_index.json',idx);dump(branch/'training_integrity.json',dict(status='PASS',updates=1000,reused=True,source=str(V26/'branches/A_decoder_continue')))
 for name in ('gradient_summary.csv',):
  (branch/name).write_bytes((V26/'branches/A_decoder_continue'/name).read_bytes())
 records=[]
 for d in DATASETS:
  origin=V27 if d=='mcljcv' else V26
  for m,oldm in (('original','original'),('B1_initial','B1_initial'),('A_continue_1000','A_decoder_continue_1000')):
   for v in sources(d):
    for q in range(10):
     p=origin/'parts/ip_-1'/d/oldm/f'video_{v["video_index"]:02d}_qp{q}.json';r=load(p)
     assert r['source_sha256']==v['source_sha256'] and r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
     assert r['independent_decode_pass'] and r['state_sync_pass'] and r['real_RANS'] and r['frames']==64
     assert sha(r['feature_path'])==r['feature_sha256'] and sha(r['bitstream_path'])==r['bitstream_sha256']
     assert r['compression_hash_before']==r['compression_hash_after']==ec['compression_hash']
     assert r['loaded_receiver_module_hashes']==(ec['original_receiver_hashes'] if m=='original' else ec['deployment_receiver_hashes'][m])
     if m!='original':assert r['checkpoint_sha256']==ec['checkpoints'][m]['sha256']
     r.update(method=m,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,source_method=oldm,cache_key=cache_key(v,q,m,-1))
     validate(r,v,q,m);dump(point(d,m,v['video_index'],q),r);records.append(dict(dataset=d,method=m,QP=q,sequence=v['name'],path=str(p),sha256=sha(p)))
 dump(ROOT/'audits/reuse_points.json',dict(status='PASS',reused=len(records),expected=1380,records=records));print('REUSE PASS',len(records),flush=True)
if __name__=='__main__':main()
