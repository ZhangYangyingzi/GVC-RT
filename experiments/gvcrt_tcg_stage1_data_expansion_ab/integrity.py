"""Independent count, tensor, receipt, Cartesian-product and aggregation checks."""
from ab_common import *
from gtflow import METRICS,write
import csv,gzip,collections
def main():
 import torch
 p=load(ROOT/'protocol.json');S=p['additional_global_updates'];parent=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True)
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 initial=[];arms={};plans={}
 for arm in ('A_old_pool','B_expanded_ulong'):
  index=load(ROOT/f'plans/{arm}/index.json');assert sha(ROOT/f'plans/{arm}/index.json')==p['plan_index_sha256'][arm]
  plan=[]
  for s in index['shards']:assert sha(ROOT/s['file'])==s['sha256'];plan.extend(load(ROOT/s['file']))
  assert len(plan)==4*S;plans[arm]=plan
  st=torch.load(ROOT/f'{arm}/checkpoints/step_0.pt',map_location='cpu',weights_only=True)
  assert equal(st['tcg_model_state'],parent['tcg_model_state']) and equal(st['optimizer'],parent['optimizer']) and equal(st['all_rank_RNG_states'],parent['all_rank_RNG_states'])
  initial.append(st)
  for step in range(1,S+1):
   r=json.loads(gzip.decompress((ROOT/f'{arm}/receipts/{step:06d}.json.gz').read_bytes()))
   assert r['additional_step']==step and r['absolute_optimizer_step']==5000+step and r['rank_sample_cursor']==4*step
   assert r['finite_loss'] and r['finite_nonzero_gradient'] and r['only_TCG_changed'] and r['buffer_reset'] and r['rank_sync']=='PASS'
   assert len(r['rank_audits'])==4 and {a['rank'] for a in r['rank_audits']}==set(range(4))
   assert len({a['TCG_hash'] for a in r['rank_audits']})==1
   for a in r['rank_audits']:
    row=plan[4*(step-1)+a['rank']];assert a['sample_sha256']==digest(row) and a['external_qp']==row['external_qp'] and a['domain']==row['domain']
    assert a['frozen_check']=='EXACT_DEVICE_BYTES_PASS' and a['frozen_hash']==parent['frozen_state_hash'] and a['non_TCG_delta_norm']==0 and a['TCG_delta_norm']>0
    assert math.isfinite(a['local_gradient_norm']) and a['local_gradient_norm']>0 and math.isfinite(a['reduced_preclip_gradient_norm']) and a['reduced_preclip_gradient_norm']>0
    assert len(a['frame_loss_trace'])==15 and [t['position'] for t in a['frame_loss_trace']]==list(range(49,64))
    assert all(math.isfinite(v) for v in a['losses'].values())
  assert len(list((ROOT/arm/'receipts').glob('*.json.gz')))==S
  last=torch.load(ROOT/f'{arm}/checkpoints/latest.pt',map_location='cpu',weights_only=True)
  assert last['additional_step']==last['plan_cursor']==S and last['absolute_optimizer_step']==5000+S and last['rank_sample_cursor']==4*S
  assert {int(s['step']) for s in last['optimizer']['state'].values()}=={5000+S}
  assert len(last['all_rank_RNG_states'])==4 and all(x is None for b in last['TCG_buffers'] for x in b.values())
  for cp in load(ROOT/f'{arm}/checkpoints_manifest.json')['checkpoints']:assert sha(ROOT/f'{arm}/checkpoints'/cp['filename'])==cp['sha256']
  arms[arm]=dict(status='PASS',committed_additional_updates=S,absolute_optimizer_step=5000+S,rank_samples=4*S)
 for a,b in zip(plans['A_old_pool'],plans['B_expanded_ulong']):
  assert a['domain']==b['domain'] and a['external_qp']==b['external_qp']
  if a['domain']=='uvg':assert a==b
 from evaluation_runner import states,eval_videos
 evaluations={}
 for kind,expected in [('parent_gtflow',156),('development',32*3*len(states('development'))),('final',540)]:
  points=[load(f) for f in (ROOT/f'evaluation/{kind}/points').glob('*.json')];assert len(points)==expected and all(x['status']=='PASS' for x in points)
  keys=[]
  for point in points:
   r=point['result'];assert all(math.isfinite(float(r[m])) for m in METRICS)
   key=(r.get('split','historical_reference'),r['domain'],r['sequence'],r['QP'],r.get('model_state',r['step']));keys.append(key)
   stream=(P/'large_artifacts/evaluation/bitstreams' if kind=='parent_gtflow' else ROOT/f'large_artifacts/{kind}/bitstreams')/r['bitstream_file']
   assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==int(r['real_bytes'])==point['bytes_consumed']
   assert point['independent_decode'] and point['model_hashes_unchanged'] and all(x['cleared'] for x in point['TCG_reset_audits'])
   assert point['fresh_encode']==(kind!='parent_gtflow') and point['input_stream_reused']==(kind=='parent_gtflow')
   # Match point receipt to its frame/transition files by stream basename.
   name=Path(r['bitstream_file']).stem
   fr=list(csv.DictReader((ROOT/f'evaluation/{kind}/frames/{name}.csv').open()));tr=list(csv.DictReader((ROOT/f'evaluation/{kind}/transitions/{name}.csv').open()))
   assert len(fr)==64 and len(tr)==63 and [int(x['frame']) for x in fr]==list(range(64))
   assert sum(int(x['real_bits']) for x in fr)==8*int(r['real_bytes'])
   assert [int(x['to_frame']) for x in tr]==list(range(1,64))
   for m in ('MS_SSIM','LPIPS','DISTS'):assert abs(sum(float(x[m]) for x in fr)/64-float(r[m]))<1e-10
   for m in ('FloLPIPS','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC'):assert abs(sum(float(x[m]) for x in tr)/63-float(r[m]))<1e-10
   assert r['WarpingError']==r['WarpingError_ReconFlow_PWC']
  assert len(set(keys))==expected
  # Full expected model/video/QP set, not just an equal row count.
  if kind=='parent_gtflow':expected_keys={('historical_reference',v['dataset'],v['name'],q,s) for v in eval_videos('historical_reference') for q in (0,5,9) for s in (0,1000,2500,5000)}
  else:
   splits=['development'] if kind=='development' else ['historical_reference','new_holdout_test']
   expected_keys={(split,v['dataset'],v['name'],q,label) for split in splits for v in eval_videos(split) for q in (0,5,9) for label,_,_,_ in states(kind)}
  assert set(keys)==expected_keys;evaluations[kind]=dict(status='PASS',completed_points=expected,expected_points=expected)
 reserved=load(ROOT/'manifests/reserved_union.local.json')['records'];reserved_ids={r['identity'] for r in reserved}
 def ident(r):return __import__('re').search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',r['video']).group() if r['domain']=='ulong' else Path(r['video']).stem.lower()
 trainids={ident(r) for r in plans['B_expanded_ulong']};devtest={ident(r) for r in load(ROOT/'manifests/evaluation_rows.json')}
 assert not(trainids&reserved_ids or trainids&devtest)
 from ab_model_context import tensor_hash
 assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
 dump(ROOT/'final_integrity.json',dict(status='PASS',arms=arms,evaluations=evaluations,exact_parent_TCG_optimizer_RNG=True,paired_UVG_and_QP=True,actual_optimizer_and_sample_counts=True,receipt_contiguity=True,full_evaluation_Cartesian_products=True,no_nonfinite_metrics=True,frame_transition_aggregation=True,stream_hashes_and_consumption=True,training_reserved_dev_test_overlap=[],checkpoint_hashes=True,parent_unchanged=True,source_manifest_sha256=sha(ROOT/'source_manifest.json'),protocol_sha256=sha(ROOT/'protocol.json')))
 status('ready_for_publication','RUNNING',training=arms,evaluation=evaluations)
if __name__=='__main__':main()
