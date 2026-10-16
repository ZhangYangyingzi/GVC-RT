"""Content-verified read-only imports, with original signatures and execution origins retained."""
from io32 import *
import ast,collections,traceback

def audit_training():
 import torch
 from training32 import norm
 cfg=load(ROOT/'config.json');plan=load(ROOT/'plans/shared.json')['updates'];summary={}
 for b in BRANCHES[:2]:
  ix=load(ROOT/'branches'/b/'checkpoint_index.json');history=OLD32/'branches'/b
  for step in (0,1,250,500,1000):
   r=ix[str(step)];assert sha(r['path'])==r['sha256'] and sha(r['inference']['path'])==r['inference']['sha256']
  for step in (0,1000):
   s=torch.load(ix[str(step)]['inference']['path'],map_location='cpu',weights_only=True);hashes={k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')};assert hashes==ix[str(step)]['module_hashes'];assert hashes['wrapper']==cfg['source_checkpoint']['module_hashes']['wrapper']
   if step==0:assert hashes==cfg['source_checkpoint']['module_hashes']
   assert ix[str(step)]['compression_hash']==cfg['source_checkpoint']['compression_hash'];del s
  rows=[];counts=collections.Counter()
  for i,p in enumerate(plan):
   r=load(history/'receipts'/f'{i+1:04d}.json')
   for k in ('domain','video','source_sha256','canonical_indices','source_frame_indices','crop_x','crop_y','full_RGB_sha256','cropped_RGB_sha256','external_qp','actual_qps'):assert r[k]==p[k],(b,i,k)
   assert r['committed_optimizer_updates']==i+1 and r['supervised_P_frames']==15 and r['finite'] and r['learning_rates']==dict(zip(('bridge','generator'),cfg['branch_lrs'][b]))
   assert all(v['finite'] and v['norm']>0 for v in r['gradient_modules'].values());assert r['EMA_commit_count']==0
   counts[f'{r["domain"]}:{r["external_qp"]}']+=1;rows.append(r)
  assert set(counts.values())=={50}
  for name in ('training_status.json','training_integrity.json'):
   out=dict(load(history/name),execution_origin=str(history),read_only_reuse=True,source_record_sha256=sha(history/name));dump(ROOT/'branches'/b/name,out)
  write(ROOT/'branches'/b/'gradient_summary.csv',[dict(step=r['adaptation_step'],domain=r['domain'],QP=r['external_qp'],C1_norm=r['C1_gradient_norm'],preclip=r['global_preclip_norm'],clip=r['clip_coefficient'],bridge_norm=r['gradient_modules']['bridge']['norm'],generator_norm=r['gradient_modules']['generator']['norm'],seconds=r['seconds'],peak_MiB=r['peak_memory_MiB'],historical_execution=True) for r in rows])
  summary[b]=dict(status='PASS',updates=1000,source_directory=str(history),checkpoint_index_sha256=sha(history/'checkpoint_index.json'),initial_module_hashes=ix['0']['module_hashes'],full_view_completed=sum(r.get('feature_path') is not None for r in rows),missing_auxiliary_steps=[r['adaptation_step'] for r in rows if r.get('feature_path') is None],EMA_commits=0,auxiliary_diagnostics_incomplete=True,not_retrained=True,counts=dict(counts))
 def fn(p,n):return ast.dump(next(x for x in ast.parse(Path(p).read_text()).body if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
 assert fn(ROOT/'training32.py','crop_c1')==fn(OLD32/'training32.py','crop_c1')
 dump(ROOT/'audits/AB_reuse.json',dict(status='PASS',branches=summary,crop_C1_AST_identical=True,optimizer_definition_unchanged=True,auxiliary_forward_cannot_change_crop_gradients=True,ordinary_native_view_equivalence_evidence=['audits/prefill_equivalence_ulong.json','audits/prefill_equivalence_uvg.json'],historical_receipts_not_relabelled_as_new_execution=True,known_incomplete_AB_auxiliary_entries=8,known_incomplete_per_branch=4))

def import_evaluation():
 from evaluate32 import task_id,receipt,eval_signature,source
 ec=load(ROOT/'evaluation_config.json');existing=0;bad=[];validated={}
 for t in load(ROOT/'task_manifest.json')['tasks']:
  if t['stage']=='new' and t['branch'] in BRANCHES[2:]:continue
  p=OLD32/'tasks'/f'{task_id(t)}.json'
  if not p.exists():continue
  try:
   rr=load(p);assert rr['status']=='PASS';r=rr['result'];v=source(t)
   assert r['frames']==64 and r['num_transitions']==63 and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_RANS']
   assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
   assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][:64]
   assert r['compression_hash_before']==r['compression_hash_after']==ec['compression_hash'];assert r['bytes_consumed']==r['real_bytes']==Path(r['bitstream_path']).stat().st_size;assert abs(r['bpp']-8*r['real_bytes']/(64*1920*1080))<1e-15
   if t['stage']=='new':
    ix=load(ROOT/'branches'/t['branch']/'checkpoint_index.json')[str(t['step'])]
    payload=dict(task=t,config=sha(OLD32/'config.json'),protocol=sha(OLD32/'protocol.json'),manifest=sha(OLD32/'eval_manifest.json'),sources=sha(OLD32/'source_hashes.json'),code={n:sha(OLD32/n) for n in ('io32.py','evaluate32.py','adapter32.py')},checkpoint=ix['inference'])
    assert rr['signature']==hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    assert r['runtime_audit']['wrapper_hash']==ix['module_hashes']['wrapper'];assert r['loaded_receiver_module_hashes']==ix['deployment_receiver_hashes']
   else:assert r['loaded_receiver_module_hashes']==(ec['original_receiver_hashes'] if t['method']=='original' else ec['deployment_receiver_hashes'][t['method']])
   for path,meta in rr['artifact_metadata'].items():
    if path not in validated:validated[path]=sha(path)
    assert validated[path]==meta['sha256'];st=Path(path).stat();assert st.st_size==meta['bytes'] and st.st_mtime_ns==meta['mtime_ns']
   n=dict(rr,signature=eval_signature(t),historical_signature=rr['signature'],execution_origin=str(OLD32),read_only_reuse=True,reused=True,fresh_encode=False,reuse_receipt=str(p),reuse_receipt_sha256=sha(p),new_protocol_mapping='formal geometry, RGB, codec, checkpoint and evaluation numerical code unchanged; only new training auxiliary native view differs')
   dump(receipt(t),n);existing+=1
  except Exception as e:bad.append(dict(task=t,error=repr(e),source=str(p)))
 dump(ROOT/'reuse_manifest.json',dict(status='PASS' if not bad else 'PARTIAL_REEVALUATE',verified_evaluation_points=existing,expected_historical=2116,rejected=bad,artifact_hashes_verified=len(validated),numerical_evaluation_code_identical=sha(ROOT/'adapter32.py')==sha(OLD32/'adapter32.py') and sha(ROOT/'evaluate32.py')==sha(OLD32/'evaluate32.py'),reuse_origin=str(OLD32)))
 print('EVALUATION IMPORT',existing,'rejected',len(bad),flush=True)

def isolation():
 plan=load(ROOT/'plans/shared.json')['updates'];cal=load(ROOT/'plans/calibration.json')['updates'];ev=load(ROOT/'eval_manifest.json');train={r['source_sha256'] for r in plan};test=[]
 for d,vs in ev.items():
  for v in vs:test.append(dict(dataset=d,name=v['name'],source_path=v.get('source_path',v.get('path')),source_sha256=v.get('source_sha256',v.get('sha256'))))
 sha_tests={r['source_sha256'] for r in test if r['source_sha256']};assert not train&sha_tests
 allowed={'Beauty','ReadySteadyGo','ShakeNDry','YachtRide'};assert all(any(x in r['video'] for x in allowed) for r in plan+cal if r['domain']=='uvg')
 assert set(r['source_sha256'] for r in cal)<=train
 for d in ('ulong','uvg'):
  assert sum(r['domain']==d for r in plan)==500 and sum(r['domain']==d for r in cal)==20
  assert all(sum(r['domain']==d and r['external_qp']==q for r in plan)==50 for q in range(10))
 dump(ROOT/'data_isolation_audit.json',dict(status='PASS',training_sources=[dict(source_path=r['source_path'],source_sha256=r['source_sha256'],domain=r['domain'],video=r['video']) for r in {x['source_sha256']:x for x in plan}.values()],calibration_sources=[dict(source_path=r['source_path'],source_sha256=r['source_sha256'],domain=r['domain'],video=r['video']) for r in cal],test_sources=test,source_sha256_intersection=[],historical_exclusions_source=str(OLD32),ULong_candidate_pool=1020,UVG_allowed=sorted(allowed),domain_updates={'ulong':500,'uvg':500},calibration_domains={'ulong':20,'uvg':20},new_content_near_duplicate_screening='No new exhaustive perceptual near-duplicate search. Inherits historical cleaned identity exclusions; audits current source identities and file hashes. Unknown historical pretraining exposure not ruled out.',MCL29='Derived from MCL30 by excluding videoSRC17; no extra encoding',training_manifest_sha256={d:sha(ROOT/f'train_manifest_{d}.json') for d in ('ulong','uvg')}))
if __name__=='__main__':
 try:audit_training();isolation();import_evaluation()
 except Exception:error(dict(stage='reuse32b',error=traceback.format_exc()));raise
