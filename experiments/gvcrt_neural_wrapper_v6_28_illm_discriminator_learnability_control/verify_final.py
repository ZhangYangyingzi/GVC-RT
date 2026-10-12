from io21 import *
import collections,traceback
BR=('A_raw_current','B_rms_current','C_zero_current','D_raw_official_optimizer')
def main():
 state=load(ROOT/'pipeline_status.json');state.update(status='RUNNING',phase='integrity_verification');dump(ROOT/'pipeline_status.json',state)
 cfg=load(ROOT/'config.json');assert sha(cfg['source_checkpoint']['path'])==EXPECTED;assert sha(ROOT/'plans/shared.json')=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db';assert sha(cfg['labeler_path'])==cfg['labeler_sha256'];assert sha(cfg['D_source_checkpoint']['path'])==cfg['D_source_checkpoint']['sha256']
 assert load(ROOT/'audits/frozen_source_modules.json')['status']=='PASS' and load(ROOT/'audits/fresh_B1_v627_hash_equivalence.json')['status']=='PASS'
 for p,h in load(ROOT/'audits/frozen_source_modules.json')['original_codec_files'].items():assert sha(p)==h
 changes=[p for p,h in load(ROOT/'audits/dependencies.json').items() if sha(p)!=h];assert not changes,changes;assert load(ROOT/'audits/preflight.json')['status']=='PASS' and load(ROOT/'audits/OASIS_equivalence.json')['status']=='PASS'
 train={};valid=[];binary=[];expected_validation=0
 for b in BR:
  train[b]={}
  for stage in ('TINY','MAIN'):
   base=ROOT/'branches'/b/stage;it=load(base/'integrity.json');assert it['status']=='PASS' and it['effective_D_updates']==cfg['stages'][stage]['updates'] and it['G_updates']==0;rs=read(base/'gradient_summary.csv');assert len(rs)==it['effective_D_updates'];assert all(int(r['frames'])==cfg['stages'][stage]['frames'] for r in rs);assert all(r['gradient_clip']=='' and int(r['G_updates'])==0 for r in rs)
   if stage=='MAIN':
    full=read(base/'gradient_summary_full.csv');assert set(collections.Counter((r['domain'],r['external_QP']) for r in full).values())=={250}
   train[b][stage]=it['effective_D_updates'];idx=load(base/'checkpoint_index.json');assert sorted(map(int,idx))==cfg['stages'][stage]['checkpoints']
   for step,item in idx.items():assert sha(item['path'])==item['sha256'];binary.append(dict(kind='D_checkpoint',**item))
   expected_validation+=len(idx)*(21 if stage=='TINY' else 60)
 for p in (ROOT/'validation').glob('*/*/step_*/*/integrity.json'):
  s=load(p);assert s['status']=='PASS' and s['parameters_buffers_unchanged'] and s['condition_detached'] and s['same_real_fake_condition'] and s['G_updates']==0 and s['D_updates']==0;assert all(sha(p.parent/n)==h for n,h in s['files'].items());valid.append(str(p.relative_to(ROOT)))
 assert len(valid)==expected_validation==1704,(len(valid),expected_validation)
 for p in (ROOT/'cache/status').glob('*.json'):
  s=load(p);assert s['status']=='PASS' and sha(s['path'])==s['sha256'];
  if s.get('reuse_source'):assert sha(s['reuse_source'])==s['legacy_sha256']
  s['task_identity_sha256']=hashlib.sha256(json.dumps(dict(row_hash=s['row_hash'],B1=EXPECTED,labeler=cfg['labeler_sha256'],protocol=sha(ROOT/'protocol.json')),sort_keys=True).encode()).hexdigest();binary.append(dict(kind='fixed_B1_cache',**s))
 assert len(list((ROOT/'cache/status').glob('*.json')))==1060
 known={f['path'] for f in binary if 'path' in f}
 for p in ROOT.glob('branches/*/*/checkpoints/*.pt'):
  if str(p) not in known:binary.append(dict(kind='recovery_checkpoint',path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 for p in [Path(cfg['source_checkpoint']['path']),Path(cfg['D_source_checkpoint']['path']),Path(cfg['labeler_path'])]:binary.append(dict(kind='immutable_source_weight',path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'audits/current_code_hashes.json',{str(p):sha(p) for p in ROOT.glob('*.py')})
 dump(ROOT/'binary_file_index.json',dict(files=binary,large_files_server_only=True))
 fin=dict(status='PENDING_PUBLICATION',training=train,D_updates_total=28000,G_updates=0,validation_completed=len(valid),validation_expected=expected_validation,no_G_training=True,no_new_codec_evaluation=True,source_checkpoints_verified=True,shared_plan_verified=True,labeler_pretrained_verified=True,official_loss_value_gradient_equivalent=True,four_D_initializations_identical=True,fresh_optimizers=True,FP32_D=True,no_D_grad_clip=True,loss_denominators_verified=True,main_domain_QP_cycles_verified=True,validation_params_buffers_unchanged=True,B1_cache_causal_and_frozen=True,fresh_B1_historical_reconstruction_condition_hashes_equal=True,labeler_matches_full_D0=True,unseen_shared_frame_intersection=False,historical_files_unchanged=True,frequency_baseline_training_only=True,all_prescribed_tasks_complete=True,publication_verified=False)
 dump(ROOT/'final_integrity.json',fin);s=load(ROOT/'pipeline_status.json');s.update(status='RUNNING',phase='final_publication',training=train,validation_completed=len(valid),validation_expected=expected_validation,active={},blocked={});dump(ROOT/'pipeline_status.json',s)
 env=os.environ.copy();env['V628_PUBLICATION_PHASE']='complete_numerics';subprocess.check_call([PYTHON,str(ROOT/'publish_github.py')],cwd=REPO,env=env)
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified']
 from index_io import read_file_index
 # Uploaded shards are validated against each original byte stream.
 for item in read_file_index():
  if 'shards' in item:assert hashlib.sha256(b''.join((ROOT/x['path']).read_bytes() for x in item['shards'])).hexdigest()==item['sha256']
 keys=['config.json','protocol.json','audits/preflight.json','audits/OASIS_equivalence.json','results/final_summary.csv','results/classification_summary.csv','results/input_activation_scales.csv','REPORT.md','RESULTS_SUMMARY.md','final_integrity.json','file_index.json','binary_file_index.json']+[f'branches/{b}/{stage}/gradient_summary.csv' for b in BR for stage in ('TINY','MAIN')]
 fin.update(status='PASS',publication_verified=True,shards_verified=True,remote_commit=pub['commit'],remote_key_files=keys);dump(ROOT/'final_integrity.json',fin);s.update(status='PASS',phase='complete',publication='verified',commit=pub['commit']);dump(ROOT/'pipeline_status.json',s)
 env['V628_PUBLICATION_PHASE']='final_integrity_verified';subprocess.check_call([PYTHON,str(ROOT/'publish_github.py')],cwd=REPO,env=env);pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS'
 work=pub['worktree'];prefix=str(ROOT.relative_to(REPO))
 entries={e['path']:e for e in read_file_index()}
 for key in keys:
  targets=[e['path'] for e in entries[key]['shards']] if key in entries and 'shards' in entries[key] else [key]
  for target in targets:subprocess.check_call(['git','cat-file','-e',pub['commit']+':'+prefix+'/'+target],cwd=work)
 dump(ROOT/'audits/remote_final_verification.json',dict(status='PASS',commit=pub['commit'],key_files=keys,all_remote_blobs_verified=True,link=pub['link']));print('FINAL PASS',pub['commit'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  err=traceback.format_exc();dump(ROOT/'logs/final_failure.json',dict(status='BLOCKED',traceback=err))
  if (ROOT/'final_integrity.json').exists():
   fin=load(ROOT/'final_integrity.json');fin.update(status='BLOCKED',publication_verified=False,error_log=str(ROOT/'logs/final_failure.json'));dump(ROOT/'final_integrity.json',fin)
  state=load(ROOT/'pipeline_status.json');state.update(status='BLOCKED',phase='final_verification_or_publication',error_log=str(ROOT/'logs/final_failure.json'));dump(ROOT/'pipeline_status.json',state);raise
