import json,csv,hashlib,collections,time,math,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000';A=R.parent/'gvcrt_neural_wrapper_v6_34a_p_joint_fd_trend_screen'
def j(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def check(ok,name,detail=None):
 checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail})
 if not ok:raise RuntimeError(f'{name} failed: {detail}')
checks=[]
rest=j(R/'audits/restore_audit.json');check(rest['status']=='PASS','parent_transactions_verified')
plan=P/'plans/shared.json';check(sha(plan)=='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42','parent_plan_sha256')
check(sha(R/'plans/shared.json')==sha(plan),'copied_plan_byte_identity')
check(sha(R/'eval_manifest.json')==sha(A/'eval_manifest.json'),'evaluation_manifest_identity')
updates_expected={'N_noP_BG_FD':25,'P_PBG_FD':132};start_steps={'N_noP_BG_FD':475,'P_PBG_FD':368};shared=j(plan)['updates'];counts=collections.Counter(f"{x['source_row']['domain']}:{x['source_row']['external_qp']}" for x in shared[:500])
branch_summary={}
for b in updates_expected:
 d=R/'branches'/b;st=j(d/'training_status.json');ti=j(d/'training_integrity.json');ix=j(d/'checkpoint_index.json')['500'];ck=Path(ix['path']);
 check(st['status']=='PASS' and st['committed_updates']==500,'branch_step500_'+b,st)
 check(ti['status']=='PASS' and ti['updates_total']==500 and ti['updates_added']==updates_expected[b] and ti['optimizer_internal_step']==500,'training_integrity_'+b)
 check(ck.is_file() and sha(ck)==ix['sha256'],'checkpoint_sha_'+b,{'path':str(ck),'sha256':ix['sha256']})
 x=__import__('torch').load(ck,map_location='cpu',weights_only=True,mmap=True)
 check(x['adaptation_step']==x['committed_optimizer_updates']==x['plan_cursor']==500,'checkpoint_transaction_'+b)
 check(sum(int(v['counter']) for v in x['EMA'].values())==500,'ema_total_'+b)
 check({k:int(v['counter']) for k,v in x['EMA'].items()}==dict(counts),'ema_plan_strata_'+b)
 check(set(int(v['step']) for v in x['optimizer']['state'].values())=={500},'optimizer_internal_steps_'+b)
 check(x['branch']==b and x['plan_sha256']==sha(plan),'branch_and_plan_signature_'+b)
 calls=[json.loads(z) for z in (d/'optimizer_calls.jsonl').read_text().splitlines() if z.strip()]
 cc=collections.Counter(z['event'] for z in calls)
 check(cc['STARTED']==cc['APPLIED']==cc['COMMITTED']==updates_expected[b],'physical_optimizer_call_ledger_'+b,dict(cc))
 receipt=j(d/'receipts'/f'{start_steps[b]+1}.json');expected_item=shared[start_steps[b]]
 check(receipt['plan_row_sha256']==expected_item['source_row_canonical_sha256'],'first_continuation_plan_cursor_'+b)
 check(receipt['finite'] and receipt['optimizer_internal_step']==start_steps[b]+1,'first_continuation_numeric_'+b)
 check(ti['frozen']['compression']=='0f0c0e2358ceec024230154da4777ed5c444a8382af1ec9d7165f3d7886141cf' and ti['frozen']['I']=='502cc8648abd788b19f4659b94c530ad6aa04b33080addb2be59747ec9575199','frozen_codec_hash_'+b)
 inf=Path(ix['inference']['path']);check(inf.is_file() and sha(inf)==ix['inference']['sha256'],'inference_checkpoint_'+b)
 branch_summary[b]={'source_step':start_steps[b],'additional_updates':updates_expected[b],'cumulative_step':500,'checkpoint':str(ck),'checkpoint_sha256':ix['sha256'],'optimizer_steps':500,'ema_commits':500,'ema_counters':{k:int(v['counter']) for k,v in x['EMA'].items()},'module_hashes':ix['module_hashes'],'inference_checkpoint':str(inf),'inference_sha256':ix['inference']['sha256'],'physical_optimizer_events':dict(cc)}
 if b=='P_PBG_FD':
  first=j(d/'receipts'/f'{start_steps[b]+1}.json');wd=first['parameter_delta']['wrapper']['norm'];wg=first['gradient_FD_unit']['wrapper']['norm'];check(wd>0 and wg>0,'P_first_step_FD_gradient_and_update',{'wrapper_FD_norm':wg,'wrapper_delta_norm':wd})
 del x
# Verify held-out sources do not occur in this run's first 500 plan rows or evaluation exclusion inputs.
eval=j(R/'eval_manifest.json')['ulong'];train_sha={x['source_row']['source_sha256'] for x in shared[:500]};overlap=[v['source_sha256'] for v in eval if v['source_sha256'] in train_sha]
check(not overlap,'six_video_holdout_not_in_first500_plan',overlap)
# Verify reused receipts/artifacts and all 36 new task outputs.
reuse=j(R/'audits/historical_reuse_audit.json');check(reuse['status']=='PASS' and reuse['verified_reused_points']==90,'historical_90_reuse')
from evaluate32 import valid
manifest=j(R/'task_manifest.json')['tasks'];newtasks=[t for t in manifest if t['step']==500]
check(len(newtasks)==36,'new_point_manifest_count',len(newtasks))
for t in newtasks:
 check(valid(t),'new_eval_signature_and_artifacts',t)
 check(t['method'] in ('N_noP_BG_FD_500','P_PBG_FD_500'),'new_eval_method',t)
points=[]
with (R/'results/all_rd_points.csv').open(newline='') as f:points=list(csv.DictReader(f))
pooled=[]
with (R/'results/pooled_fid.csv').open(newline='') as f:pooled=list(csv.DictReader(f))
check(len(points)==126,'combined_logical_points',len(points))
check(len(pooled)==21,'pooled_fid_points',len(pooled))
for x in points:
 for m in ('LPIPS','DISTS','FloLPIPS','actual_bpp'):
  check(math.isfinite(float(x[m])),'finite_'+m,{'method':x['method'],'video':x['video'],'QP':x['QP']})
check(all(str(x['real_RANS']).lower() in ('true','1') and str(x['independent_decode_pass']).lower() in ('true','1') and str(x['state_sync_pass']).lower() in ('true','1') for x in points),'all_points_real_decode_sync')
summary={'status':'PASS','mode':'parent continuation to cumulative step 500','branches':branch_summary,'verified_historical_reused_points':90,'new_step500_points':36,'combined_logical_points':126,'pooled_fid_points':21,'all_real_RANS':True,'all_independent_decode':True,'all_reference_state_sync':True,'newly_added_optimizer_updates':157,'EMA_total_commits_each':500,'training_plan_SHA256':sha(plan),'evaluation_manifest_SHA256':sha(R/'eval_manifest.json'),'holdout_sources_excluded_from_first500_updates':True,'startup_repair_resolved':j(R/'audits/startup_repair.json')['status']=='RESOLVED','check_count':len(checks),'checks':checks,'completed_unix':time.time()}
(R/'final_integrity.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
status=j(R/'pipeline_status.json');status.update(computation='PASS',training='PASS',evaluation='PASS',aggregation='PASS',integrity='PASS',logical_points_completed=126,expected_logical_points=126,reused_points=90,new_points=36,updated_unix=time.time());(R/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
# Extend the mechanical report with exact continuation accounting.
report=(R/'REPORT.md').read_text();report += '\n## Continuation transaction checks\n\n- Parent complete transactions restored: N=475 and P=368.\n- New committed AdamW updates: N=25 and P=132; both cumulative endpoints are 500.\n- New continuation optimizer call ledger: STARTED/APPLIED/COMMITTED counts match each required added update count.\n- Both EMA ledgers end at 500 commits and match the first 500 shared plan stratum counts.\n- New N/P endpoint points have valid task signatures, complete artifacts, real RANS, independent decode, and reference synchronization.\n- Resolved startup file lookup failures are retained in `errors.jsonl` and documented in `audits/startup_repair.json`.\n'
(R/'REPORT.md').write_text(report);(R/'RESULTS_SUMMARY.md').write_text(report)
print(json.dumps({k:v for k,v in summary.items() if k not in ('checks',)},sort_keys=True))
