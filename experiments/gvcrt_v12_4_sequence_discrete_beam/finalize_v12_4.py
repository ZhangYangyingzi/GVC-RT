#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle';CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text())
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
 with p.open('w',newline='') as f:
  if fields:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def tag(v,q):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{q}"
cells=[(v,q,tag(v,q)) for v in MAN['videos'] for q in CFG['requested_qps']]
dones=[json.loads((ROOT/'parts'/f'{t}_done.json').read_text()) for _,_,t in cells]
if len(dones)!=12 or any(d['status']!='PASS' for d in dones):raise SystemExit('not all cells PASS')
keys=['candidate_actions','beam_states','final_trajectories','final_stream','frame_metrics','search_coverage','state_sync','bitstream_audit']
merged={k:[] for k in keys};steps=[]
for v,q,t in cells:
 finals=read(ROOT/'parts'/f'{t}_final_trajectories.csv');cellsteps=read(ROOT/'parts'/f'{t}_trajectory_steps.csv')
 if len(cellsteps)!=64*len(finals):raise RuntimeError(f'lineage row mismatch {t}')
 for i,r in enumerate(cellsteps):r['final_trajectory_id']=finals[i//64]['trajectory_id'];r['lineage_state_id']=r['trajectory_id']
 steps+=cellsteps
 for k in keys:merged[k]+=read(ROOT/'parts'/f'{t}_{k}.csv')
stepmap={r['candidate_id']:r['lineage_state_id'] for r in steps}
for r in merged['candidate_actions']:
 r['trajectory_id']=stepmap.get(r['candidate_id'],'CANDIDATE_ONLY::'+r['candidate_id'])
write(ROOT/'raw/candidate_actions.csv',merged['candidate_actions']);write(ROOT/'raw/beam_states.csv',merged['beam_states']);write(ROOT/'raw/trajectory_steps.csv',steps);write(ROOT/'raw/final_trajectories.csv',merged['final_trajectories']);write(ROOT/'raw/final_streams.csv',merged['final_stream']);write(ROOT/'raw/frame_metrics.csv',merged['frame_metrics']);write(ROOT/'raw/search_coverage.csv',merged['search_coverage']);write(ROOT/'raw/state_sync.csv',merged['state_sync']);write(ROOT/'raw/bitstream_audit.csv',merged['bitstream_audit'])
bases=[]
for v,q,t in cells:
 r=next(x for x in merged['final_trajectories'] if x['dataset']==v['dataset'] and int(x['video_id'])==int(v['video_id']) and int(x['qp'])==q and x['is_baseline_path']=='True');old=V121/'bitstreams'/f'{t}_baseline.bin';bases.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':q,'frames':64,'baseline_bytes':r['total_bytes'],'baseline_bits':int(r['total_bytes'])*8,'baseline_PSNR':r['PSNR'],'baseline_SSIM':r['SSIM'],'baseline_LPIPS':r['LPIPS'],'baseline_DISTS':r['DISTS'],'bitstream_path':r['bitstream_path'],'bitstream_sha256':r['bitstream_sha256'],'v12_1_sha256':sha(old),'sha_match_v12_1':r['bitstream_sha256']==sha(old)})
write(ROOT/'raw/baseline_cells.csv',bases)
valid=[r for r in merged['candidate_actions'] if r['valid_status']=='VALID'];invalid=[r for r in merged['candidate_actions'] if r['valid_status']!='VALID'];z_invalid=[r for r in invalid if r['symbol_family_1']=='z']
zaudit={'v12_3_decode_fail_count':10,'v12_3_failure_family':'z','fix':'reject nonzero w0/w1 symbols that become inactive after z-dependent prior recomputation','v12_4_total_actions':len(merged['candidate_actions']),'v12_4_valid_actions':len(valid),'v12_4_invalid_actions':len(invalid),'v12_4_invalid_inactive_nonzero':sum(r['valid_status']=='INVALID_INACTIVE_NONZERO' for r in invalid),'v12_4_invalid_entropy_decode_mismatch':sum(r['valid_status']=='ENTROPY_DECODE_MISMATCH' for r in invalid),'v12_4_invalid_z_actions':len(z_invalid),'invalid_entered_beam':sum(r['entered_beam']=='True' for r in invalid),'status':'PASS' if sum(r['entered_beam']=='True' for r in invalid)==0 and sum(r['valid_status']=='ENTROPY_DECODE_MISMATCH' for r in invalid)==0 else 'FAIL'}
(ROOT/'sanity/z_edit_codec_audit.json').write_text(json.dumps(zaudit,indent=2)+'\n')
selected=merged['final_stream'];ft=merged['final_trajectories'];audit=merged['bitstream_audit'];sync=merged['state_sync'];sibling=json.loads((ROOT/'sanity/sibling_state.json').read_text())
checks={'status':'PASS','expected_cells':12,'completed_cells':len(dones),'beam_width':CFG['beam_width'],'temporary_rate_debt_ratio':CFG['temporary_rate_debt_ratio'],'baseline_path_present_all_cells':sum(r['is_baseline_path']=='True' for r in ft)==12,'baseline_sha_match_v12_1':all(r['sha_match_v12_1'] for r in bases),'selected_budget_pass':all(int(r['selected_bytes'])<=int(r['baseline_bytes']) for r in selected),'selected_decode_pass':all(r['decode_status']=='PASS' for r in selected),'selected_state_sync_pass':all(r['state_sync_status']=='PASS' for r in selected),'selected_finite_pass':all(r['finite_status']=='PASS' for r in selected),'all_complete_trajectory_decode_pass':all(r['decode_status']=='PASS' for r in ft),'all_complete_trajectory_state_sync_pass':all(r['state_sync_status']=='PASS' for r in ft),'all_complete_trajectory_finite_pass':all(r['finite_status']=='PASS' for r in ft),'independent_audit_pass':all(r['decode_status']=='PASS' and r['symbol_equal']=='True' and r['reconstruction_equal']=='True' and r['finite']=='True' and int(r['frames'])==64 for r in audit),'search_state_sync_pass':all(r['status']=='PASS' for r in sync),'model_parameters_unchanged':all(d['model_unchanged'] and d['model_hash_before']==d['model_hash_after'] for d in dones),'sibling_state_isolation_pass':sibling['status']=='PASS','z_edit_codec_audit':zaudit['status'],'no_training':True,'continuous_latent_optimization':False,'future_ground_truth_used_for_proposals':False,'candidate_action_rows':len(merged['candidate_actions']),'beam_state_rows':len(merged['beam_states']),'complete_trajectories':len(ft),'trajectory_step_rows':len(steps)}
if not all(v for k,v in checks.items() if k.endswith('_pass') or k in ('baseline_path_present_all_cells','baseline_sha_match_v12_1','model_parameters_unchanged')):checks['status']='FAIL'
(ROOT/'sanity/final_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
print('V12.4 EXPERIMENT:',ROOT);print('BEAM WIDTH:',CFG['beam_width']);print('TEMPORARY DEBT RATIO:',CFG['temporary_rate_debt_ratio']);print('CELLS:',len(dones),'/ 12')
for r in selected:print(r['dataset'],r['video'],f"QP={r['qp']}",f"baseline={r['baseline_bytes']}",f"selected={r['selected_bytes']}",r['selected_type'],f"modified={r['num_modified_frames']}",f"single={r['num_single_edits']}",f"pair={r['num_pair_edits']}",f"PSNR={r['selected_PSNR']}",f"LPIPS={r['selected_LPIPS']}",f"DISTS={r['selected_DISTS']}")
for n in ('final_streams.csv','final_trajectories.csv','beam_states.csv','trajectory_steps.csv','candidate_actions.csv','search_coverage.csv','state_sync.csv','bitstream_audit.csv'):print(n+':',ROOT/'raw'/n)
print('INDEPENDENT DECODE:', 'PASS' if checks['independent_audit_pass'] else 'FAIL');print('STATE SYNC:', 'PASS' if checks['search_state_sync_pass'] else 'FAIL');print('FINITE:', 'PASS' if checks['all_complete_trajectory_finite_pass'] else 'FAIL');print('MODEL PARAMETERS UNCHANGED:', 'PASS' if checks['model_parameters_unchanged'] else 'FAIL');print('INTEGRITY:',checks['status'])
