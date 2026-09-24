#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle'
V122=ROOT.parent/'gvcrt_v12_2_same_bit_budget'
MAN=json.loads((ROOT/'manifest.json').read_text());CFG=json.loads((ROOT/'config.json').read_text())

def read_csv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def tag(v,q):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{q}"

cells=[(v,q,tag(v,q)) for v in MAN['videos'] for q in CFG['requested_qps']]
identity=read_csv(V122/'sanity/identity_audit.csv')
baselines=[];failures=[]
for v,q,t in cells:
 b=read_csv(V121/'parts'/f'{t}_baseline.csv')[0];bp=V121/'bitstreams'/f'{t}_baseline.bin';actual=sha(bp)
 expected=next(r['actual_sha256'] for r in identity if r['kind']=='reused_baseline_bitstream' and r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['qp'])==q)
 baselines.append({'dataset':b['dataset'],'video':b['video'],'video_id':b['video_id'],'qp':b['qp'],'frames':b['frames'],'baseline_bytes':b['actual_bytes'],'baseline_bits':b['actual_bits'],'baseline_PSNR':b['PSNR'],'baseline_SSIM':b['MS_SSIM'],'baseline_LPIPS':b['LPIPS'],'baseline_DISTS':b['DISTS'],'baseline_bitstream_path':str(ROOT/'bitstreams'/f'{t}_baseline.bin'),'baseline_sha256':actual,'v12_2_baseline_sha256':expected,'baseline_sha_match':actual==expected})
 failures.append(json.loads((ROOT/'parts'/f'{t}_failure_audit.json').read_text()))
write_csv(ROOT/'raw/baseline_cells.csv',baselines)
write_csv(ROOT/'raw/failure_cells.csv',failures)

parts={
 'candidate_exact_eval':('_candidate_exact_eval.partial.csv','candidate_exact_eval.csv'),
 'accepted_edits':('_accepted_edits.partial.csv','accepted_edits.csv'),
 'frame_metrics':('_frame_metrics.partial.csv','frame_metrics.csv'),
 'state_sync':('_state_sync.partial.csv','state_sync.csv'),
 'search_coverage':('_search_coverage.partial.csv','search_coverage.csv')}
merged={}
for key,(suffix,out) in parts.items():
 rows=[]
 for _,_,t in cells:rows+=read_csv(ROOT/'parts'/f'{t}{suffix}')
 merged[key]=rows;write_csv(ROOT/'raw'/out,rows)

write_csv(ROOT/'raw/discrete_committed_frame_metrics.csv',merged['frame_metrics'])
baseline_frame_rows=[r for r in read_csv(V121/'raw/frame_metrics.csv') if r.get('branch')=='baseline']
for r in baseline_frame_rows:r['stream_status']='BASELINE_VALIDATED'
for r in merged['frame_metrics']:r['branch']='discrete_oracle';r['beta']='';r['stream_status']='COMMITTED_BEFORE_CELL_FAILURE'
write_csv(ROOT/'raw/frame_metrics.csv',baseline_frame_rows+merged['frame_metrics'])

status_rows=[]
for field in ('decode_status','finite_status','accepted','single_or_pair','symbol_family_1'):
 values=sorted(set(r[field] for r in merged['candidate_exact_eval']))
 for value in values:status_rows.append({'field':field,'value':value,'count':sum(r[field]==value for r in merged['candidate_exact_eval'])})
write_csv(ROOT/'raw/candidate_status_counts.csv',status_rows)

bmap={(r['dataset'],int(r['video_id']),int(r['qp'])):r for r in baselines}
final=[];audits=[]
for f in failures:
 key=(f['dataset'],int(f['video_id']),int(f['qp']));b=bmap[key]
 edits=[r for r in merged['accepted_edits'] if (r['dataset'],int(r['video_id']),int(r['qp']))==key]
 changed={'z':set(),'w0':set(),'w1':set()};ns=np=0
 for e in edits:
  ns+=e['edit_type']=='single';np+=e['edit_type']=='pair'
  for j in ('1','2'):
   fam=e.get('symbol_family_'+j,'');idx=e.get('symbol_index_'+j,'')
   if fam in changed and idx!='':changed[fam].add(idx)
 row={'dataset':f['dataset'],'video':f['video'],'video_id':f['video_id'],'qp':f['qp'],'baseline_bytes':b['baseline_bytes'],'discrete_bytes':'','rate_ratio':'','rate_gap_bytes':'','rate_gap_percent':'','baseline_PSNR':b['baseline_PSNR'],'discrete_PSNR':'','delta_PSNR':'','baseline_SSIM':b['baseline_SSIM'],'discrete_SSIM':'','delta_SSIM':'','baseline_LPIPS':b['baseline_LPIPS'],'discrete_LPIPS':'','delta_LPIPS':'','baseline_DISTS':b['baseline_DISTS'],'discrete_DISTS':'','delta_DISTS':'','num_accepted_single_edits':ns,'num_accepted_pair_edits':np,'num_changed_z_symbols':len(changed['z']),'num_changed_w0_symbols':len(changed['w0']),'num_changed_w1_symbols':len(changed['w1']),'bitstream_path':'','bitstream_sha256':'','decode_status':'NOT_RUN_NO_VALID_FULL_STREAM','state_sync_status':f['decode_status_for_committed_frames'],'finite_status':f['finite_status_for_committed_frames'],'cell_status':'FAIL','failure_type':f['failure_type'],'failure_frame':f['failure_frame'],'selected_cumulative_bytes':f['selected_cumulative_bytes'],'baseline_cumulative_bytes':f['baseline_cumulative_bytes'],'budget_gap_bytes':f['budget_gap_bytes']}
 final.append(row)
 audits.append({'dataset':f['dataset'],'video':f['video'],'video_id':f['video_id'],'qp':f['qp'],'bitstream_path':'','bitstream_sha256':'','frames_decoded':0,'all_entropy_symbols_recovered_exactly':'NOT_RUN_NO_VALID_FULL_STREAM','reconstruction_max_abs_diff':'','decode_status':'NOT_RUN_NO_VALID_FULL_STREAM','state_sync_status':f['decode_status_for_committed_frames'],'finite_status':f['finite_status_for_committed_frames'],'cell_status':'FAIL','failure_frame':f['failure_frame'],'failure_type':f['failure_type']})
write_csv(ROOT/'raw/final_streams.csv',final);write_csv(ROOT/'raw/bitstream_audit.csv',audits)
checks={'status':'FAIL','expected_cells':12,'attempted_cells':len(failures),'completed_full_64_frame_cells':0,'failed_cells':len(failures),'all_failures':'CAUSAL_BUDGET_VIOLATION','baseline_sha_match':all(r['baseline_sha_match'] for r in baselines),'candidate_exact_decode_pass':all(r['decode_status']=='PASS' for r in merged['candidate_exact_eval']),'candidate_decode_pass_count':sum(r['decode_status']=='PASS' for r in merged['candidate_exact_eval']),'candidate_decode_fail_count':sum(r['decode_status']!='PASS' for r in merged['candidate_exact_eval']),'candidate_finite_pass':all(r['finite_status']=='PASS' for r in merged['candidate_exact_eval']),'accepted_candidate_decode_pass':all(r['decode_status']=='PASS' and r['finite_status']=='PASS' and float(r['state_max_abs'])==0 for r in merged['candidate_exact_eval'] if r['accepted']=='True'),'committed_state_sync_pass':all(r['status']=='PASS' for r in merged['state_sync']),'committed_finite_pass':all(str(r['finite']).lower()=='true' for r in merged['state_sync']),'model_parameters_unchanged':all(f['model_unchanged'] and f['model_hash_before']==f['model_hash_after'] for f in failures),'final_independent_decode':'NOT_RUN_NO_VALID_FULL_STREAM','actual_rans_candidate_rate':True,'continuous_latent_optimization':False,'training':False,'future_frames_used':False,'candidate_rows':len(merged['candidate_exact_eval']),'accepted_edits':len(merged['accepted_edits']),'committed_frame_rows':len(merged['frame_metrics'])}
(ROOT/'sanity/final_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
print('EXPERIMENT DIRECTORY:',ROOT)
print('FULL 64-FRAME CELLS COMPLETED: 0 / 12')
print('FAILED CELLS: 12')
for r in final:print(r['dataset'],r['video'],f"QP={r['qp']}",f"failure_frame={r['failure_frame']}",f"selected_cumulative={r['selected_cumulative_bytes']}",f"baseline_cumulative={r['baseline_cumulative_bytes']}",f"gap={r['budget_gap_bytes']}",f"single={r['num_accepted_single_edits']}",f"pair={r['num_accepted_pair_edits']}")
for n in ('baseline_cells.csv','candidate_exact_eval.csv','accepted_edits.csv','frame_metrics.csv','final_streams.csv','state_sync.csv','bitstream_audit.csv','search_coverage.csv','failure_cells.csv'):print(n+':',ROOT/'raw'/n)
print('COMMITTED-FRAME STATE SYNC:', 'PASS' if checks['committed_state_sync_pass'] else 'FAIL')
print('COMMITTED-FRAME FINITE:', 'PASS' if checks['committed_finite_pass'] else 'FAIL')
print('MODEL PARAMETERS UNCHANGED:', 'PASS' if checks['model_parameters_unchanged'] else 'FAIL')
print('FINAL INDEPENDENT DECODE:',checks['final_independent_decode'])
print('INTEGRITY:',checks['status'])
