import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;EX=R.parent;V34=EX/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500';V32=EX/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
plan=json.load(open(V34/'plans/shared.json'));refplan=json.load(open(V32/'plans/shared.json'));assert len(refplan['updates'])>=1000
assert [x['source_plan_index'] for x in plan['updates'][:1000]]==list(range(1000))
assert [x['source_row'] for x in plan['updates'][:1000]]==refplan['updates'][:1000]
manifest=json.load(open(R/'eval_manifest.json'));eval_rows=[v for vs in manifest.values() for v in vs]
eval_paths={str(Path(v['source_path']).resolve()) for v in eval_rows};eval_sha={v['source_sha256'] for v in eval_rows};eval_names={(v['dataset'],v['name']) for v in eval_rows};eval_frames={h for v in eval_rows for h in v['frame_rgb_sha256']}
ref_paths=set();ref_sha=set();ref_ids=set();ref_frames=set();receipt_records=[]
for i,entry in enumerate(plan['updates'][:1000]):
 row=entry['source_row'];rp=V32/'reference_stats/receipts'/f'{i:04d}.json';d=json.load(open(rp));assert d['status']=='PASS' and d['plan_index']==i and d['optimizer_updates']==0
 assert d['source_sha256']==row['source_sha256'] and d['QP']==row['external_qp'] and d['video']==row['video'] and d['domain']==row['domain']
 assert d['full_RGB_sha256']==row['full_RGB_sha256']
 ref_paths.add(str(Path(row['source_path']).resolve()));ref_sha.add(row['source_sha256']);ref_ids.add((row['domain'],row['video']));ref_frames.update(d['full_RGB_sha256'])
 receipt_records.append({'plan_index':i,'receipt':str(rp),'receipt_sha256':sha(rp),'source_sha256':d['source_sha256'],'domain':d['domain'],'QP':d['QP'],'video':d['video'],'status':'PASS'})
inter={'source_path':sorted(eval_paths&ref_paths),'source_sha256':sorted(eval_sha&ref_sha),'domain_name':sorted([list(x) for x in eval_names&ref_ids]),'frame_rgb_sha256_count':len(eval_frames&ref_frames)}
assert not any(inter.values()),inter
out={'status':'PASS','source_plan_sha256':sha(V32/'plans/shared.json'),'plan_prefix_equal_to_v634b':True,'GT_reference_receipts_verified':len(receipt_records),'GT_reference_optimizer_updates':0,'eval_video_count':len(eval_rows),'eval_vs_full_1000_reference_intersections':inter,'receipt_index_hash':hashlib.sha256(''.join(x['receipt_sha256'] for x in receipt_records).encode()).hexdigest(),'near_duplicate_scope':'Exact source path/SHA, domain-name and decoded RGB-frame hash intersection; does not claim exhaustive perceptual near-duplicate or official pretraining exposure.'}
(R/'audits/gt_reference_isolation.json').write_text(json.dumps(out,indent=2)+'\n')
print('GT reference isolation PASS',len(receipt_records),inter)
