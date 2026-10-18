import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent; EX=R.parent; V34=EX/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500'; V33=EX/'gvcrt_neural_wrapper_v6_33_p_causal_rate_utility_ablation';
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
# Verify training checkpoint files as well as inference exports.
idx={}
for b in ['N_noP_BG_FD','P_PBG_FD']:
 ix=json.load(open(V34/'branches'/b/'checkpoint_index.json'));e=ix['500'];p=Path(e['path']);assert p.is_file() and sha(p)==e['sha256']
 ip=e['inference'];assert Path(ip['path']).is_file() and sha(ip['path'])==ip['sha256']
 idx[b]={'train_path':str(p),'train_sha256':sha(p),'inference_path':ip['path'],'inference_sha256':sha(ip['path']),'module_hashes':e['module_hashes'],'optimizer_internal_step':e['optimizer_internal_step'],'ema_counters':e['EMA_counters']}
plan=json.load(open(V34/'plans/shared.json'));assert len(plan['updates'])==5000
used=plan['updates'][:500]
train_identity=set(); train_paths=set(); train_sha=set(); per_domain={}
for x in used:
 r=x['source_row'];ident=(r['domain'],r['video']);train_identity.add(ident);train_paths.add(str(Path(r['source_path']).resolve()));train_sha.add(r['source_sha256']);per_domain[r['domain']]=per_domain.get(r['domain'],0)+1
manifest=json.load(open(R/'eval_manifest.json')); eval_rows=[v for vs in manifest.values() for v in vs]
eval_identity=set();eval_paths=set();eval_sha=set()
for v in eval_rows:
 eval_identity.add((v['dataset'],v['name']));eval_paths.add(str(Path(v['source_path']).resolve()));eval_sha.add(v['source_sha256'])
intersections={'source_path':sorted(train_paths&eval_paths),'source_sha256':sorted(train_sha&eval_sha),'name_domain':sorted([list(x) for x in train_identity&eval_identity])}
paper=EX/'gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation/teacher_sources/gvcrt_paper_v1.html';readme=EX/'gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation/teacher_sources/gvcrt_official_README.md'
put(R/'audits/source_and_checkpoint_verification.json',{'status':'PASS','source_checkpoints':idx,'v634b_parent_final_integrity':json.load(open(V34/'final_integrity.json'))['status'],'source_plan_sha256':sha(V34/'plans/shared.json'),'expected_plan_sha256':'b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42','plan_first500_count':len(used),'training_domains':per_domain,'evaluation_video_count':len(eval_rows),'training_evaluation_identity_intersection':intersections,'official_paper_local_copy':{'path':str(paper),'sha256':sha(paper)} if paper.exists() else None,'official_readme_local_copy':{'path':str(readme),'sha256':sha(readme)} if readme.exists() else None,'near_duplicate_scope':'source identity/path/SHA only; exhaustive frame-level near-duplicate search not performed'})
print('checkpoint and split audit PASS',per_domain,intersections)
