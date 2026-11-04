"""Immutable group-isolated paired plans, with native hashes and explicit trusted reuse."""
from ab_common import *
from source_graph import identity
from reuse import parent_rows
from ab_data_reader import source_file
import collections,concurrent.futures,random,traceback
def verify_window(row):
 dest=ROOT/f'plans/verified/{digest(row)}.json'
 if dest.exists():return load(dest)
 import cv2,numpy as np
 from PIL import Image
 cv2.setNumThreads(1);cap=None;full=[];crops=[];rgb=hashlib.sha256()
 with source_file(row) if row['domain']=='ulong' else contextlib.nullcontext(None) as path:
  try:
   if path is not None:cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,row['canonical_indices'][0])
   else:v=next(v for v in load(P/'plans/train_manifest_uvg.local.json')['videos'] if v['name']==row['video'])
   for i in row['canonical_indices']:
    if cap is not None:
     ok,a=cap.read();assert ok,(row['video'],i);a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    else:
     png=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(png)==v['frame_file_sha256'][i]
     with Image.open(png) as image:a=np.asarray(image,dtype=np.uint8).copy()
    assert a.shape==(row['height'],row['width'],3)
    c=a[row['crop_y']:row['crop_y']+256,row['crop_x']:row['crop_x']+256].copy();assert c.shape==(256,256,3)
    rgb.update(a.tobytes());full.append(hashlib.sha256(a.tobytes()).hexdigest());crops.append(hashlib.sha256(c.tobytes()).hexdigest())
  finally:
   if cap is not None:cap.release()
 result=dict(row,full_RGB_sha256=full,cropped_RGB_sha256=crops,rgb_sha256=rgb.hexdigest())
 forbidden=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/audits/forbidden_content_windows.json'
 if forbidden.exists() and row['domain']=='ulong':
  banned=set(load(forbidden)['consecutive16_frame_digest_sha256'])
  assert not any(hashlib.sha256(''.join(full[i:i+16]).encode()).hexdigest() in banned for i in range(49)),('HISTORICAL_FORBIDDEN_CONTENT_WINDOW',row['video'])
 if 'full_RGB_sha256' in row:assert full==row['full_RGB_sha256'] and crops==row['cropped_RGB_sha256']
 dump(dest,result);return result
def content_key(r):return digest([r['domain'],r['source_sha256'],r['canonical_indices'],r['source_frame_indices'],r['crop_x'],r['crop_y'],r.get('width',1920),r.get('height',1080)])
def content_row(r):return {k:r[k] for k in ('domain','video','source_path','source_sha256','source_size_bytes','width','height','canonical_indices','source_frame_indices','crop_x','crop_y','crop_size') if k in r}
def statistics(rr):
 byvideo=collections.defaultdict(list)
 for r in rr:byvideo[(r['domain'],identity(r['video']))].append(r)
 stats=[]
 for (domain,i),rs in sorted(byvideo.items()):
  window_tuples=sorted({tuple(r['source_frame_indices']) for r in rs});sets=[set(t) for t in window_tuples]
  stats.append(dict(domain=domain,identity=i,source_group=rs[0]['source_group'],visits=len(rs),padding_visits=sum(r.get('padding',False) for r in rs),coverage_pass_visits=dict(collections.Counter(str(r.get('coverage_pass')) for r in rs)),start_counts=dict(collections.Counter(str(r['source_frame_indices'][0]) for r in rs)),unique_windows=len(window_tuples),overlapping_unique_window_pairs=sum(not x.isdisjoint(y) for k,x in enumerate(sets) for y in sets[k+1:]),unique_window_pair_denominator=len(sets)*(len(sets)-1)//2,unique_crops=len({(r['crop_x'],r['crop_y']) for r in rs}),exact_unique_combinations=len({digest([r['source_frame_indices'],r['crop_x'],r['crop_y'],r['external_qp']]) for r in rs}),QP_counts=dict(collections.Counter(r['external_qp'] for r in rs))))
 exact={digest([identity(r['video']),r['source_frame_indices'],r['crop_x'],r['crop_y'],r['external_qp']]) for r in rr}
 return dict(rank_samples=len(rr),domains=dict(collections.Counter(r['domain'] for r in rr)),video_identities=len(byvideo),source_groups=len({r['source_group'] for r in rr}),unique_windows=len({digest([r['domain'],identity(r['video']),r['source_frame_indices']]) for r in rr}),unique_crops=len({digest([r['domain'],identity(r['video']),r['crop_x'],r['crop_y']]) for r in rr}),exact_unique_combinations=len(exact),exact_repeats=len(rr)-len(exact),QP_counts=dict(collections.Counter(r['external_qp'] for r in rr)),per_video=stats)
def main():
 with lock('prepare_plans.lock'):
  assert load(ROOT/'audits/source_groups.json')['status']=='PASS'
  nodes={n['identity']:n for n in load(ROOT/'graph/nodes.json')};mapping=load(ROOT/'graph/final_identity_to_group.json');groups=load(ROOT/'graph/final_components.json')
  expected_links=[('d4264516-a1a6-4cdb-ac5b-8fdb1d41d9aa','91d934b0-9ea5-4055-85c2-a2fdcd747a0f'),('062c9c83-4a98-4535-8166-00657e9b9328','2734b7a8-0a72-4281-aadb-f980a8025f1e'),('e2543816-b159-4f25-bed6-c8fdfbebabe7','30fba0d0-f936-4a94-8a56-a5b184829018'),('30fba0d0-f936-4a94-8a56-a5b184829018','7c350d1d-bfdc-424d-8a9a-d33847c3c591'),('7c350d1d-bfdc-424d-8a9a-d33847c3c591','117f393c-002c-413b-bdc7-a17c07852add')]
  regression=[]
  for left,right in expected_links:
   distance=[(int(x,16)^int(y,16)).bit_count() for x,y in zip(nodes[left]['initial_profile']['dHash'],nodes[right]['initial_profile']['dHash'])]
   assert max(distance)<=4 and mapping[left]==mapping[right],('KNOWN_SOURCE_LINK_NOT_REPRODUCED',left,right,distance)
   regression.append(dict(left=left,right=right,first64_anchor_dHash_distances=distance,final_source_group=mapping[left]))
  dump(ROOT/'audits/known_source_links.json',dict(status='PASS',records=regression,graph_computed_without_forcing_reference_counts=True))
  reserved_groups={g['group_id'] for g in groups if g['quarantined']};historical_groups={g['group_id'] for g in groups if g['historical_members']}
  oldrows=parent_rows();oldids={identity(r['video']) for r in oldrows if r['domain']=='ulong'};assert len(oldids)==936
  qualified={i for i,n in nodes.items() if n['in_archive_inventory'] and n['technical_eligible'] and mapping[i] not in reserved_groups}
  # Only byte-identical files are deduplicated; anchor-related distinct clips
  # remain identities within a common group. Prefer a parent-old representative.
  bysha=collections.defaultdict(list)
  for i in qualified:bysha[nodes[i]['hashes'][0]].append(i)
  aliases={}
  first_row={}
  for k,r in enumerate(oldrows):first_row.setdefault(identity(r['video']),k)
  for h,ids in bysha.items():
   order=sorted(ids,key=lambda i:(i not in oldids,first_row.get(i,10**9),i))
   for alias in order[1:]:aliases[alias]=order[0]
  qualified-=aliases.keys();A=sorted(oldids&qualified);removed=sorted(oldids-set(A));assert A
  UVG={identity(r['video']) for r in oldrows if r['domain']=='uvg'};assert all(mapping[i] not in reserved_groups for i in UVG),'UVG_FIXED_SLOTS_TOUCH_A_RESERVE_COMPONENT'
  candidates=sorted((i for i in qualified if mapping[i] not in historical_groups),key=lambda i:(hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest(),i))
  selected=[];selected_groups=set()
  for i in candidates:
   if mapping[i] in selected_groups:continue
   selected.append(i);selected_groups.add(mapping[i])
   if len(selected)==64:break
  assert len(selected)==64,'INSUFFICIENT_NEW_UNEXPOSED_SOURCE_GROUPS'
  dev,test=selected[:32],selected[32:];B=sorted(i for i in qualified if mapping[i] not in selected_groups);assert set(A)<=set(B)
  N_A,N_B=len(A),len(B);S=10*((3*N_B+19)//20)
  def meta(i):
   n=nodes[i];return dict(n['media'],identity=i,source_group=mapping[i],source_SHA256=n['hashes'][0],group_history_exposed=mapping[i] in historical_groups)
  for label,ids in [('A_clean_old_pool',A),('B_expanded_ulong',B),('development',dev),('new_holdout_test',test)]:dump(ROOT/f'manifests/{label}.json',[meta(i) for i in ids])
  quarantine=[dict(identity=i,source_group=mapping[i],reason='COMPONENT_TOUCHES_HISTORICAL_RESERVED_SOURCE') for i,n in nodes.items() if n['in_archive_inventory'] and mapping[i] in reserved_groups]
  dump(ROOT/'graph/quarantine.json',dict(groups=[g for g in groups if g['quarantined']],stock_identities=quarantine,old_removed=removed,byte_aliases=aliases))
  parent_schedule=load(P/'plans/schedule.json')['updates'];valid=collections.defaultdict(list)
  for update in parent_schedule:
   for rank,item in enumerate(update['ranks']):
    base=oldrows[item['base_row']]
    if base['domain']=='ulong' and identity(base['video']) in A:valid[item['external_qp']].append(dict(base_row=item['base_row'],parent_global_update=update['global_update'],rank=rank,item=item))
  for q in range(10):valid[q].sort(key=lambda x:(x['base_row'],x['parent_global_update'],x['rank']));assert valid[q]
  qpfile=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/qp_semantics_audit.json';qps=load(qpfile)['actual_qps']
  sample_rng=random.Random(SEED);visits=[]
  for passno in range(1,4):
   order=B.copy();sample_rng.shuffle(order);visits.extend((i,passno,False) for i in order)
  visits.extend((B[k%N_B],None,True) for k in range(2*S-3*N_B));assert len(visits)==2*S
  starts=collections.defaultdict(list)
  def make_row(i,q,passno,padding,split,rng):
   v=meta(i);available=list(range(0,v['frames']-63,64));rng.shuffle(available);fresh=[x for x in available if x not in starts[i]]
   if fresh:start=fresh[0]
   else:
    start=rng.randrange(v['frames']-63)
    if len(set(starts[i]))<v['frames']-63:
     while start in starts[i]:start=(start+1)%(v['frames']-63)
   starts[i].append(start);indices=list(range(start,start+64))
   return dict(domain='ulong',video=v['video'],source_path=v['source_path'],source_sha256=v['source_sha256'],source_size_bytes=v['source_size_bytes'],width=v['width'],height=v['height'],source_fps=fps_lookup[i],source_group=mapping[i],canonical_indices=indices,source_frame_indices=indices,crop_x=rng.randrange(v['width']-255),crop_y=rng.randrange(v['height']-255),crop_size=256,external_qp=q,actual_frame_qps=qps[str(q)][:64],actual_qps=qps[str(q)][49:64],coverage_pass=passno,padding=padding,split=split)
  # Preindex fps once: make_row never consumes model RNG and does not seek media.
  fps_lookup={r['identity']:r['probe']['avg_frame_rate'] for r in load(ROOT/'inventory/qualified.json')}
  def new_row(i,q,passno,padding,split,rng):
   # Metadata lookup is attached separately to keep all sampling deterministic.
   row=make_row(i,q,passno,padding,split,rng);row['source_fps']=fps_lookup[i];return row
  aa=[];bb=[];replacements=[];cursor=collections.Counter();visit=0
  for step in range(S):
   sched=parent_schedule[step%5000]
   for rank,item0 in enumerate(sched['ranks']):
    item=item0;row0=oldrows[item0['base_row']];replaced=False
    if row0['domain']=='ulong' and identity(row0['video']) not in A:
     candidates_q=valid[item0['external_qp']];replacement=candidates_q[cursor[item0['external_qp']]%len(candidates_q)];cursor[item0['external_qp']]+=1;item=replacement['item'];replaced=True
     replacements.append(dict(additional_update=step+1,rank=rank,external_qp=item0['external_qp'],original_parent_update=sched['global_update'],original_item=item0,original_source_row=row0,replacement_parent_slot=replacement,replacement_source_row=oldrows[item['base_row']],reason='quarantined_component' if mapping[identity(row0['video'])] in reserved_groups else 'byte_alias_or_actual_ineligible_source'))
    old=dict(oldrows[item['base_row']],width=1920,height=1080,external_qp=item0['external_qp'],actual_frame_qps=qps[str(item0['external_qp'])][:64],actual_qps=qps[str(item0['external_qp'])][49:64],split='train',parent_schedule_cursor=step%5000,parent_sample_sha256=item0['sample_sha256'],replacement_applied=replaced,source_group=mapping[identity(oldrows[item['base_row']]['video'])])
    aa.append(old)
    if old['domain']=='uvg':bb.append(dict(old))
    else:
     i,passno,pad=visits[visit];visit+=1;bb.append(new_row(i,item0['external_qp'],passno,pad,'train',sample_rng))
   assert collections.Counter(r['domain'] for r in aa[-4:])=={'uvg':2,'ulong':2}
  assert visit==2*S and len(aa)==len(bb)==4*S
  dump(ROOT/'plans/A_slot_replacements.json.gz',dict(order='same external QP, parent base-row then original global-update/rank, independent cyclic per-QP cursors',replacements=replacements,counts_by_QP=dict(cursor)))
  historical=[]
  removed_rows={k for k,r in enumerate(oldrows) if r['domain']=='ulong' and identity(r['video']) in removed}
  for update in parent_schedule:
   for rank,item in enumerate(update['ranks']):
    if item['base_row'] in removed_rows:historical.append(dict(parent_global_update=update['global_update'],rank=rank,item=item,source_row=oldrows[item['base_row']],source_group=mapping[identity(oldrows[item['base_row']]['video'])]))
  dump(ROOT/'historical_exposure.json',dict(parent_checkpoint_sha256=PARENT_SHA,parent_had_prior_exposure=True,filtering_does_not_erase_training_memory=True,old_original_count=936,removed=removed,removed_parent_base_rows=len(removed_rows),removed_parent_rank_samples=len(historical),occurrences=historical,group_history_index_sha256=sha(ROOT/'manifests/historical_project_identities.json')))
  eval_rng=random.Random(int(hashlib.sha256(f'{SEED}:evaluation-windows'.encode()).hexdigest(),16));evalrows=[new_row(i,0,None,False,split,eval_rng) for split,ids in [('development',dev),('new_holdout_test',test)] for i in ids]
  pending={};trusted={content_key(dict(r,width=1920,height=1080)):r for r in oldrows}
  for r in aa+bb+evalrows:pending.setdefault(content_key(r),content_row(r))
  verified={};reused=[];native=[]
  forbidden=load(ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/audits/forbidden_content_windows.json');banned=set(forbidden['consecutive16_frame_digest_sha256'])
  uvg_meta={v['name']:v for v in load(P/'plans/train_manifest_uvg.local.json')['videos']};checked_png=set()
  for key,r in pending.items():
   if key not in trusted:native.append((key,r));continue
   old=trusted[key];assert old['source_sha256']==r['source_sha256']
   if r['domain']=='uvg':
    v=uvg_meta[r['video']]
    for idx in r['canonical_indices']:
     path=Path(v['input_dir'])/f'frame_{idx:06d}.png'
     if str(path) not in checked_png:assert sha(path)==v['frame_file_sha256'][idx];checked_png.add(str(path))
   else:
    assert not any(hashlib.sha256(''.join(old['full_RGB_sha256'][i:i+16]).encode()).hexdigest() in banned for i in range(49))
    source_profile=load(ROOT/f"graph/profiles/{identity(r['video'])}.json")
    assert source_profile['source_sha256']==r['source_sha256'] and source_profile['screen_rule_sha256']==sha(ROOT/'graph/screen_protocol.json')
    if r['source_path']!=nodes[identity(r['video'])]['media']['source_path']:assert sha(r['source_path'])==r['source_sha256']
   verified[key]=dict(r,full_RGB_sha256=old['full_RGB_sha256'],cropped_RGB_sha256=old['cropped_RGB_sha256'],rgb_sha256=None,verification='trusted parent native64-window audit, exact source/window/crop matched; source SHA rechecked by v2 graph and UVG PNG bytes rechecked')
   reused.append(dict(content_key=key,source_sha256=r['source_sha256'],parent_row_canonical_sha256=digest(old),full_frame_hash_list_sha256=digest(old['full_RGB_sha256']),crop_hash_list_sha256=digest(old['cropped_RGB_sha256'])))
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   for count,((key,_),result) in enumerate(zip(native,pool.map(verify_window,(r for _,r in native),chunksize=1)),1):
    verified[key]=result
    if count%25==0:status('native_plan_window_verification',completed=count,expected=len(native),trusted_reused=len(reused));event('NATIVE_PLAN_WINDOWS_VERIFIED',completed=count,expected=len(native))
  assert len(verified)==len(pending)
  def attach(r):
   v=verified[content_key(r)];return dict(r,full_RGB_sha256=v['full_RGB_sha256'],cropped_RGB_sha256=v['cropped_RGB_sha256'],rgb_sha256=v['rgb_sha256'],window_verification='trusted_parent_reuse' if content_key(r) in trusted else 'v2_native_decode')
  for arm,rr in [('A_clean_old_pool',aa),('B_expanded_ulong',bb)]:
   shards=[]
   for start in range(0,len(rr),100):
    f=ROOT/f'plans/{arm}/rows_{start//100:05d}.json.gz';dump(f,[attach(r) for r in rr[start:start+100]]);shards.append(dict(file=str(f.relative_to(ROOT)),sha256=sha(f),rows=min(100,len(rr)-start)))
   stats=statistics(rr);dump(ROOT/f'plans/{arm}/coverage_statistics.json.gz',stats);dump(ROOT/f'plans/{arm}/index.json',dict(S=S,rank_samples=4*S,shards=shards,coverage_statistics_sha256=sha(ROOT/f'plans/{arm}/coverage_statistics.json.gz'),**{k:v for k,v in stats.items() if k!='per_video' and k!='rank_samples'}))
  dump(ROOT/'manifests/evaluation_rows.json',[attach(r) for r in evalrows]);assert all(verified[content_key(r)]['rgb_sha256'] is not None for r in evalrows)
  inherited=load(P/'protocol.json');checkpoints=sorted({0,2,S,*range(500,S+1,500),*(math.ceil(S*f/4) for f in (1,2,3))})
  protocol=dict(task=TASK_ID,protocol_revision=REVISION,baseline_commit=BASELINE,seed=SEED,parent_checkpoint_sha256=PARENT_SHA,parent_protocol_sha256=sha(P/'protocol.json'),parent_absolute_optimizer_step=5000,N_A=N_A,N_B=N_B,removed_old_training_identities=removed,quarantined_source_groups=len(reserved_groups),training_source_groups=len({mapping[i] for i in B}),additional_global_updates=S,absolute_final_optimizer_step=5000+S,physical_gpus=[4,5,6,7],world_size=4,effective_batch_size=4,domain_slots_per_update={'uvg':2,'ulong':2},A_then_B_serial=True,coverage_passes=3,padding_ulong_samples=2*S-3*N_B,checkpoint_additional_steps=checkpoints,development_steps=sorted({0,S,*(math.ceil(S*f/4) for f in (1,2,3))}),primary_checkpoint='additional_step_S',primary_evaluation_split='new_holdout_test',frame_loss=inherited['frame_loss'],reconstruction_backward_scale=4096,motion_weight=.01,AdamW=inherited['AdamW'],global_clip=1.,raft_sha256=inherited['raft_sha256'],inherit_parent_precision=True,detach_all_history=True,no_BPTT=True,trainable_parameters=137760,no_router=True,plan_index_sha256={arm:sha(ROOT/f'plans/{arm}/index.json') for arm in ('A_clean_old_pool','B_expanded_ulong')},evaluation_rows_sha256=sha(ROOT/'manifests/evaluation_rows.json'),source_group_map_sha256=sha(ROOT/'graph/final_identity_to_group.json'),source_screen_protocol_sha256=sha(ROOT/'graph/screen_protocol.json'),internal_QP_map_sha256=sha(qpfile),PWC_GTFlow='PWC(GT_t,GT_t-1); existing backwarp(recon_t-1,flow); full native RGB L1 then mean63',PWC_sampler='pixel-center grid, align_corners=False, W-1/H-1 normalization; width11 dx1 samples1.1pixels, compatibility only',GT_flow_offline_only=True,cache_cap_GiB=80,large_artifact_cap_GiB=20,disk_free_reserve_GiB=100,CPU_preparation_workers=4,no_hyperparameter_search=True,no_checkpoint_selection=True)
  dump(ROOT/'protocol.json',protocol)
  train_groups={mapping[i] for i in B}|{mapping[i] for i in UVG};dev_groups={mapping[i] for i in dev};test_groups={mapping[i] for i in test}
  assert not(train_groups&reserved_groups or train_groups&dev_groups or train_groups&test_groups or dev_groups&test_groups or selected_groups&historical_groups)
  dump(ROOT/'split_audit.json',dict(status='PASS',protocol_revision=REVISION,N_A=N_A,N_B=N_B,removed_old_training_identities=removed,quarantined_source_groups=len(reserved_groups),train_groups=sorted(train_groups),development_groups=sorted(dev_groups),test_groups=sorted(test_groups),historical_reserved_groups=sorted(reserved_groups),new_dev_test_historical_group_overlap=[],train_dev_test_group_overlap=[],claim='在所记录来源和筛查规则下，训练/开发/测试来源组无交集。',limitations='Sparse fixed screens do not exhaust all offsets/crops/reencodes or establish foundation-pretraining exclusion',new_native_windows_verified=len(native),trusted_source_matched_native_windows_reused=len(reused)))
  dump(ROOT/'audits/window_verification.json.gz',dict(status='PASS',native_decoded=len(native),trusted_reused=reused,total_unique_windows=len(pending),source_graph_profile_hashes={i:sha(ROOT/f'graph/profiles/{i}.json') for i in B}))
  runtime={name:sha(REPO/name) for group in ('training_sources','evaluation_sources') for name in load(P/'source_manifest.json').get(group,{}) if not name.startswith('experiments/gvcrt_tcg_stage1_expanded_5000/')}
  for name in ('experiments/gvcrt_tcg_stage1_evaluation/evaluate.py','experiments/gvcrt_parallel_v5a3_v60/engine.py','experiments/gvcrt_parallel_v5a3_v60/parallel_utils.py','experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/adapter35.py','experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/frames35.py','experiments/gvcrt_neural_wrapper_v4_progressive_joint/train.py','experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py','experiments/gvcrt_neural_wrapper_v4_1_high_convergence_metrics/metric_runtime.py','experiments/gvcrt_neural_wrapper_v2_joint/core.py','experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py','src/utils/common.py','src/utils/stream_helper.py','expericent_generation_input/expericent_interface_causal_controls_v9/src/common.py','expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py'):
   assert (REPO/name).is_file(),('MISSING_FIXED_RUNTIME_DEPENDENCY',name)
   runtime[name]=sha(REPO/name)
  original_helpers={name:sha(ORIGINAL/name) for name in ('expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py','expericent_generation_input/expericent_interface_causal_controls_v9/src/common.py')}
  dump(ROOT/'source_manifest.json',dict(parent=load(ROOT/'audits/source_and_parent.json'),code={f.name:sha(f) for f in ROOT.glob('*.py')},read_only_runtime=runtime,original_helpers=original_helpers,execution_protocol_sha256=sha(ROOT/'EXECUTION_PROTOCOL.md'),revision_diff_sha256=sha(ROOT/'revision_diff.json'),immutable_plans=protocol['plan_index_sha256'],inventory_sha256=sha(ROOT/'inventory/qualified.json'),source_graph_map_sha256=protocol['source_group_map_sha256'],screen_protocol_sha256=protocol['source_screen_protocol_sha256'],reused_from_E1_manifest_sha256=sha(ROOT/'audits/reuse.json')))
  dump(ROOT/'audits/plans_frozen.json',dict(status='PASS',protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),verified_unique_windows=len(pending),native_decoded_windows=len(native),trusted_reused_windows=len(reused),A_index_sha256=protocol['plan_index_sha256']['A_clean_old_pool'],B_index_sha256=protocol['plan_index_sha256']['B_expanded_ulong']))
  event('PLANS_FROZEN',N_A=N_A,N_B=N_B,S=S,absolute_final_step=5000+S,removed_old=removed)
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise
