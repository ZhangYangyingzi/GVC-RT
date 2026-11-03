"""Freeze all pairs only after complete native-decode verification; independent RNG."""
from ab_common import *
from audit import identity,rows
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
 if forbidden.exists() and row.get('split','train')=='train' and row['domain']=='ulong':
  banned=set(load(forbidden)['consecutive16_frame_digest_sha256'])
  assert not any(hashlib.sha256(''.join(full[i:i+16]).encode()).hexdigest() in banned for i in range(49)),('HISTORICAL_FORBIDDEN_CONTENT_WINDOW',row['video'])
 if 'full_RGB_sha256' in row:assert full==row['full_RGB_sha256'] and crops==row['cropped_RGB_sha256']
 dump(dest,result);return result
def main():
 with lock('prepare_plans.lock'):
  assert load(ROOT/'inventory/status.json')['status']=='PASS'
  conflict=ROOT/'audits/data_source_conflict.json'
  if conflict.exists() and load(conflict)['status']=='BLOCKED':raise RuntimeError('PROTOCOL_BLOCKED_SOURCE_GROUP_SPLIT_ISOLATION; required old936 cannot be silently changed')
  entries=load(ROOT/'inventory/qualified.json');reserved=load(ROOT/'manifests/reserved_union.local.json')['records'];history=set(load(ROOT/'manifests/historical_project_identities.json')['identities'])
  oldrows=rows();oldids={identity(r['video']) for r in oldrows if r['domain']=='ulong'};oldsha={identity(r['video']):r['source_sha256'] for r in oldrows if r['domain']=='ulong'}
  banned={r['identity'] for r in reserved};bannedsha={h for r in reserved for h in r['hashes']}
  protected=[r for r in entries if r['status']=='ELIGIBLE' and (r['identity'] in banned or r['sha256'] in bannedsha)]
  protected_rgb={tuple(r['fingerprint_first64_RGB']):r['identity'] for r in protected}
  used={};excluded=[];aliases=[];by_sha={};by_rgb={}
  for e in sorted(entries,key=lambda e:(e['archive'],e['entry'])):
   if e['status']!='ELIGIBLE':excluded.append(e);continue
   i=e['identity'];h=e['sha256'];rgb=tuple(e['fingerprint_first64_RGB'])
   if i in banned or h in bannedsha:excluded.append(dict(e,reason='HISTORICAL_RESERVED_IDENTITY_OR_SHA256'));continue
   if rgb in protected_rgb:
    excluded.append(dict(e,reason='RESERVED_NATIVE_RGB_ANCHOR_MATCH',same_source_group=protected_rgb[rgb]));continue
   if i in used:
    assert used[i]['sha256']==h,('CANONICAL_IDENTITY_HAS_DIFFERENT_BYTES',i)
    aliases.append(dict(identity=i,duplicate_of=i,method='same canonical identity',archive=e['archive']));continue
   target=by_sha.get(h) or by_rgb.get(rgb)
   if target:
    aliases.append(dict(identity=i,duplicate_of=target,method='SHA256' if h in by_sha else 'exact native RGB frame 0/31/63',archive=e['archive']))
    # Keep old identities as a required control; ambiguous compatibility stops.
    if i in oldids:raise RuntimeError(f'PARENT_REQUIRED_IDENTITY_DUPLICATE_INCOMPATIBILITY:{i}:{target}')
    excluded.append(dict(e,reason='CONFIRMED_OR_ANCHOR_CONTENT_DUPLICATE',same_source_group=target));continue
   by_sha[h]=i;by_rgb[rgb]=i;used[i]=e
  assert oldids<=used.keys(),('OLD_936_NOT_ALL_ELIGIBLE',sorted(oldids-used.keys()))
  assert all(used[i]['sha256']==oldsha[i] for i in oldids),'PARENT_SOURCE_BYTES_INCOMPATIBLE'
  # Near-duplicate screening across all admitted sources, not filename claims.
  # All three anchor dHashes must be within four bits. Candidates are grouped
  # conservatively for split isolation; no claim this is exhaustive retrieval.
  groups={i:i for i in used}
  def find(i):
   while groups[i]!=i:groups[i]=groups[groups[i]];i=groups[i]
   return i
  buckets=collections.defaultdict(list);near=[]
  for i,e in used.items():
   fp=[int(h,16) for h in e['fingerprint_first64_dhash']]
   candidates=set()
   for anchor in range(3):
    for byte in range(8):candidates.update(buckets[(anchor,byte,(fp[anchor]>>(8*byte))&255)])
   for j in sorted(candidates):
    other=[int(h,16) for h in used[j]['fingerprint_first64_dhash']];distance=[(a^b).bit_count() for a,b in zip(fp,other)]
    if max(distance)<=4:
     near.append(dict(left=i,right=j,Hamming_distance=distance,relation='POTENTIAL_SAME_SOURCE_REENCODE_UNCONFIRMED'));a,b=find(i),find(j);groups[max(a,b)]=min(a,b)
   for anchor in range(3):
    for byte in range(8):buckets[(anchor,byte,(fp[anchor]>>(8*byte))&255)].append(i)
  grouping={i:find(i) for i in used};group_members=collections.defaultdict(list)
  for i,g in grouping.items():group_members[g].append(i)
  candidates=[i for i in used if not (set(group_members[grouping[i]])&history)]
  candidates.sort(key=lambda i:(hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest(),i))
  selected=[];selected_groups=set()
  for i in candidates:
   if grouping[i] in selected_groups:continue
   selected.append(i);selected_groups.add(grouping[i])
   if len(selected)==64:break
  assert len(selected)==64,('FEWER_THAN_64_NEW_QUALIFIED_SOURCE_GROUPS',len(selected))
  dev=selected[:32];test=selected[32:];new_reserved={i for i in used if grouping[i] in selected_groups}
  train=sorted(set(used)-new_reserved);N=len(train);S=10*math.ceil(3*N/20)
  assert oldids<=set(train) and len(oldids)==936
  # Reuse existing extracted files only when reliable per-file SHA agrees.
  lookup={}
  pool=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/train_manifest_ulong_expanded.json'
  for v in load(pool)['videos']:
   if Path(v['path']).is_file():lookup[v['sha256']]=v['path']
  for r in oldrows:
   if r['domain']=='ulong':lookup[r['source_sha256']]=r['source_path']
  def metadata(i):
   e=used[i];return dict(e,source_path=lookup.get(e['sha256'],f"zip://{e['archive']}::{e['entry']}"),source_sha256=e['sha256'],source_size_bytes=e['size_bytes'])
  metas={i:metadata(i) for i in used}
  for label,seq in [('train',train),('development',dev),('new_holdout_test',test)]:dump(ROOT/f'manifests/{label}.json',[metas[i] for i in seq])
  dump(ROOT/'inventory/excluded.json',excluded);dump(ROOT/'inventory/duplicate_aliases.json',aliases);dump(ROOT/'inventory/potential_content_relations.json',near)
  qp_path=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/qp_semantics_audit.json'
  qp_table=load(qp_path)['actual_qps']
  rng=random.Random(SEED);visits=[]
  for coverage in range(3):
   order=train.copy();rng.shuffle(order);visits.extend((i,coverage+1,False) for i in order)
  visits.extend((train[k%N],None,True) for k in range(2*S-len(visits)))
  starts=collections.defaultdict(list)
  def newrow(i,coverage,padding,q,label):
   v=metas[i];available=list(range(0,v['frames']-63,64));rng.shuffle(available)
   fresh=[s for s in available if s not in starts[i]]
   if fresh:start=fresh[0]
   else:
    remaining=v['frames']-63;start=rng.randrange(remaining)
    if remaining>len(set(starts[i])):
     while start in starts[i]:start=(start+1)%remaining
   starts[i].append(start);inds=list(range(start,start+64))
   return dict(domain='ulong',video=v['entry'],source_path=v['source_path'],source_sha256=v['source_sha256'],source_size_bytes=v['size_bytes'],width=v['width'],height=v['height'],source_fps=v['probe']['avg_frame_rate'],canonical_indices=inds,source_frame_indices=inds,crop_x=rng.randrange(v['width']-255),crop_y=rng.randrange(v['height']-255),crop_size=256,external_qp=q,actual_qps=qp_table[str(q)][49:64],actual_frame_qps=qp_table[str(q)][:64],coverage_pass=coverage,padding=padding,split=label)
  schedule=load(P/'plans/schedule.json')['updates'];a=[];b=[];k=0
  for step in range(S):
   source=schedule[step%5000];aa=[];bb=[]
   for rank,item in enumerate(source['ranks']):
    old=dict(oldrows[item['base_row']],external_qp=item['external_qp'],width=1920,height=1080,split='train',parent_schedule_cursor=step%5000,parent_sample_sha256=item['sample_sha256'])
    old.update(actual_qps=qp_table[str(item['external_qp'])][49:64],actual_frame_qps=qp_table[str(item['external_qp'])][:64])
    aa.append(old)
    if old['domain']=='uvg':bb.append(dict(old))
    else:
     i,c,pad=visits[k];k+=1;bb.append(newrow(i,c,pad,item['external_qp'],'train'))
   assert collections.Counter(x['domain'] for x in aa)=={'uvg':2,'ulong':2}
   assert [r['domain'] for r in aa]==[r['domain'] for r in bb]
   a.extend(aa);b.extend(bb)
  evalrows=[newrow(i,None,False,0,label) for label,seq in [('development',dev),('new_holdout_test',test)] for i in seq]
  tasks={}
  # A has exact repeats: verify each unique content row once, reattach slot metadata.
  def content(r):return {k:v for k,v in r.items() if k not in ('external_qp','actual_qps','actual_frame_qps','coverage_pass','padding','split','parent_schedule_cursor','parent_sample_sha256','adaptation_step')}
  for r in a+b+evalrows:
   c=content(r);tasks.setdefault(digest(c),c)
  verified={}
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   for i,(key,value) in enumerate(zip(tasks,pool.map(verify_window,tasks.values(),chunksize=1)),1):
    verified[key]=value
    if i%25==0:
     status('plan_window_verification',completed_windows=i,expected_windows=len(tasks));event('WINDOW_VERIFIED',completed=i,expected=len(tasks))
  def attach(r):return dict(r,**{k:verified[digest(content(r))][k] for k in ('full_RGB_sha256','cropped_RGB_sha256','rgb_sha256')})
  for label,rr in [('A_old_pool',a),('B_expanded_ulong',b)]:
   shards=[]
   for start in range(0,len(rr),100):
    f=ROOT/f'plans/{label}/rows_{start//100:05d}.json';dump(f,[attach(r) for r in rr[start:start+100]]);shards.append(dict(file=str(f.relative_to(ROOT)),sha256=sha(f),rows=min(100,len(rr)-start)))
   unique={digest((r['video'],r['source_frame_indices'],r['crop_x'],r['crop_y'],r['external_qp'])) for r in rr}
   dump(ROOT/f'plans/{label}/index.json',dict(S=S,rank_samples=4*S,shards=shards,exact_unique_combinations=len(unique),exact_repeats=len(rr)-len(unique),domains=dict(collections.Counter(r['domain'] for r in rr)),QP_counts=dict(collections.Counter(r['external_qp'] for r in rr))))
   pervideo=collections.defaultdict(list)
   for r in rr:pervideo[(r['domain'],identity(r['video']))].append(r)
   statistics=[]
   for (domain,i),rs in sorted(pervideo.items()):
    windows=[r['source_frame_indices'][0] for r in rs];unique_starts=sorted(set(windows))
    window_sets=[set(t) for t in sorted({tuple(r['source_frame_indices']) for r in rs})]
    overlaps=sum(not x.isdisjoint(y) for k,x in enumerate(window_sets) for y in window_sets[k+1:])
    statistics.append(dict(domain=domain,identity=i,visits=len(rs),padding_visits=sum(r.get('padding',False) for r in rs),starts=windows,unique_start_count=len(unique_starts),unique_start_pair_count=len(unique_starts)*(len(unique_starts)-1)//2,overlapping_unique_start_pairs=overlaps,unique_crops=len({(r['crop_x'],r['crop_y']) for r in rs}),exact_unique_combinations=len({digest((r['source_frame_indices'],r['crop_x'],r['crop_y'],r['external_qp'])) for r in rs}),QP_counts=dict(collections.Counter(r['external_qp'] for r in rs))))
   dump(ROOT/f'plans/{label}/coverage_statistics.json',dict(per_video=statistics,definition='A visit is one sample, not a new independent video; overlapping unique windows share at least one actual source-frame index. QPs and repeats are recorded explicitly.'))
  dump(ROOT/'manifests/evaluation_rows.json',[attach(r) for r in evalrows])
  checkpoints=sorted({0,2,S,*range(500,S+1,500),*(math.ceil(S*f/4) for f in (1,2,3))})
  inherited=load(P/'protocol.json')
  protocol=dict(task='tcg-stage1-data-expansion-ab-20261103',baseline_commit=BASELINE,seed=SEED,parent_checkpoint_sha256=PARENT_SHA,parent_protocol_sha256=sha(P/'protocol.json'),parent_absolute_optimizer_step=5000,N_A=936,N_B=N,additional_global_updates=S,absolute_final_optimizer_step=5000+S,physical_gpus=[4,5,6,7],world_size=4,effective_batch_size=4,domain_slots_per_update={'uvg':2,'ulong':2},A_then_B_serial=True,coverage_passes=3,padding_ulong_samples=2*S-3*N,coverage_definition='One identity visit per pass, not exhaustive windows; overlapping windows explicitly counted',checkpoint_additional_steps=checkpoints,development_steps=sorted({0,S,*(math.ceil(S*f/4) for f in (1,2,3))}),primary_checkpoint='additional_step_S',frame_loss=inherited['frame_loss'],reconstruction_backward_scale=4096,motion_weight=.01,AdamW=inherited['AdamW'],global_clip=1.,raft_sha256=inherited['raft_sha256'],inherit_parent_precision=True,detach_all_history=True,no_BPTT=True,trainable_parameters=137760,no_router=True,plan_index_sha256={label:sha(ROOT/f'plans/{label}/index.json') for label in ('A_old_pool','B_expanded_ulong')},evaluation_rows_sha256=sha(ROOT/'manifests/evaluation_rows.json'),PWC_GTFlow='PWC(GT_t,GT_t-1); backwarp(recon_t-1,flow); mean full-native-image RGB L1, then mean 63 transitions',PWC_ReconFlow='Existing parent WarpingError alias; PWC(recon_t,recon_t-1)',GT_flow_offline_only=True,cache_cap_GiB=80,large_artifact_cap_GiB=20,disk_free_reserve_GiB=100,CPU_preparation_workers=4,no_hyperparameter_search=True,no_checkpoint_selection=True)
  dump(ROOT/'protocol.json',protocol)
  dump(ROOT/'source_manifest.json',dict(parent=load(ROOT/'audits/source_and_parent.json'),code={f.name:sha(f) for f in ROOT.glob('*.py')},immutable_plans=protocol['plan_index_sha256'],inventory_sha256=sha(ROOT/'inventory/qualified.json')))
  dump(ROOT/'split_audit.json',dict(status='PASS',N_A=936,N_B=N,development_count=32,new_test_count=32,parent_ids_all_in_B=oldids<=set(train),training_reserved_overlap=[],training_dev_test_group_overlap=[],historical_use_screened=True,content_screen_method='SHA256, canonical identity, exact RGB anchors 0/31/63, three-anchor dHash <=4 grouped conservatively',limitations='Anchor fingerprints do not prove absence of all temporally shifted/cropped/reencoded source relations; unresolved candidates retained in records',plans_all_native_decoded=True))
  dump(ROOT/'audits/plans_frozen.json',dict(status='PASS',protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),verified_unique_windows=len(tasks),A_index_sha256=protocol['plan_index_sha256']['A_old_pool'],B_index_sha256=protocol['plan_index_sha256']['B_expanded_ulong']))
  event('PLANS_FROZEN',N_A=936,N_B=N,S=S,absolute_final_step=5000+S)
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise
