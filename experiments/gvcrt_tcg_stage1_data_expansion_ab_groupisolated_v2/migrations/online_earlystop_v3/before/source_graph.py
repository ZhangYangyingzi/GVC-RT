"""Complete candidate/reserved/history graph before exclusions; conservative components."""
from ab_common import *
from reuse import parent_rows
from ab_data_reader import source_file
import collections,concurrent.futures,gzip,re,traceback
def identity(x):
 m=re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(x),re.I)
 return m.group().lower() if m else Path(str(x)).stem.lower().split('_1920')[0]
def fingerprint(rgb):
 import cv2,numpy as np
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
 return dict(RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),dHash=f'{int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),"big"):016x}')
def descriptors():
 entries=load(ROOT/'inventory/qualified.json');reserved=load(ROOT/'manifests/reserved_union.local.json')['records'];hist=set(load(ROOT/'manifests/historical_project_identities.json')['identities'])
 old=parent_rows();oldids={identity(r['video']) for r in old if r['domain']=='ulong'}
 paths={}
 # This is a candidate-path index, not evidence every candidate was trained on.
 pool=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/train_manifest_ulong_expanded.json'
 for v in load(pool)['videos']:
  if Path(v['path']).is_file():paths[v['sha256']]=v['path']
 for v in reserved:
  for path in v['media_paths']:
   if Path(path).is_file():
    for h in v['hashes']:paths[h]=path
 for r in old:
  if r['domain']=='ulong':paths[r['source_sha256']]=r['source_path']
 nodes={}
 for r in entries:
  i=r['identity'];nodes[i]=dict(identity=i,kind='ulong',in_archive_inventory=True,reserved=False,history=i in hist,parent_old=i in oldids,hashes=[r['sha256']],media=dict(domain='ulong',video=r['entry'],source_path=paths.get(r['sha256'],f"zip://{r['archive']}::{r['entry']}"),source_sha256=r['sha256'],source_size_bytes=r['size_bytes'],width=r['width'],height=r['height'],frames=r['frames']),initial_profile=dict(start=0,frame_indices=[0,31,63],RGB_sha256=r['fingerprint_first64_RGB'],dHash=r['fingerprint_first64_dhash'],provenance='E1 hash-verified native first64 anchors'),technical_eligible=r['status']=='ELIGIBLE')
 manifests=load(ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/eval_manifest.json')
 raw={identity(v['name']):v for vs in manifests.values() for v in vs}
 for v in load(P/'plans/train_manifest_uvg.local.json')['videos']:
  raw[identity(v['name'])]=dict(v,name=v['name'],dataset='uvg_train',source_path=v['source_path'],source_sha256=v['source_sha256'])
 conversion=load(ORIGINAL/'experiments/gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb/ffmpeg_conversion_audit.json')['inherited_command']
 for r in reserved:
  i=r['identity']
  if i not in nodes:nodes[i]=dict(identity=i,kind='external_reserved',in_archive_inventory=False,reserved=True,history=True,parent_old=False,hashes=r['hashes'],technical_eligible=False)
  nodes[i].update(reserved=True,history=True,reserve_evidence=r['sources'])
 for i,v in raw.items():
  if v['dataset']=='ulong':continue
  command=v.get('ffmpeg_command',v.get('conversion_command',conversion))
  pixel=command[command.index('-pixel_format')+1];geometry=command[command.index('-video_size')+1];vf=command[command.index('-vf')+1]
  assert pixel=='yuv420p' and geometry==f"{v['width']}x{v['height']}"
  path=Path(v['source_path']);assert path.is_file(),('MISSING_EXTERNAL_MEDIA',i,str(path))
  size=path.stat().st_size;framebytes=v['width']*v['height']*3//2;assert size%framebytes==0
  node=nodes.setdefault(i,dict(identity=i,kind=v['dataset'],in_archive_inventory=False,reserved=False,history=True,parent_old=False,hashes=[v['source_sha256']],technical_eligible=False))
  node['media']=dict(source_path=str(path),source_sha256=v['source_sha256'],frames=size//framebytes,width=v['width'],height=v['height'],pixel_format=pixel,bit_depth=8,color_filter=vf,original_conversion_command=command,expected_frame0_RGB=v['frame_rgb_sha256'][0] if 'frame_rgb_sha256' in v else v.get('full_RGB_sha256',[None])[0],source_manifest='v6.34 eval or parent UVG train manifest')
 for i in hist:nodes.setdefault(i,dict(identity=i,kind='history_metadata_only',in_archive_inventory=False,reserved=False,history=True,parent_old=False,hashes=[],technical_eligible=False))
 missing=[i for i,n in nodes.items() if n['reserved'] and not n['in_archive_inventory'] and 'media' not in n]
 assert not missing,('UNRESOLVED_NON_ULONG_RESERVE_MEDIA',missing)
 dump(ROOT/'graph/nodes.json',list(nodes.values()));return nodes
def profile_job(node):
 import cv2,numpy as np
 cv2.setNumThreads(1);i=node['identity'];media=node['media'];dest=ROOT/f'graph/profiles/{i}.json'
 if dest.exists():
  r=load(dest);assert r['source_sha256']==media['source_sha256'] and r['screen_rule_sha256']==sha(ROOT/'graph/screen_protocol.json');return r
 maximum=media['frames']-64;assert maximum>=0
 starts=sorted({round(maximum*f/4) for f in range(5)});profiles=[]
 if node['in_archive_inventory']:
  with source_file(media) as path:
   cap=cv2.VideoCapture(str(path))
   try:
    for start in starts:
     cap.set(cv2.CAP_PROP_POS_FRAMES,start);anchors=[]
     for pos in range(64):
      ok,a=cap.read();assert ok,(i,start,pos)
      assert a.shape==(media['height'],media['width'],3)
      if pos in (0,31,63):anchors.append(fingerprint(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)))
     profiles.append(dict(start=start,frame_indices=[start,start+31,start+63],RGB_sha256=[a['RGB_sha256'] for a in anchors],dHash=[a['dHash'] for a in anchors]))
   finally:cap.release()
  expected=node['initial_profile'];assert profiles[0]['RGB_sha256']==expected['RGB_sha256'] and profiles[0]['dHash']==expected['dHash'],'E1_ANCHOR_REPLAY_MISMATCH'
 else:
  path=Path(media['source_path']);assert sha(path)==media['source_sha256']
  framebytes=media['width']*media['height']*3//2
  with path.open('rb') as f:
   for start in starts:
    anchors=[]
    for pos in (start,start+31,start+63):
     f.seek(pos*framebytes);raw=f.read(framebytes);assert len(raw)==framebytes
     command=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format',media['pixel_format'],'-video_size',f"{media['width']}x{media['height']}",'-i','pipe:0','-vf',media['color_filter'],'-frames:v','1','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
     rgb=subprocess.check_output(command,input=raw);assert len(rgb)==media['width']*media['height']*3
     anchors.append(fingerprint(np.frombuffer(rgb,dtype=np.uint8).reshape(media['height'],media['width'],3)))
    profiles.append(dict(start=start,frame_indices=[start,start+31,start+63],RGB_sha256=[a['RGB_sha256'] for a in anchors],dHash=[a['dHash'] for a in anchors]))
  if media.get('expected_frame0_RGB'):assert profiles[0]['RGB_sha256'][0]==media['expected_frame0_RGB'],'RAW_YUV_CONVERSION_DOES_NOT_MATCH_AUDITED_RGB'
 result=dict(identity=i,source_sha256=media['source_sha256'],screen_rule_sha256=sha(ROOT/'graph/screen_protocol.json'),source_frames=media['frames'],profiles=profiles,full_video_all_windows_checked=False)
 dump(dest,result);return result
def graph(nodes,profile_map,phase):
 parent={i:i for i in nodes};edges=[]
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def add(a,b,kind,**evidence):
  if a==b:return
  edges.append(dict(left=a,right=b,kind=kind,**evidence));a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 bysha=collections.defaultdict(list)
 for i,n in nodes.items():
  for h in n['hashes']:bysha[h].append(i)
 for h,ids in bysha.items():
  for i in ids[1:]:add(ids[0],i,'same_file_SHA256',sha256=h,claim='identical file bytes')
 rgb_index={};buckets=collections.defaultdict(list);seen=[]
 for i in sorted(profile_map):
  for profile in profile_map[i]:
   fp=[int(h,16) for h in profile['dHash']];rgb=tuple(profile['RGB_sha256']);tag=(i,profile['start'])
   if rgb in rgb_index:
    other=seen[rgb_index[rgb]];add(i,other[0],'three_native_RGB_anchors_equal',left_start=profile['start'],right_start=other[1],claim='conservative source grouping, not whole-video duplicate proof')
   else:rgb_index[rgb]=len(seen)
   candidates=set()
   for byte in range(8):candidates.update(buckets[(byte,(fp[0]>>(8*byte))&255)])
   for ix in sorted(candidates):
    j,start,otherfp=seen[ix]
    if i==j or (fp[0]^otherfp[0]).bit_count()>4:continue
    distances=[(a^b).bit_count() for a,b in zip(fp,otherfp)]
    if max(distances)<=4:add(i,j,'three_dHash_distances_le4',left_start=profile['start'],right_start=start,distances=distances,claim='suspected source relation; conservative component')
   idx=len(seen);seen.append((i,profile['start'],fp))
   for byte in range(8):buckets[(byte,(fp[0]>>(8*byte))&255)].append(idx)
 # Record the inventory-anchor regression before adding further known evidence.
 initial_roots={find(i) for i,n in nodes.items() if n['in_archive_inventory']}
 initial_protected={find(i) for i,n in nodes.items() if n['reserved']}
 anchor_regression=dict(stock_components=len(initial_roots),stock_reserved_identities=sum(n['reserved'] and n['in_archive_inventory'] for n in nodes.values()),stock_quarantined_groups=len(initial_roots&initial_protected),stock_quarantined_identities=sum(n['in_archive_inventory'] and find(i) in initial_protected for i,n in nodes.items()),removed_old_training_identities=sorted(i for i,n in nodes.items() if n['parent_old'] and find(i) in initial_protected))
 # Existing exact forbidden-content and documented source alias evidence.
 forbidden=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/audits/forbidden_content_windows.json'
 data=load(forbidden);byframe={}
 for source in data['sources']:
  i=identity(source['identity'])
  if i in nodes:
   for h in source['full_frame_RGB_sha256']:byframe[h]=i
 for i,profiles in profile_map.items():
  for p in profiles:
   for h in p['RGB_sha256']:
    if h in byframe:add(i,byframe[h],'native_RGB_matches_existing_forbidden_source',RGB_sha256=h,evidence_file=str(forbidden.relative_to(ORIGINAL)),evidence_sha256=sha(forbidden))
 overlap=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/audits/overlap_integrity.json'
 documented=load(overlap)
 if 'videoSRC17_1920x1080_24' in documented['confirmed']:add('videosrc17','kimono1','existing_confirmed_shifted_native_RGB_overlap',evidence_file=str(overlap.relative_to(ORIGINAL)),evidence_sha256=sha(overlap),scope='Existing audited MCL17/Kimono source relation; no model quality file read')
 components=collections.defaultdict(list)
 for i in nodes:components[find(i)].append(i)
 records=[];mapping={}
 for root,members in sorted(components.items()):
  group=digest(sorted(members));rs=[i for i in members if nodes[i]['reserved']];hs=[i for i in members if nodes[i]['history']];stock=[i for i in members if nodes[i]['in_archive_inventory']]
  record=dict(group_id=group,members=sorted(members),stock_members=sorted(stock),reserved_members=sorted(rs),historical_members=sorted(hs),quarantined=bool(rs),new_unexposed_candidate=not hs and not rs)
  records.append(record)
  for i in members:mapping[i]=group
 dump(ROOT/f'graph/{phase}_components.json',records);dump(ROOT/f'graph/{phase}_identity_to_group.json',mapping)
 with (ROOT/f'graph/{phase}_edges.json.gz').open('wb') as f:f.write(gzip.compress(json.dumps(edges,allow_nan=False,separators=(',',':')).encode(),mtime=0))
 stock_groups={mapping[i] for i in nodes if nodes[i]['in_archive_inventory']};q=[r for r in records if r['quarantined']];old_removed=sorted(i for r in q for i in r['stock_members'] if nodes[i]['parent_old'])
 summary=dict(status='PASS',phase=phase,nodes=len(nodes),edges=len(edges),components=len(records),stock_components=len(stock_groups),stock_reserved_identities=sum(n['reserved'] and n['in_archive_inventory'] for n in nodes.values()),stock_quarantined_groups=sum(bool(r['stock_members']) for r in q),stock_quarantined_identities=sum(len(r['stock_members']) for r in q),quarantined_groups=len(q),removed_old_training_identities=old_removed,N_A=936-len(old_removed),all_reserved_nodes_included=True,all_historical_nodes_included=True)
 summary['anchor_rule_regression_before_forbidden_content_edges']=anchor_regression
 dump(ROOT/f'graph/{phase}_summary.json',summary);event('SOURCE_GRAPH_COMPLETE',**summary);return summary
def main():
 with lock('source_graph.lock'):
  assert load(ROOT/'audits/reuse.json')['status']=='PASS'
  disk();nodes=descriptors()
  rule=dict(version=REVISION,graph_before_exclusions=True,initial_anchors=[0,31,63],dHash='RGB -> cv2 gray -> INTER_AREA 9x8 -> horizontal comparisons -> big-endian64',candidate_edge='All three dHash distances <=4 or all three native RGB hashes equal; file SHA exact and known aliases/forbidden-content edges also included',conservative_connected_components=True,whole_video_duplicate_proof='file SHA only; anchor matching does not establish whole-video equality',dispersed_screen='For every accessible archive and external YUV source, window starts round((frames-64)*k/4), k=0..4; native RGB anchors start+0/31/63. All sampled window triples compared across all sources, including train/history/reserved, before selecting dev/test. No model quality is read.',maximum_CPU_workers=4,no_scientific_independence_exhaustiveness_claim=True)
  if (ROOT/'graph/screen_protocol.json').exists():assert load(ROOT/'graph/screen_protocol.json')==rule
  else:dump(ROOT/'graph/screen_protocol.json',rule)
  initial={i:[n['initial_profile']] for i,n in nodes.items() if 'initial_profile' in n}
  baseline=graph(nodes,initial,'initial')
  regression=dict(reference=dict(stock_components=8478,stock_reserved_identities=204,stock_quarantined_groups=203,stock_quarantined_identities=245,removed_old_training_count=3),observed=baseline,force_matching_counts=False)
  dump(ROOT/'audits/grouping_regression.json',regression)
  jobs=[n for n in nodes.values() if 'media' in n];profiles={};expected=len(jobs)
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   for count,r in enumerate(pool.map(profile_job,jobs,chunksize=1),1):
    profiles[r['identity']]=r['profiles']
    if count%25==0:
     status('dispersed_source_content_screen',completed_sources=count,expected_sources=expected)
     event('SOURCE_PROFILES_VERIFIED',completed=count,expected=expected)
  final=graph(nodes,profiles,'final')
  dump(ROOT/'audits/source_groups.json',dict(status='PASS',new_screen_protocol_sha256=sha(ROOT/'graph/screen_protocol.json'),initial=baseline,final=final,scope='在所记录来源和筛查规则下，训练/开发/测试来源组无交集。最终划分在计划冻结时再次断言；不声称穷尽所有时间偏移/裁剪/重编码关系，不声称基础预训练从未见过。'))
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise
