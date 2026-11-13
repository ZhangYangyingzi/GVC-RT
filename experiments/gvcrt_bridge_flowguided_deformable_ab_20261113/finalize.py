from util import *
import csv,math,gc,collections,ast,concurrent.futures
GROUPS=['A_zero_base','B_flow_guided']

def wait_assets():
 for name in ['render0','render1','profile_parts']:
  while True:
   p=ROOT/'receipts'/f'{name}.status.json'
   if p.exists():
    r=load(p)
    if r['stage']=='COMPLETE':break
    proc=Path('/proc')/str(r['pid'])
    if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':raise RuntimeError(('ASSET_PROCESS_FAILED',name,str(ROOT/'logs'/f'{name}.log')))
   heartbeat('finalize','WAITING_FOR_ASSET',asset=name);time.sleep(3)
 assert (ROOT/'audits/performance_components.json').exists()


def checkpoints():
 manifest=[];initial=[];base=[]
 for g in GROUPS:
  folder=ROOT/'checkpoints'/g;init=load(folder/'initialization.json');initial.append(init['state_sha256']);done=load(folder/'completion.json');assert done['committed_updates']==300 and done['stop_reason']=='BUDGET_300';base.append(done['frozen_before_after_sha256'])
  for step in [0,1,10,50,100,200,300]:
   path=folder/f'step_{step:04d}.pt';ck=readtensor(path);assert ck['committed_updates']==step and ck['group']==g and ck['protocol_sha256']==check_protocol();assert ck['training_plan_sha256']==sha(ROOT/'training_plan.json') and ck['initialization_sha256']==sha(ROOT/'checkpoints/common_initialization.pt');manifest.append(dict(group=g,step=step,path=str(path),bytes=path.stat().st_size,sha256=sha(path),protocol_sha256=ck['protocol_sha256'],implementation=ck['implementation'],uploaded=False));del ck
  for step in [0,1,10,50,100,200,300]:
   d=load(folder/f'diagnostic_{step:04d}.json');assert all(math.isfinite(v) for v in d['gradient_groups'].values())
   if step==0:assert d['gradient_groups']['output_projection']>0 and all(v==0 for k,v in d['gradient_groups'].items() if k!='output_projection')
   if step>=10:assert all(d['gradient_groups'][k]>0 for k in ['offset_head','offset_upstream','DCN','fusion_upstream','q_head','q_upstream'])
  for step in [0,50,100,200,300]:
   p=ROOT/'evaluation/milestones'/g/f'{step:04d}.json';assert load(p)['status']=='PASS' and not load(p)['stop']
 assert len(set(initial))==1 and len(set(base))==1;dump(ROOT/'checkpoint_manifest.json',dict(status='PASS',initialization_sha256=sha(ROOT/'checkpoints/common_initialization.pt'),common_initial_state_hash=initial[0],frozen_base_hash_concat_P_I=base[0],checkpoints=manifest))
 return manifest

def metrics_integrity():
 from evaluate import available_jobs,tag,proofpath
 js,done=available_jobs();assert done and len(js)==150;proofs=[load(proofpath(j)) for j in js];by=collections.defaultdict(list);frames=0;identities=0;rowskeys=set()
 for p in proofs:
  assert p['status']=='PASS' and p['protocol_sha256']==check_protocol() and p['summary']['frames']==64;by[p['job']['id'],p['job']['qp']].append(p)
  path=ROOT/'metrics/frames'/f"{tag(p['job'])}.csv";rr=list(csv.DictReader(path.open()));assert len(rr)==64 and [int(r['frame']) for r in rr]==list(range(64));frames+=len(rr)
  for r in rr:
   key=(tag(p['job']),r['frame']);assert key not in rowskeys;rowskeys.add(key)
   for k in ['PSNR','LPIPS','DISTS','RGB_MSE','edge_strength','GT_edge_strength','HF_energy','HF_GT_error']:assert math.isfinite(float(r[k]))
   for k in ['FloLPIPS','MC_RGB','MC_Sobel']:
    if r.get(k):assert math.isfinite(float(r[k]))
   assert r['RWE']=='' and r['RWE_status']=='RWE_NOT_VERIFIED'
  if p['identity64_exact']:identities+=1
 assert frames==9600 and identities==16
 for (id,qp),ps in by.items():assert len({p['stream_sha256'] for p in ps})==1 and len({json.dumps(p['codec_feature_hashes']) for p in ps})==1
 assert len(by)==15;return dict(points=150,frames=frames,unique_video_QP=15,exact64_identity_points=identities,shared_bitstreams_features=True,no_duplicate_rows=True,finite_metrics=True,RWE='NOT_VERIFIED_NO_NUMBERS')


def large_manifest():
 paths=[]
 for base in ['cache','checkpoints','videos']:
  for p in (ROOT/base).rglob('*'):
   if p.is_file() and p.suffix in ['.pt','.bin','.npz','.mkv','.mp4']:paths.append(p)
 def entry(p):
  digest=sha(p);receipt=p.with_suffix(p.suffix+'.json')
  if receipt.exists():r=load(receipt);assert p.stat().st_size==r['bytes'] and digest==r['sha256'],('ARTIFACT_CORRUPT',str(p))
  return dict(path=str(p),bytes=p.stat().st_size,sha256=digest,uploaded=False,reproduction='REPRODUCE.md; original source/plan/model/precision identities are in RUN_MANIFEST.json and PROTOCOL.json')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:items=list(pool.map(entry,paths))
 dump(ROOT/'large_artifacts.json',dict(large_artifacts_uploaded=False,total_bytes=sum(p['bytes'] for p in items),files=items));return len(items)

def main():
 check_protocol();assert load(ROOT/'audits/native_validity_correction/complete.json')['status']=='PASS';wait_assets();heartbeat('finalize','VERIFYING_CHECKPOINTS_METRICS');cp=checkpoints();check=metrics_integrity()
 from report_finalize import rows,tables,plots,write_report
 data=rows();assert len(data)==150;agg,paired=tables(data);figures=plots(data);heartbeat('finalize','EXPORTING_AND_VERIFYING_VIDEOS')
 from media_finalize import deliver
 media=deliver();write_report(data,agg,paired,figures);heartbeat('finalize','VERIFYING_ARTIFACT_HASHES');large_count=large_manifest()
 # Preserve all earlier audited source references unchanged and record final code identities.
 run=load(ROOT/'RUN_MANIFEST.json')
 for p,h in run['reuse_evidence'].items():assert sha(p)==h,('PREVIOUS_READONLY_SOURCE_CHANGED',p)
 code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*.py')};code.update({str(p.relative_to(ROOT)):sha(p) for base in ['vendor','runtime_tree','original_tree'] for p in (ROOT/base).rglob('*.py') if 'runtime_dependencies' not in p.parts});dump(ROOT/'source_implementation_final.json',dict(files=code,protocol_sha256=check_protocol(),training_source_identity='Per-checkpoint implementation hashes retained; postprocessing code is separate',datetime=stamp()))
 run.update(completed_local=stamp(),actual_groups={g:load(ROOT/'checkpoints'/g/'completion.json') for g in GROUPS},evaluations=check,figures=figures,video_count=len(media),checkpoint_count=len(cp),large_artifact_count=large_count,RWE_status='RWE_NOT_VERIFIED',task_scope='one paired A/B round only; no extra training');dump(ROOT/'RUN_MANIFEST.json',run)
 dump(ROOT/'publication_status.json',dict(status='LOCAL_COMPLETE_GITHUB_PENDING',branch='codex/bridge-flowguided-deformable-ab-20261113',remote_verified=False,large_artifacts_uploaded=False))
 summary=dict(status='LOCAL_COMPLETE_GITHUB_PENDING',experiment_directory=str(ROOT),groups={g:{k:load(ROOT/'checkpoints'/g/'completion.json')[k] for k in ['committed_updates','stop_reason']} for g in GROUPS},evaluated_points=150,evaluated_frames=9600,frames_per_point=64,baseline_step0='PASS64 exact',beta0='PASS64 exact',frozen='PASS',bitstream='PASS',RWE='RWE_NOT_VERIFIED',report_path=str(ROOT/'REPORT.md'),metrics_path=str(ROOT/'metrics'),figures_path=str(ROOT/'figures'),video_index_path=str(ROOT/'VIDEO_INDEX.md'),github=load(ROOT/'publication_status.json'),remaining=['Git commit/push and remote commit/blob verification'],resume_command='Only after own processes exited: '+PYTHON+' -B '+str(ROOT/'manage.py')+' resume');dump(ROOT/'summary.json',summary)
 dump(ROOT/'final_integrity.json',dict(status='PASS',datetime=stamp(),protocol_sha256=check_protocol(),**check,groups={g:300 for g in GROUPS},same_capacity_and_initialization=True,all_original_weights_frozen=True,actual_video_count=42,lossless_RGB8_roundtrip_all=True,checkpoint_count=len(cp),large_artifact_count=large_count,prior_sources_unchanged=True,source_manifest=str(ROOT/'source_implementation_final.json'),offline_native_validity_correction='PASS72 MC-only; original metrics retained; stop decisions invariant',source_path_migrations_exact_SHA=4,test135_used=False))
 heartbeat('finalize','COMPLETE');event('LOCAL_DELIVERY_COMPLETE',points=150,frame_rows=9600,lossless_videos=42)

if __name__=='__main__':
 with lock('finalize.lock'):main()
