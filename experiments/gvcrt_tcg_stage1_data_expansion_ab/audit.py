"""Read-only source, checkpoint and historical split audit; never quality selection."""
from ab_common import *
import collections,re
UUID=re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',re.I)
def identity(s):
 m=UUID.search(str(s));return m.group().lower() if m else Path(str(s)).stem.lower().split('_1920')[0]
def rows():return [r for f in sorted((P/'plans').glob('base_rows_*.json')) for r in load(f)]
def records(x):
 if isinstance(x,list):
  for y in x:yield from records(y)
 elif isinstance(x,dict):
  if any(k in x for k in ('source_path','local_path','filename','source_sha256','file_sha256')):yield x
  else:
   for y in x.values():yield from records(y)
def main():
 assert git('rev-parse','HEAD')==BASELINE
 assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
 pins=load(P/'source_manifest.json');verified=[]
 for group in ('training_sources','evaluation_sources','runtime_acceleration','engineering_sources'):
  for n,h in pins.get(group,{}).items():
   f=P/n if '/' not in n else P.parents[1]/n
   assert sha(f)==h,('PARENT_SOURCE_CHANGED',str(f));verified.append(dict(file=n,sha256=h))
 for f in ('protocol.json','data_plan.json','source_manifest.json','train.py','loss.py','evaluate.py','runtime_protocol_addendum.json','checkpoints_manifest.json'):
  (ROOT/'audits'/('parent_'+f.replace('.py','.txt'))).write_bytes((P/f).read_bytes())
 import torch
 st=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True)
 assert st['committed_global_updates']==st['plan_cursor']==5000
 assert {int(s['step']) for s in st['optimizer']['state'].values()}=={5000}
 assert len(st['all_rank_RNG_states'])==4 and all(x is None for b in st['TCG_buffers'] for x in b.values())
 assert st['protocol_sha256']==sha(P/'protocol.json')
 rr=rows();used={identity(x['video']) for x in rr};usedhash={x['source_sha256'] for x in rr}
 assert len(used)==940 and len({identity(x['video']) for x in rr if x['domain']=='ulong'})==936
 files=[(P/'evaluation/manifest.local.json',None),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/validation_manifest.json',None),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v4_1_high_convergence_metrics/test_manifest.json',None),
 (ORIGINAL/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json','reserved_splits'),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/plans/diagnostic.json','heldout_sources'),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/audits/data_isolation.json','test_sources'),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/eval_manifest.json',None),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/eval_manifest.json',None),
 (ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_36_noP_basic_vs_full_loss_matched_500/audits/data_isolation.json','selected')]
 union={};sources=[]
 for f,key in files:
  data=load(f);selected={k:v for k,v in data['splits'].items() if k!='train'} if key=='reserved_splits' else data[key] if key else data
  sources.append(dict(file=str(f.relative_to(ORIGINAL)) if f.is_relative_to(ORIGINAL) else 'parent/'+str(f.relative_to(P)),sha256=sha(f),selector=key))
  for r in records(selected):
   name=next((r[k] for k in ('filename','name','video','source_path','local_path') if k in r),None)
   if name is None:continue
   ident=identity(name);h=next((r[k] for k in ('source_sha256','file_sha256','sha256') if k in r),None)
   z=union.setdefault(ident,dict(identity=ident,hashes=[],sources=[],media_paths=[]))
   if h and h not in z['hashes']:z['hashes'].append(h)
   z['sources'].append(sources[-1])
   for k in ('source_path','local_path','path'):
    if k in r and r[k] not in z['media_paths']:z['media_paths'].append(r[k])
 # Read only manifest/split metadata to conservatively exclude prior project use.
 # v6.18 contains a previously prepared 8665-video candidate pool. A candidate
 # manifest is not evidence that every member was sampled by an optimizer.
 # Its completed C branch has a fixed 1000-row actual plan (500 ULong slots).
 prior=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage'
 exclusions=load(prior/'audits/exclusions.json')
 for ident in exclusions['identities']:
  union.setdefault(ident,dict(identity=ident,hashes=[],sources=[dict(file=str((prior/'audits/exclusions.json').relative_to(ORIGINAL)),sha256=sha(prior/'audits/exclusions.json'),selector='identities')],media_paths=[]))
 history=set(used)|set(union);history_files=[];candidate_only=[]
 roots=[ORIGINAL/'experiments',ORIGINAL/'expericent_generation_input']
 for folder in roots:
  for pattern in ('**/train_manifest*.json','**/validation_manifest.json','**/test_manifest.json','**/data_split.json'):
   for f in sorted(folder.glob(pattern)):
    if any(s in f.parts for s in ('.git','runtime','exports','publication','.publication_worktree')):continue
    try:
     if f.name=='train_manifest_ulong_expanded.json':
      candidate_only.append(dict(file=str(f.relative_to(ORIGINAL)),sha256=sha(f),actual_use_evidence='plans/C.json; candidate membership alone does not mean optimizer exposure'));continue
     data=load(f);ss=json.dumps(data);history.update(m.lower() for m in UUID.findall(ss));history_files.append(dict(file=str(f.relative_to(ORIGINAL)),sha256=sha(f)))
    except (ValueError,OSError):continue
 for f in (prior/'plans/AB.json',prior/'plans/C.json'):
  data=load(f);history.update(x.lower() for x in UUID.findall(json.dumps(data)));history_files.append(dict(file=str(f.relative_to(ORIGINAL)),sha256=sha(f),selector='actual fixed executed sample plan'))
 conflicts=[dict(identity=i,reason='identity',reserved=union[i]) for i in sorted(used&union.keys())]
 for i,r in union.items():
  hit=usedhash&set(r['hashes'])
  if hit:conflicts.append(dict(identity=i,reason='file_hash',hashes=sorted(hit),reserved=r))
 dump(ROOT/'manifests/reserved_union.local.json',dict(records=list(union.values()),sources=sources))
 dump(ROOT/'manifests/historical_project_identities.json',dict(identities=sorted(history),source_manifests=history_files,candidate_only=candidate_only,scope='Local train/validation/test/data_split metadata plus parent actual plan and explicit reserve union; expanded candidate manifests resolved to actual executed plans; no foundation-pretraining exclusion claim; sealed quality results never read'))
 dump(ROOT/'split_audit.json',dict(status='FAILED' if conflicts else 'RESERVATION_AUDIT_PASS',training_identities=940,N_A=936,conflicts=conflicts,reserved_identities=len(union),new_splits='NOT_ASSIGNED'))
 dump(ROOT/'audits/source_and_parent.json',dict(status='PASS',baseline_commit=BASELINE,parent_checkpoint=dict(filename='step_5000.pt',sha256=PARENT_SHA,size_bytes=(P/'checkpoints/step_5000.pt').stat().st_size),optimizer_absolute_step=5000,rank_RNG_count=4,verified_sources=verified,read_only_parent=True))
 if conflicts:raise RuntimeError('HISTORICAL_RESERVED_TRAIN_CONFLICT; dependent training prohibited')
 event('HISTORICAL_AUDIT_PASS',N_A=936,reserved_identities=len(union),historically_observed=len(history))
if __name__=='__main__':main()
