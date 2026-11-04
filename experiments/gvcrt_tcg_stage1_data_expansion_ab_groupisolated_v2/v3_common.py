"""Versioned online validation protocol, atomic state and unchanged base provenance."""
from ab_common import *
import collections,csv,io
V3='online-development-earlystop-v3'
MIG=ROOT/'migrations/online_earlystop_v3'
ARMS=('A_clean_old_pool','B_expanded_ulong')
def protocol():return load(ROOT/'protocol_v3.json')
def verify_sources():
 m=load(ROOT/'source_manifest_v3.json')
 assert sha(ROOT/'protocol_v3.json')==m['protocol_sha256']
 assert sha(ROOT/'source_manifest.json')==m['base_source_manifest_sha256']
 for name,h in m['base_code'].items():assert sha(ROOT/name)==h,('BASE_SOURCE_CHANGED',name)
 base=load(ROOT/'source_manifest.json')
 for name,h in base['read_only_runtime'].items():assert sha(REPO/name)==h,('READ_ONLY_RUNTIME_CHANGED',name)
 for name,h in base['original_helpers'].items():assert sha(ORIGINAL/name)==h,('ORIGINAL_HELPER_CHANGED',name)
 for record in load(P/'source_manifest.json')['model_file_sources']:assert sha(REPO/'checkpoints'/record['filename'])==record['sha256']
 for name,h in m['code'].items():assert sha(ROOT/name)==h,('V3_SOURCE_CHANGED',name)
 old=load(ROOT/'protocol.json')
 for arm in ARMS:assert sha(ROOT/f'plans/{arm}/index.json')==old['plan_index_sha256'][arm]
 assert sha(ROOT/'manifests/evaluation_rows.json')==old['evaluation_rows_sha256']
 assert sha(ROOT/'graph/final_identity_to_group.json')==old['source_group_map_sha256'] and sha(ROOT/'graph/screen_protocol.json')==old['source_screen_protocol_sha256']
 assert sha(ROOT/'inventory/qualified.json')==base['inventory_sha256']
 return m
def rows(arm):
 index=load(ROOT/f'plans/{arm}/index.json');out=[]
 for shard in index['shards']:
  assert sha(ROOT/shard['file'])==shard['sha256'];out.extend(load(ROOT/shard['file']))
 assert len(out)==4*12710
 return out
def latest_path(arm):return ROOT/arm/'checkpoints_v3/latest.pt'
def checkpoint(arm,step):
 if step==0:return P/'checkpoints/step_5000.pt'
 for path in (ROOT/arm/f'checkpoints_v3/step_{step}.pt',ROOT/arm/f'checkpoints/step_{step}.pt'):
  if path.exists():return path
 raise FileNotFoundError(f'Missing saved {arm} checkpoint at additional step {step}')
def committed(arm):
 import torch
 state=torch.load(latest_path(arm),map_location='cpu',weights_only=True)
 n=state['additional_step'];assert state['plan_cursor']==n and state['absolute_optimizer_step']==5000+n and state['rank_sample_cursor']==4*n
 return n
def stopping_init():
 return dict(status='RUNNING',best_raw=dict(step=0,M=1.0),improvement_anchor=1.0,bad_check_count=0,checks=[],last_validation_step=0,stop_reason=None,theoretical_stop_step=None)
def stopping_update(state,step,M,receipt):
 assert math.isfinite(M) and step>0
 if step in [x['step'] for x in state['checks']]:
  existing=next(x for x in state['checks'] if x['step']==step);assert existing['M']==M and existing['evaluation_sha256']==receipt;return state
 assert state['stop_reason'] is None and step>state['last_validation_step']
 new=json.loads(json.dumps(state));before=new['improvement_anchor']
 if M<before-.005:new['improvement_anchor']=M;new['bad_check_count']=0;substantive=True
 else:new['bad_check_count']+=1;substantive=False
 if M<new['best_raw']['M']:new['best_raw']=dict(step=step,M=M)
 new['last_validation_step']=step
 if new['bad_check_count']>=3:new.update(status='EARLY_STOPPED_PATIENCE',stop_reason='EARLY_STOPPED_PATIENCE',theoretical_stop_step=step)
 elif step==12710:new.update(status='MAX_BUDGET_REACHED',stop_reason='MAX_BUDGET_REACHED',theoretical_stop_step=step)
 new['checks'].append(dict(step=step,M=M,anchor_before=before,improvement_anchor=new['improvement_anchor'],substantive_improvement=substantive,bad_check_count=new['bad_check_count'],best_raw=dict(new['best_raw']),evaluation_sha256=receipt))
 return new
def early_state(arm):
 p=ROOT/arm/'early_stop_v3.json'
 return load(p) if p.exists() else stopping_init()
def development_score(records,baseline=None):
 names={Path(r['video']).stem for r in load(ROOT/'manifests/evaluation_rows.json') if r['split']=='development'}
 expected={(name,q) for name in names for q in (0,5,9)}
 assert len(names)==32 and len(records)==96 and {(r['sequence'],int(r['QP'])) for r in records}==expected
 means={str(q):sum(float(r['FloLPIPS']) for r in records if int(r['QP'])==q)/32 for q in (0,5,9)}
 assert all(math.isfinite(x) and x>=0 for x in means.values())
 if baseline is None:
  assert all(x>0 for x in means.values()),'INVALID_PARENT_BASELINE_DENOMINATOR'
  return dict(F_q=means,M=1.,aggregation='equal32-video means within each QP, then mean of3 ratios to parent5000 QP means')
 assert set(baseline)=={'0','5','9'} and all(math.isfinite(x) and x>0 for x in baseline.values()),'INVALID_PARENT_BASELINE_DENOMINATOR'
 M=sum(means[q]/baseline[q] for q in ('0','5','9'))/3;assert math.isfinite(M)
 return dict(F_q=means,M=M,aggregation='equal32-video means within each QP, then mean of3 ratios to parent5000 QP means')
def write_csv(path,records):
 from gtflow import write
 write(path,records)
def update_status(stage,**details):
 arms={}
 for a in ARMS:
  n=committed(a) if latest_path(a).exists() else 0;s=early_state(a)
  arms[a]=dict(committed_additional_updates=n,absolute_optimizer_step=5000+n,status=s['status'],best_dev_step=s['best_raw']['step'],bad_check_count=s['bad_check_count'],next_validation_step=None if s['stop_reason'] else min((s['last_validation_step']//500+1)*500,12710))
 dump(ROOT/'run_status.json',dict(status='RUNNING',stage=stage,protocol_revision=V3,updated=time.time(),arms=arms,**details))
def gpu_mapping(rank,cards):
 devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
 device_map={int(v[0].strip()):dict(uuid=v[1].strip(),name=v[2].strip()) for line in devices.strip().splitlines() if len(v:=line.split(','))==3}
 expected=device_map[cards[rank]]['uuid'];allowed={device_map[k]['uuid'] for k in cards};own=[]
 for attempt in range(10):
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  own=[line.split(',')[1].strip() for line in apps.strip().splitlines() if line.split(',')[0].strip()==str(os.getpid())]
  if expected in own:break
  time.sleep(.2)
 assert expected in own and set(own)<=allowed,('ACTUAL_GPU_MAPPING_MISMATCH',os.getpid(),cards[rank],own)
 return dict(pid=os.getpid(),physical_GPU=cards[rank],physical_rank=rank,expected_UUID=expected,observed_process_GPU_UUIDs=own,name=device_map[cards[rank]]['name'],CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
