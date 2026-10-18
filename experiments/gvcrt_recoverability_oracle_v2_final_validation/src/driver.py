"""Bounded, resumable calibration, frozen selection, independent rate-only testing."""
from protocol import *
import itertools,random,traceback,math

def state(**items):
 value=dict(updated=time.time(),**items);dump('manifests/run_status.json',value);dump('run_status.json',value)
def resources():
 output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(row.split(',')[0]):list(map(int,row.split(',')[1:])) for row in output.splitlines()}
def succeeded(job):
 path=ROOT/'manifests/receipts'/(job['id']+'.json')
 return path.exists() and load(path)['status']=='PASS' and load(path)['job_sha256']==sha(ROOT/'manifests/jobs'/(job['id']+'.json'))
def stage(name,jobs):
 pending=[j for j in jobs if not succeeded(j)];running=[];idle=time.time()
 dump('manifests/stages/'+name+'.json',dict(ids=[j['id'] for j in jobs],time=time.time()))
 out=ROOT/'logs/stage_exits'/(name+'.json');exits=load(out) if out.exists() else []
 while pending or running:
  for process,job,gpu,attempt,started in list(running):
   rc=process.poll()
   if rc is None and time.time()-started>1200:
    process.terminate();rc=process.wait(timeout=30)
   if rc is None:continue
   running.remove((process,job,gpu,attempt,started))
   record=ROOT/'manifests/receipts'/(job['id']+'.json')
   receipt=load(record) if record.exists() else dict(status='FAILED',reason='Worker exited without receipt')
   exits.append(dict(id=job['id'],pid=process.pid,gpu=gpu,attempt=attempt,exit_code=rc,status=receipt['status'],finished=time.time()))
   if rc==0 and succeeded(job):
    if job['kind']!='repeat':put_ref(job,record,'v2_real_encode_decode')
   elif 'OutOfMemoryError' in receipt.get('traceback','') and attempt<2:pending.append(job)
  mem=resources();busy={g for _,_,g,_,_ in running}
  with (ROOT/'logs/GPU_resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),stage=name,GPUs=mem))+'\n')
  for gpu in sorted(cfg()['GPU_ids'],key=lambda g:-mem[g][0]):
   if not pending or gpu in busy or mem[gpu][0]<cfg()['minimum_free_MiB']:continue
   job=pending.pop(0);attempt=1+len(list((ROOT/'manifests/attempts').glob(job['id']+'_*.json')))
   command=[PYTHON,'-B','-u',str(ROOT/'src/worker.py'),'--job',str(ROOT/'manifests/jobs'/(job['id']+'.json')),'--gpu',str(gpu),'--attempt',str(attempt)]
   with (ROOT/'logs'/(job['id']+'_'+str(attempt)+'.log')).open('w') as f:
    process=subprocess.Popen(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   started=time.time();running.append((process,job,gpu,attempt,started))
   dump('manifests/commands/'+job['id']+'_'+str(attempt)+'.json',dict(command=command,pid=process.pid,gpu=gpu,start=started,GPU_snapshot=mem))
  if running:idle=time.time()
  if pending and not running and time.time()-idle>cfg()['resource_wait_limit_seconds']:
   for job in pending:dump('manifests/receipts/'+job['id']+'.json',dict(status='FAILED',job=job,launched=False,reason='No authorized GPU had 7000 MiB free for the registered resource timeout'))
   pending=[]
  state(status='RUNNING',stage=name,pending=len(pending),active=[dict(id=j['id'],pid=p.pid,gpu=g) for p,j,g,_,_ in running],completed=sum(bool(succeeded(j)) for j in jobs),planned=len(jobs))
  dump('logs/stage_exits/'+name+'.json',exits)
  if running or pending:time.sleep(10)
 dump('manifests/stage_results/'+name+'.json',dict(all_successful=all(succeeded(j) for j in jobs),jobs=[dict(id=j['id'],success=bool(succeeded(j))) for j in jobs]))

def available(phase,d,q,plan):
 ref=lookup(phase,d,q,plan)
 if ref and usable(read_receipt(ref)):return ref
 return None
def new_job(phase,d,q,plan,kind='candidate',**kw):
 job=register(phase,d,q,plan,kind,**kw)
 assert len(list((ROOT/'manifests/jobs').glob('*.json')))<=cfg()['maximum_new_GPU_jobs'],'Registered GPU job cap exceeded'
 return job
def prediction_table():
 saved=ROOT/'oracle/rate_allocation/candidate_predictions.json'
 if saved.exists():return load(saved)
 rows=[];priors=load(ROOT/'manifests/native_rate_priors.json')
 for d in load(ROOT/'manifests/videos.json'):
  for q in cfg()['QPs']:
   target=baseline('calibration',d,q)['rate']['native_codec_stream_bits'];singles={}
   for i,b in enumerate(cfg()['blocks']):
    for f in (.5,2.):
     ref=available('calibration',d,q,[dict(b,factor=f)]);assert ref
     r=read_receipt(ref);sy=load(V1/'oracle/local_sensitivity/symbols'/(r['job']['id']+'.json'))[0]
     singles[i,f]=dict(bits=r['rate']['actual_total_bits']-target-80,loss=r['result']['LPIPS']-baseline('calibration',d,q)['result']['LPIPS'],changed=sy['effective_q_changed_fraction'])
   for dq in cfg()['global_q_offsets']:
    qq=q+dq
    if str(qq) not in priors[d]:continue
    base_bits=priors[d][str(qq)]['bits']
    for ls in itertools.product((1.,.5,2.),repeat=4):
     plan=plan_for(ls)
     if not plan:continue
     noop=any(singles[i,f]['changed']==0 for i,f in enumerate(ls) if f!=1.)
     bits=base_bits+80+sum(singles[i,f]['bits'] for i,f in enumerate(ls) if f!=1.)
     value=sum(singles[i,f]['loss'] for i,f in enumerate(ls) if f!=1.)
     rows.append(dict(candidate_id=signature('calibration',d,qq,plan),point=point(d,q),dataset=d,base_qp=q,qp=qq,plan=plan,levels=ls,target_bits=target,predicted_bits=bits,predicted_delta_lpips=value,predicted_rate_matched=matched(bits,target),prediction_scope='same-QP additive surrogate' if qq==q else 'neighbor-QP transferred local-delta surrogate; actual verification required',rejected_noop=noop,selection_rejection='q_floor_exact_noop' if noop else None))
 dump(saved,rows)
 write('oracle/rate_allocation/candidate_predictions.csv',[dict(r,plan=json.dumps(r['plan']),levels=json.dumps(r['levels'])) for r in rows])
 return rows

def calibration_jobs(predictions):
 path=ROOT/'manifests/calibration_search.json'
 if path.exists():return [load(ROOT/'manifests/jobs'/(x+'.json')) for x in load(path)['new_job_ids']]
 jobs={};selection=[]
 for d in load(ROOT/'manifests/videos.json'):
  for q in cfg()['QPs']:
   group=[r for r in predictions if r['point']==point(d,q) and not r['rejected_noop']]
   at_q=[r for r in group if r['qp']==q]
   feasible=[r for r in at_q if r['predicted_rate_matched']]
   def gap(r):return max(r['predicted_bits']-r['target_bits'],.995*r['target_bits']-r['predicted_bits'],0)
   ranked=sorted(feasible,key=lambda r:(r['predicted_delta_lpips'],abs(r['predicted_bits']-r['target_bits']),r['candidate_id'])) if feasible else sorted(at_q,key=lambda r:(gap(r),r['predicted_delta_lpips'],r['candidate_id']))
   chosen=ranked[:2]
   minimum=min(at_q,key=lambda r:(r['predicted_bits'],r['candidate_id']))
   if minimum['candidate_id'] not in {r['candidate_id'] for r in chosen}:chosen.append(minimum)
   for qq in (q-1,q+1):
    pool=[r for r in group if r['qp']==qq]
    if pool:chosen.append(min(pool,key=lambda r:(gap(r),abs(r['predicted_bits']-r['target_bits']),r['predicted_delta_lpips'],r['candidate_id'])))
   for r in chosen:
    ref=available('calibration',d,r['qp'],r['plan'])
    selection.append(dict(candidate_id=r['candidate_id'],point=r['point'],reused=bool(ref),predicted_bits=r['predicted_bits'],predicted_delta_lpips=r['predicted_delta_lpips']))
    if not ref:
     job=new_job('calibration',d,r['qp'],r['plan']);jobs[job['id']]=job
 dump(path,dict(new_job_ids=list(jobs),selected=selection))
 return list(jobs.values())

def freeze_policies():
 path=ROOT/'oracle/rate_allocation/frozen_policy.json'
 if path.exists():return load(path)
 import numpy as np
 from PIL import Image
 policies=[];candidates=[]
 videos=load(ROOT/'manifests/videos.json')
 for d in videos:
  v=videos[d]['calibration'];importance=[]
  def frame(i):
   name=f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png'
   with Image.open(Path(v['input_dir'])/name) as im:return np.asarray(im).astype(np.float32)/255
  for b in cfg()['blocks']:
   diff=np.abs(frame(b['frame'])-frame(b['frame']-1));y,x,h,w=(b[k]*16 for k in ('y','x','h','w'));importance.append(float(diff[y:y+h,x:x+w].mean()))
  for q in cfg()['QPs']:
   target=baseline('calibration',d,q)['rate']['native_codec_stream_bits'];pool=[]
   # Only predeclared candidate encodings, actual local/joint data on calibration videos.
   for key,ref in sorted(refs().items()):
    r=read_receipt(ref);j=r['job']
    phase=j.get('phase','calibration')
    if phase!='calibration' or j['dataset']!=d or j['qp'] not in (q-1,q,q+1) or not j['plan']:continue
    bits=r['rate']['actual_total_bits'];valid=usable(r) and matched(bits,target)
    item=dict(encoding_key=key,point=point(d,q),qp=j['qp'],plan=j['plan'],bits=bits,target_bits=target,actual_rate_matched=valid,exact_rate=bits==target,LPIPS=r['result']['LPIPS'],DISTS=r['result']['DISTS'],FloLPIPS=r['result']['FloLPIPS'],delta_lpips=r['result']['LPIPS']-baseline('calibration',d,q)['result']['LPIPS'],receipt_path=ref['receipt_path'],rejection=None if valid else 'actual_total_rate_outside_registered_band',heuristic_score=sum(importance[i]*(1 if f==.5 else -1 if f==2. else 0) for i,f in enumerate(levels(j['plan']))))
    candidates.append(item)
    if valid:pool.append(item)
   exact=[r for r in pool if r['exact_rate']]
   if exact:pool=exact
   common=dict(point=point(d,q),dataset=d,base_qp=q,target_bits=target,importance=importance,pool_size=len(pool),calibration_only=True,independent_test_metrics_used=False)
   if not pool:
    policies.append(dict(common,status='no_feasible_candidate',strategy=None,reason='No measured nonempty calibration plan matched the predeclared actual-rate band after bounded joint/global-QP search'));continue
   picked=[('D_oracle',None,min(pool,key=lambda r:(r['LPIPS'],abs(r['bits']-target),r['encoding_key']))),('C_temporal_or_content_heuristic',None,max(pool,key=lambda r:(r['heuristic_score'],-abs(r['bits']-target),r['encoding_key'])))]
   for seed in cfg()['random_seeds']:picked.append(('B_random',seed,random.Random(seed+q+sum(map(ord,d))).choice(pool)))
   for strategy,seed,chosen in picked:
    policies.append(dict(common,status='FROZEN',strategy=strategy,seed=seed,calibration_choice=chosen,frozen_plan=chosen['plan'],test_global_qps=[q-1,q,q+1],algorithm='minimum measured joint calibration LPIPS in the bounded verified candidate pool' if strategy=='D_oracle' else 'fixed temporal RGB importance score' if strategy.startswith('C_') else 'uniform fixed-seed sample from the identical actual-feasible calibration pool',global_optimal_codec_allocation=False))
 dump('oracle/rate_allocation/calibration_actual_candidates.json',candidates)
 dump(path,dict(registered_config_sha256=sha(ROOT/'configs/experiment.json'),frozen_before_test_encoding=True,time=time.time(),policies=policies))
 return load(path)

def main():
 assert sha(ROOT/'configs/experiment.json')==load(ROOT/'manifests/experiment_manifest.json')['config_sha256']
 # Exact repeats cover all four calibration datasets; pilot independently decodes persisted streams.
 pilots=[]
 for d,q,block,standalone in [('hevc_b',7,None,True),('hevc_b',7,1,True)]:
  plan=[] if block is None else [dict(cfg()['blocks'][block],factor=.5)]
  source=available('calibration',d,q,plan);assert source
  pilots.append(new_job('calibration',d,q,plan,'repeat',repeat_source=source['receipt_path'],standalone=standalone))
 stage('correctness_pilot',pilots)
 if not all(succeeded(j) for j in pilots):
  dump('manifests/stopping.json',dict(reason='Correctness pilot failed; no strategy expansion',stage='correctness_pilot'));return
 repeat=[]
 for d,q,block in [('uvg_holdout',2,0),('mcljcv',2,0),('ulong',7,1)]:
  plan=[dict(cfg()['blocks'][block],factor=.5)];source=available('calibration',d,q,plan);assert source
  repeat.append(new_job('calibration',d,q,plan,'repeat',repeat_source=source['receipt_path']))
 stage('sensitivity_repeats',repeat)
 if not all(succeeded(j) for j in repeat):
  dump('manifests/stopping.json',dict(reason='Local sensitivity repeat failed; do not expand',stage='sensitivity_repeats'));return
 predictions=prediction_table();stage('bounded_joint_and_QP_search',calibration_jobs(predictions))
 frozen=freeze_policies();frozen_sha=sha(ROOT/'oracle/rate_allocation/frozen_policy.json')
 baselines=[]
 for d in load(ROOT/'manifests/videos.json'):
  for q in cfg()['QPs']:
   if not available('test',d,q,[]):baselines.append(new_job('test',d,q,[],'test_baseline'))
 stage('independent_baselines',baselines)
 assert frozen_sha==sha(ROOT/'oracle/rate_allocation/frozen_policy.json')
 trials={}
 for policy in frozen['policies']:
  if policy['status']!='FROZEN':continue
  if not available('test',policy['dataset'],policy['base_qp'],[]):continue
  for q in policy['test_global_qps']:
   if not available('test',policy['dataset'],q,policy['frozen_plan']):
    job=new_job('test',policy['dataset'],q,policy['frozen_plan'],'test_candidate');trials[job['id']]=job
 stage('independent_frozen_policy_trials',list(trials.values()))
 assert frozen_sha==sha(ROOT/'oracle/rate_allocation/frozen_policy.json')
 dump('manifests/test_policy_integrity.json',dict(status='PASS',frozen_policy_sha256=frozen_sha,test_quality_used_for_selection=False,global_QP_selection='actual-rate gap only; performed by summarize.py',no_search_expansion=True))
 dump('manifests/stopping.json',dict(reason='Fixed bounded validation schedule exhausted; no further search or tolerance relaxation',stage='DONE'))

if __name__=='__main__':
 try:main()
 except Exception:
  dump('logs/driver_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
