"""Resumable bounded jobs; independent failures do not cancel other jobs."""
from base import *
import itertools,random,traceback,signal

def receipt(j):
 p=ROOT/'manifests/receipts'/f'{j["id"]}.json'
 return load(p) if p.exists() else None
def passed(j):
 r=receipt(j);return r and r['status']=='PASS' and r['job_sha256']==sha(ROOT/'manifests/jobs'/f'{j["id"]}.json')
def register(j):
 p=ROOT/'manifests/jobs'/f'{j["id"]}.json'
 if p.exists():assert load(p)==j,('immutable job changed',j['id'])
 else:dump(p,j)
 return j
def point_jobs():
 return [(v,q,key(v,q)+'_baseline') for v in load(ROOT/'manifests/videos.json') for q in load(ROOT/'configs/experiment.json')['QPs']]
def base_job(v,q,b,kind='baseline',plan=None,suffix=None,**extra):
 j=dict(id=b if kind=='baseline' else b.replace('_baseline','_'+(suffix or kind)),kind=kind,dataset=v['dataset'],qp=q,plan=plan or [],**extra)
 if kind!='baseline':j['baseline_id']=b
 return register(j)
def resources():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(x.split(',')[0]):list(map(int,x.split(',')[1:])) for x in out.splitlines()}
def stage(name,jobs):
 pending=[j for j in jobs if not passed(j)];running=[];exits=[];retries={}
 dump('manifests/stages/'+name+'.json',dict(jobs=[j['id'] for j in jobs],prepared=time.time()))
 while pending or running:
  for p,j,g in list(running):
   rc=p.poll()
   if rc is None:continue
   running.remove((p,j,g));r=receipt(j)
   exits.append(dict(id=j['id'],pid=p.pid,gpu=g,exit=rc,status=r.get('status') if r else 'MISSING_RECEIPT'))
   if rc and r and 'OutOfMemoryError' in r.get('traceback','') and retries.get(j['id'],0)<1:
    retries[j['id']]=retries.get(j['id'],0)+1
    dump('logs/failed_attempts/'+j['id']+'_'+str(p.pid)+'.json',r);pending.append(j)
  mem=resources();busy={g for p,j,g in running}
  for g in (4,5,6,7):
   if pending and g not in busy and mem[g][0]>=7000:
    j=pending.pop(0);path=ROOT/'manifests/jobs'/f'{j["id"]}.json'
    command=[PYTHON,'-B','-u',str(ROOT/'src/worker.py'),'--job',str(path),'--gpu',str(g)]
    with (ROOT/'logs'/f'{j["id"]}.log').open('a') as f:p=subprocess.Popen(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump('manifests/commands/'+j['id']+'.json',dict(command=command,pid=p.pid,gpu=g,job_sha256=sha(path),GPU_snapshot=mem,start=time.time()))
    running.append((p,j,g))
  dump('run_status.json',dict(status='RUNNING',stage=name,pending=len(pending),active=[dict(id=j['id'],pid=p.pid,gpu=g) for p,j,g in running],finished=sum(bool(passed(j)) for j in jobs),planned=len(jobs),GPU_snapshot=mem,updated=time.time()))
  dump('logs/stage_exits/'+name+'.json',exits)
  if pending or running:time.sleep(30)
 dump('manifests/stage_results/'+name+'.json',dict(status='PASS' if all(passed(j) for j in jobs) else 'PARTIAL',jobs=[dict(id=j['id'],status=receipt(j)['status'] if receipt(j) else 'MISSING') for j in jobs],exits=exits))

def build_allocations(points,locals):
 import numpy as np
 from PIL import Image
 cfg=load(ROOT/'configs/experiment.json');out=[];selections=[]
 sys.path.insert(0,str(V32));from adapter32 import frames_for
 for index,(v,q,b) in enumerate(points):
  single=[j for j in locals if j['baseline_id']==b]
  if len(single)!=8 or not all(passed(j) for j in single):continue
  costs={(j['block_index'],j['plan'][0]['factor']):receipt(j)['delta']['delta_actual_bits'] for j in single}
  losses={(j['block_index'],j['plan'][0]['factor']):receipt(j)['delta']['delta_lpips'] for j in single}
  frames=frames_for(v);importance=[]
  for a in cfg['blocks']:
   f,y,x,h,w=(a[k] for k in ('frame','y','x','h','w'));diff=(frames[f]-frames[f-1]).abs()
   importance.append(float(diff[:,:,16*y:16*(y+h),16*x:16*(x+w)].mean()))
  choices=[]
  for levels in itertools.product((1.,.5,2.),repeat=4):
   cost=sum(costs[(i,f)] for i,f in enumerate(levels) if f!=1)
   value=sum(losses[(i,f)] for i,f in enumerate(levels) if f!=1)
   if cost<=0:choices.append(dict(levels=levels,predicted_delta_bits=cost,predicted_delta_lpips=value,heuristic_score=sum(importance[i]*(1 if f==.5 else -1 if f==2 else 0) for i,f in enumerate(levels))))
  assert choices
  rr=random.Random(cfg['seed']+index);pool=[p for p in choices if sum(f!=1 for f in p['levels'])>=2] or choices
  chosen=dict(B_random=rr.choice(pool),C_temporal_RGB=max(choices,key=lambda p:(p['heuristic_score'],p['predicted_delta_bits'])),D_oracle=min(choices,key=lambda p:(p['predicted_delta_lpips'],-p['predicted_delta_bits'],p['levels'])))
  for strategy,c in chosen.items():
   plan=[dict(cfg['blocks'][i],factor=f) for i,f in enumerate(c['levels']) if f!=1]
   j=base_job(v,q,b,'allocation',plan,strategy,strategy=strategy)
   out.append(j);selections.append(dict(baseline_id=b,strategy=strategy,plan=plan,temporal_RGB_importance=importance,**c,algorithm='exhaustive81 additive-surrogate states' if strategy=='D_oracle' else 'fixed-seed random over predicted-budget-feasible states' if strategy=='B_random' else 'maximize temporal RGB difference score under measured additive cost',global_optimal_actual_allocation=False,interaction_assumption='surrogate only; actual joint re-encoding required',budget_bits=receipt(base_job(v,q,b))['rate']['actual_total_bits']))
 dump('oracle/rate_allocation/selection_manifest.json',selections)
 return out

def main():
 if not (ROOT/'configs/experiment.json').exists():subprocess.run([PYTHON,str(ROOT/'src/audit_paths.py')],check=True)
 points=point_jobs();baselines=[base_job(v,q,b) for v,q,b in points]
 stage('baseline',baselines)
 valid=[(v,q,b) for v,q,b in points if passed(base_job(v,q,b))]
 repeats=[base_job(v,q,b,'repeat') for v,q,b in valid];stage('repeat',repeats)
 valid=[(v,q,b) for v,q,b in valid if passed(base_job(v,q,b,'repeat'))]
 assert valid,'No reproducible original baseline available'
 cfg=load(ROOT/'configs/experiment.json');locals=[]
 for v,q,b in valid:
  for i,a in enumerate(cfg['blocks']):
   for factor in (.5,2.):
    j=dict(id=b.replace('_baseline',f'_block{i}_f{factor:g}'),kind='local',dataset=v['dataset'],qp=q,baseline_id=b,block_index=i,plan=[dict(a,factor=factor)])
    locals.append(register(j))
 v,q,b=valid[0];noop=base_job(v,q,b,'noop',[dict(cfg['blocks'][0],factor=1.)])
 pilot=[j for j in locals if j['baseline_id']==b and j['block_index']==0]
 stage('local_correctness',[noop,*pilot])
 gate=passed(noop) and all(passed(j) for j in pilot)
 dump('audit/local_control_correctness.json',dict(status='PASS' if gate else 'FAILED',real_local_rate_control=gate,native_arbitrary_block_QP=False,experimental_stream_extension=True,pilot_jobs=[noop['id']]+[j['id'] for j in pilot],checks='native noop equivalence; all RANS-decoded symbols exact; all causal DPB features exact; independent persisted-stream decode; whole-model weight hashes unchanged; side info included in actual rate'))
 if not gate:raise RuntimeError('Experimental local control correctness failed; real allocation disabled. Inspect pilot failures before recovery.')
 stage('local_interventions',locals)
 joints=[base_job(v,q,b,'joint',[dict(cfg['blocks'][0],factor=.5),dict(cfg['blocks'][1],factor=2.)]) for v,q,b in valid]
 stage('joint_verification',joints)
 allocations=build_allocations(valid,locals);stage('allocation',allocations)
 calibrations=[]
 for j in allocations:
  if not passed(j):continue
  target=load(ROOT/'manifests/receipts'/f'{j["baseline_id"]}.json')['rate']['actual_total_bits']
  if receipt(j)['rate']['actual_total_bits']==target:continue
  for q2 in (j['qp']-1,j['qp']+1):
   if 0<=q2<=9:
    trial=dict(j,id=j['id']+'_calibq'+str(q2),kind='calibration',qp=q2);calibrations.append(register(trial))
 stage('global_QP_calibration',calibrations)
 subprocess.run([PYTHON,str(ROOT/'src/finalize.py')],check=True,cwd=REPO)
 subprocess.run([PYTHON,str(ROOT/'src/publish.py')],check=True,cwd=REPO)

if __name__=='__main__':
 try:main()
 except Exception:
  dump('logs/driver_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));dump('run_status.json',dict(status='FAILED_REQUIRES_RECOVERY',traceback=traceback.format_exc(),time=time.time()));raise
