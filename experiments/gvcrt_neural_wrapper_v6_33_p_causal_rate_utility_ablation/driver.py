"""Bounded independent workers; memory checks, failures retained, no shared writers."""
from runtime import *
import os
import subprocess
import time
import collections

def launch(argv,name):
    log=(ROOT/'logs'/f'{name}.log').open('a')
    p=subprocess.Popen([PYTHON,'-B','-u',*map(str,argv)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    return p

def free_memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in out.splitlines()}

class AdoptedEvaluation:
    """Keep live independent workers across scheduler-only recovery."""
    def __init__(self,pid,branch,shard,shards):
        self.pid=pid;self.branch=branch;self.shard=shard;self.shards=shards
    def alive(self):
        try:
            args=Path(f'/proc/{self.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            return str(ROOT/'evaluate.py') in args and f'--method {self.branch}' in args and f'--shard {self.shard} ' in args
        except (OSError,UnicodeError):return False
    def poll(self):
        if self.alive():return None
        tasks=[(d,v,q) for d,vs in load(V32/'eval_manifest.json').items() for v in vs for q in range(10)]
        from evaluate import task_name
        expected=[ROOT/'evaluation/points'/(task_name(d,v['video_index'],self.branch,q)+'.json') for i,(d,v,q) in enumerate(tasks) if i%self.shards==self.shard]
        return 0 if all(p.exists() and load(p).get('status')=='PASS' for p in expected) else 1
    def terminate(self):
        if self.alive():os.kill(self.pid,15)

class AdoptedUtility(AdoptedEvaluation):
    def __init__(self,pid,pair,which):self.pid=pid;self.pair=pair;self.which=which
    def alive(self):
        try:
            args=Path(f'/proc/{self.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            return str(ROOT/'utility.py') in args and f'--pair {self.pair} ' in args and f'--which {self.which} ' in args
        except (OSError,UnicodeError):return False
    def poll(self):
        if self.alive():return None
        pair=load(ROOT/'utility/matched_pair_plan.json')['pairs'][self.pair];point=pair[self.which+'_point']
        p=ROOT/'utility'/f'{pair["dataset"]}_{pair["video"]}_{pair[self.which]}_{point["QP"]}.json'
        return 0 if p.exists() and load(p).get('status')=='PASS' else 1

def main():
    assert load(ROOT/'sanity/full_pipeline.json')['status']=='PASS'
    previous_state=load(ROOT/'driver_status.json') if (ROOT/'driver_status.json').exists() else {}
    utility_adoptions=previous_state.get('running',[]) if previous_state.get('stage')=='utility' else []
    queue=[];launched=set();running=[];done=[];failed=[];lastlaunch=0
    retries=collections.Counter();cooldown={};recoveries=[]
    shards=12
    previous=ROOT/'driver_status.json'
    if previous.exists() and load(previous).get('stage')=='training_and_evaluation':
        for item in load(previous).get('running',[]):
            p=AdoptedEvaluation(item['pid'],item['branch'],item['shard'],shards)
            if p.alive():running.append((p,item['branch'],item['shard'],item['gpu']))
    adopted={(b,s) for p,b,s,g in running}
    dump('evaluation/scheduler_adoption.json',dict(workers=[dict(pid=p.pid,branch=b,shard=s,gpu=g) for p,b,s,g in running],time=time.time(),model_or_optimizer_modified=False))
    while True:
        for b in BRANCHES[1:]:
            integrity=ROOT/'branches'/b/'training_integrity.json'
            if integrity.exists() and load(integrity)['status']=='PASS' and b not in launched:
                for s in range(shards):
                    if (b,s) not in adopted and AdoptedEvaluation(-1,b,s,shards).poll()!=0:queue.append((b,s))
                launched.add(b)
        for p,b,s,g in list(running):
            rc=p.poll()
            if rc is None:continue
            running.remove((p,b,s,g))
            record=dict(branch=b,shard=s,gpu=g,exit=rc,pid=p.pid)
            error=ROOT/'logs'/f'evaluation_failure_{p.pid}.json'
            oom=error.exists() and 'OutOfMemoryError' in load(error).get('traceback','')
            if rc and oom and retries[(b,s)]<5:
                retries[(b,s)]+=1;queue.append((b,s));cooldown[g]=time.time()+300
                recoveries.append(dict(record,retry=retries[(b,s)],reason='External/shared GPU memory contention; persisted completed points retained'))
                dump('evaluation/resource_recoveries.json',recoveries)
            else:(done if rc==0 else failed).append(record)
        train_errors=list((ROOT/'logs').glob('error_*.json'))
        if failed or train_errors:
            # Stop only workers belonging to this experiment; preserve every checkpoint.
            for p,b,s,g in running:p.terminate()
            dump('execution_failure.json',dict(status='FAILED',stage='execution',worker_failures=failed,training_errors=[load(p) for p in train_errors],reason='A worker failed; inspect preserved traceback and recover before reporting PASS.'))
            return
        if queue and time.time()-lastlaunch>20:
            mem=free_memory();counts=collections.Counter(g for p,b,s,g in running)
            limits=load(ROOT/'evaluation/resource_recovery_protocol.json')
            eligible=[g for g in (4,5,6,7) if mem[g]>=(limits.get('min_free_MiB_to_launch',9000) if counts[g]==0 else limits.get('additional_worker_min_free_MiB',18000)) and counts[g]<limits.get('max_evaluation_workers_per_gpu',1) and time.time()>=cooldown.get(g,0)]
            if eligible:
                g=max(eligible,key=lambda g:mem[g]);b,s=queue.pop(0)
                p=launch([ROOT/'evaluate.py','--method',b,'--gpu',g,'--shard',s,'--shards',shards],f'eval_{b}_{s}')
                running.append((p,b,s,g));lastlaunch=time.time()
        dump('driver_status.json',dict(stage='training_and_evaluation',trained=sorted(launched),queued=len(queue),running=[dict(pid=p.pid,branch=b,shard=s,gpu=g) for p,b,s,g in running],done=done,failed=failed,updated=time.time()))
        if len(launched)==3 and not queue and not running:break
        time.sleep(30)
    fid=launch([ROOT/'aggregate_fid.py'],'fid_aggregation')
    import shutil
    archive=ROOT/'latency/B1_reference'
    if not archive.exists():
        archive.mkdir(parents=True)
        for name in ('latency_summary.csv','latency_raw.csv','complexity.json'):shutil.copy2(ROOT/'latency'/name,archive/name)
        shutil.copy2(ROOT/'latency_integrity.json',archive/'latency_integrity.json')
    final_cp=ROOT/'branches'/BRANCHES[3]/'checkpoints/adaptation_step_1000.pt'
    existing_latency=load(ROOT/'latency_integrity.json')
    if existing_latency.get('status')!='PASS' or existing_latency.get('checkpoint',{}).get('sha256')!=sha(final_cp):
        samples={g:[] for g in (4,5,6,7)}
        for _ in range(3):
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            for line in out.splitlines():
                g,u=map(int,line.split(','))
                if g in samples:samples[g].append(u)
            time.sleep(1)
        while True:
            mem=free_memory();eligible=[g for g in samples if mem[g]>=6000]
            if eligible:break
            dump('driver_status.json',dict(stage='waiting_for_latency_GPU',free_MiB=mem,minimum_free_MiB=6000,updated=time.time()))
            time.sleep(30)
        gpu=min(eligible,key=lambda g:(sum(samples[g])/len(samples[g]),-mem[g]));dump('latency/GPU_selection.json',dict(selected=gpu,utilization_samples=samples,free_MiB=mem,criterion='lowest sampled utilization among allowed GPUs with >=6000MiB free; observed B1 same-architecture peak3309MiB; no quality used'))
        p=launch([ROOT/'latency.py','--checkpoint',final_cp,'--gpu',gpu],'latency_final_D');assert p.wait()==0
    p=launch([ROOT/'utility.py','--select'],'utility_select');assert p.wait()==0
    pairs=load(ROOT/'utility/matched_pair_plan.json')['pairs'];assert pairs,'No actual-rate utility pairs within 2%'
    jobs=[(i,w) for i in range(len(pairs)) for w in ('noP','P')];running=[];utility_retries=collections.Counter();utility_cooldown={}
    for item in utility_adoptions:
        p=AdoptedUtility(item['pid'],item['pair'],item['which'])
        if p.alive():running.append((p,item['pair'],item['which'],item['gpu']))
    active={(i,w) for p,i,w,g in running}
    jobs=[(i,w) for i,w in jobs if (i,w) not in active and AdoptedUtility(-1,i,w).poll()!=0]
    dump('utility/scheduler_adoption.json',dict(workers=[dict(pid=p.pid,pair=i,which=w,gpu=g) for p,i,w,g in running],remaining_jobs=jobs,time=time.time(),completed_results_unchanged=True))
    while jobs or running:
        for p,i,w,g in list(running):
            rc=p.poll()
            if rc is None:continue
            running.remove((p,i,w,g))
            error=ROOT/'logs'/f'utility_failure_{p.pid}.json'
            if rc and error.exists() and 'OutOfMemoryError' in load(error).get('traceback','') and utility_retries[(i,w)]<5:
                utility_retries[(i,w)]+=1;jobs.append((i,w));utility_cooldown[g]=time.time()+300
                dump('utility/resource_recoveries.json',dict(retries={str(k):v for k,v in utility_retries.items()},last_failed_pid=p.pid,reason='shared GPU memory contention; no completed utility result overwritten'))
            else:assert rc==0,('utility',i,w,rc)
        mem=free_memory();busy={g for p,i,w,g in running}
        for g in (4,5,6,7):
            if jobs and g not in busy and mem[g]>=7000 and time.time()>=utility_cooldown.get(g,0):
                i,w=jobs.pop(0);p=launch([ROOT/'utility.py','--pair',i,'--which',w,'--gpu',g],f'utility_{i}_{w}');running.append((p,i,w,g))
        dump('driver_status.json',dict(stage='utility',queued=len(jobs),running=[dict(pid=p.pid,pair=i,which=w,gpu=g) for p,i,w,g in running],updated=time.time()))
        time.sleep(30)
    assert fid.wait()==0,'FID aggregation failed; inspect log'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
    with (ROOT/'logs/report.log').open('a') as f:
        rc=subprocess.call(['/data1/anaconda3_new/anaconda_program/bin/python','-B','-u',str(ROOT/'report.py')],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert rc==0,('report',rc)
    p=launch([ROOT/'publish.py'],'publication');assert p.wait()==0,'Publication failed; inspect publication_status.json'
    dump('driver_status.json',dict(stage='COMPLETE',updated=time.time()))

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump('execution_failure.json',dict(status='FAILED',stage='driver',traceback=traceback.format_exc()));raise
