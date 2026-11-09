"""CPU-only live status joins cache, committed updates and wait-time receipts."""
from common_s2 import *

def run():
    env();rows=plan_rows();totals={g:dict(last=0,wait=0.) for g in GROUPS}
    with lock('locks/status_monitor.lock'):
        while True:
            status=load(ROOT/'status.json');cache=load(ROOT/'cache/progress.json');groups={}
            for g in GROUPS:
                p=ROOT/g/'training/progress.json';progress=load(p) if p.exists() else dict(committed_updates=0)
                committed=progress['committed_updates']
                for step in range(totals[g]['last']+1,committed+1):
                    receipt=ROOT/g/f'training/receipts/{step:04d}.json'
                    if not receipt.exists():break
                    totals[g]['wait']+=load(receipt)['cache_wait_seconds'];totals[g]['last']=step
                first=ROOT/f'audits/first_formal_step_{g}.json'
                completion=ROOT/g/'training/completion.json'
                next_row=rows[4*committed] if committed<300 and not completion.exists() else None
                groups[g]=dict(committed_optimizer_steps=committed,next_window=next_row['sample_key'] if next_row else None,next_cursor=4*committed,cumulative_committed_cache_wait_seconds=totals[g]['wait'],seconds_from_task_start_to_first_formal_step=load(first)['seconds'] if first.exists() else None,training_GPUs=[0,1] if g=='E' else [2,3])
            dump(ROOT/'live_pipeline_status.json',dict(status=status['status'],complete=status.get('complete',False),current_stage=status['current_stage'],heartbeat=time.time(),groups=groups,READY_windows=cache['READY_windows'],cache_bytes=cache['cache_bytes'],cache_production_windows_per_second=cache['windows_per_second'],cache_GPUs=cache.get('cache_GPUs',[0]),cache_workers=cache['gpu_workers'],active_cache_producers=cache.get('active',{}),prefetch_target=load(ROOT/'scheduler_config.json')['prefetch_windows'],disk_byte_cap=load(ROOT/'scheduler_config.json')['derived_byte_cap'],resume_command=status.get('resume_command')))
            if status['status'] in ['COMPLETE','FAILED']:break
            time.sleep(5)
if __name__=='__main__':run()
