"""Minute-spaced read-only file monitor; never schedules or signals GPU work."""
import json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name):
    try:return json.loads((ROOT/name).read_text())
    except (OSError,ValueError):return {}

def main():
    deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline:
        state=read('status.json');scheduler=read('evaluation_scheduler.json')
        publication=read('publication_status.json')
        result=dict(time=time.time(),status=state.get('status'),phase=state.get('phase'),
                    PID=state.get('PID'),
                    updates={b:read(f'branches/{b}/status.json').get('updates') for b in ('B_rgb_P','C_latent_P')},
                    completed=state.get('completed',state.get('evaluation_completed',scheduler.get('completed'))),
                    expected=state.get('expected',state.get('evaluation_expected',1380)),
                    receipt_files=len(list((ROOT/'results/points').glob('*.json'))),
                    active_workers=len(scheduler.get('running',[])) if state.get('status')=='RUNNING' else 0,
                    resource_preemptions=scheduler.get('resource_preemptions',{}),
                    publication_phase=publication.get('phase'),
                    commit=publication.get('commit'),github=publication.get('github'))
        if state.get('error'):result['error']=state['error']
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if state.get('terminal'):return
        time.sleep(60)
    print(json.dumps(dict(status='MONITOR_TIMEOUT',limit_hours=72)),flush=True)

if __name__=='__main__':main()
