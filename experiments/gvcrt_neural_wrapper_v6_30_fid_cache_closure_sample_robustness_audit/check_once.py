"""Read-only compact status, without torch or directory scanning."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
def read(name):
    try:return json.loads((R/name).read_text())
    except (OSError,ValueError):return {}
s=read('status.json');p=read('publication_status.json')
for k in ('status','phase','completed','total','running','recent_progress','heartbeat','failure_count','PID','session'):
    print(f'{k}: {s.get(k,"not recorded")}')
print('DONE:', (R/'DONE.json').exists())
print('FAILED:', (R/'FAILED.json').exists())
print('BLOCKED:', (R/'BLOCKED.json').exists())
print('report:',R/'REPORT.md')
print('publication:',p.get('status','PENDING'))
print('commit:',p.get('commit',''))
