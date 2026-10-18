import json,time
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'pipeline_status.json'
print(json.dumps(json.loads(p.read_text()) if p.exists() else {'status':'NOT_STARTED'},indent=2))
for b in ('N_noP_BG_FD','P_PBG_FD'):
 q=R/'branches'/b/'training_status.json'
 if q.exists():print(b,json.loads(q.read_text()))
print('new_step500_points',len([p for p in (R/'tasks').glob('adapt_ulong_*_500_*.json') if json.loads(p.read_text()).get('status')=='PASS']))
