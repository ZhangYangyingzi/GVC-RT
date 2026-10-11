"""Read-only monitoring; exit on terminal state, never stop workers."""
from io23 import *
while True:
    s=load(ROOT/'pipeline_status.json');steps={}
    for b in BRANCHES:
        p=ROOT/'branches'/b/'training_status.json'
        steps[b]=load(p).get('adaptation_step',0) if p.exists() else 0
    print(json.dumps(dict(status=s.get('status'),phase=s.get('phase'),updates=steps,
        completed_RD=s.get('completed_RD'),active=list(s.get('active',{})),error=s.get('error'))),flush=True)
    if s.get('status') in ('PASS','FAIL'):
        break
    time.sleep(45)
