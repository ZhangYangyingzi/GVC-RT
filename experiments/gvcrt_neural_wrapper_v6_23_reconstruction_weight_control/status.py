"""Read-only concise progress, including durable updates."""
from io23 import *
s=load(ROOT/'pipeline_status.json')
out={k:s.get(k) for k in ('status','phase','PID','completed_RD','expected_RD','training_steps','updated_unix')}
out['active']={k:{x:w[x] for x in ('PID','gpu')} for k,w in s.get('active',{}).items()}
out['actual_point_files']=len(list((ROOT/'parts').glob('ip_-1/*/*/video*.json')))
out['branches']={}
for b in BRANCHES:
    p=ROOT/'branches'/b/'training_status.json'
    out['branches'][b]=load(p) if p.exists() else {}
out['error']=s.get('error')
p=ROOT/'publication_status.json'
if p.exists():out['publication']=load(p)
print(json.dumps(out,indent=2))
