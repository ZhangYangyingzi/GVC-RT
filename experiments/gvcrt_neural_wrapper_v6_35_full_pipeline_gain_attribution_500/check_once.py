import json,time
from pathlib import Path
R=Path(__file__).resolve().parent
for name in ('pipeline_status.json','final_integrity.json','publication_status.json'):
 p=R/name
 if p.exists():
  d=json.load(open(p));print(name,json.dumps({k:d.get(k) for k in ('status','calculation_status','publication_status','completed_new','expected_new','completed_unix','commit') if k in d},sort_keys=True))
 else:print(name,'NOT_WRITTEN')
