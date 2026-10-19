import json,time,os
from pathlib import Path
R=Path(__file__).resolve().parent
out={'experiment':R.name,'updated_unix':time.time(),'training':{},'evaluation':{},'aggregation':{},'publication':{}}
for key,p in [('training',R/'branches/B_basicLoss_noP_500/training_status.json'),('evaluation',R/'evaluation_status.json'),('aggregation',R/'audits/aggregation_integrity.json'),('publication',R/'publication_status.json')]:
 if p.exists():
  try:out[key]=json.load(open(p))
  except Exception as e:out[key]={'status':'READ_ERROR','error':repr(e)}
 else:out[key]={'status':'NOT_STARTED'}
print(json.dumps(out,indent=2))
