"""No torch, no recursive scans: read one atomic pipeline status."""
from pathlib import Path
import json,time
p=Path(__file__).resolve().parent/'pipeline_status.json'
if not p.exists():print('Pipeline status not yet written')
else:
 s=json.loads(p.read_text());keys=['status','calculation','report','publication','training_updates','prefill','calibration','evaluation_completed','evaluation_expected','remaining_jobs','terminal']
 print(json.dumps({k:s.get(k) for k in keys},ensure_ascii=False,indent=2));print('heartbeat_age_seconds',round(time.time()-s.get('heartbeat_unix',time.time())))
 for w in s.get('running',[]):print('worker',w['id'],'physical_GPU',w.get('gpu'),'PID',w['PID'])
 for k,v in s.get('failures',{}).items():print('ERROR',k,v.get('log'),v.get('error','')[-1200:])
