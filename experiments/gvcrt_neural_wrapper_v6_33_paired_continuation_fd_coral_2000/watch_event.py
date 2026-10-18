"""Wait for a new failure/terminal state, or one hourly summary. No model imports."""
from pathlib import Path
import json,time,os
root=Path(__file__).resolve().parent;start=time.time();err=root/'errors.jsonl';offset=err.stat().st_size if err.exists() else 0
while True:
 p=root/'pipeline_status.json'
 if p.exists():
  s=json.loads(p.read_text());fail=s.get('failures',{});terminal=s.get('terminal',False)
  if fail or terminal or time.time()-start>=3600:
   print(json.dumps({k:s.get(k) for k in ('status','calculation','report','publication','training_updates','additional_training_updates','new_prefill','new_calibration','prefill','calibration','evaluation_completed','evaluation_expected','terminal','failures')},ensure_ascii=False,indent=2),flush=True);break
  if time.time()-s.get('heartbeat_unix',time.time())>300:
   pid=s.get('driver',{}).get('PID');alive=pid and Path(f'/proc/{pid}/stat').exists()
   if not alive:print('ERROR: DRIVER_NOT_ALIVE',pid,flush=True);break
 if err.exists() and err.stat().st_size>offset:
  with err.open() as f:f.seek(offset);print('NEW_ERROR_EVENT\n'+f.read(),flush=True)
  break
 time.sleep(20)
