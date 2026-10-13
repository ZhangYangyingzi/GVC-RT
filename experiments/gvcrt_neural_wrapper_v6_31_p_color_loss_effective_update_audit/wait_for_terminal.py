"""One persistent event waiter: output hourly, on error, and on terminal only."""
import json,time
from pathlib import Path
root=Path(__file__).resolve().parent;last_hour=None;error_size=0
while True:
 p=root/'status.json'
 if p.exists():
  d=json.loads(p.read_text());h=int(time.time()//3600)
  if h!=last_hour:print('HOURLY',json.dumps({k:d.get(k) for k in ('status','phase','counts','heartbeat','publication_status')}),flush=True);last_hour=h
  e=root/'errors.jsonl';size=e.stat().st_size if e.exists() else 0
  if size!=error_size:print('ERROR_EVENT',size,'see errors.jsonl',flush=True);error_size=size
  if d.get('status') in ('DONE','FAILED','BLOCKED'):print('TERMINAL',d['status'],flush=True);break
 time.sleep(60)
