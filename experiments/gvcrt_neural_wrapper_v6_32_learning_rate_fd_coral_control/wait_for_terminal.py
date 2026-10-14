"""Same event waiter throughout the job; hourly/error/terminal output only."""
import json,time,os
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
 last_hour=None;last_errors=0
 while True:
  try:s=json.loads((R/'status.json').read_text())
  except (OSError,ValueError):s={}
  h=int(time.time()//3600);e=s.get('error_count',0)
  terminal=s.get('terminal',False)
  if h!=last_hour or e!=last_errors or terminal:print(json.dumps(s,ensure_ascii=False),flush=True);last_hour=h;last_errors=e
  if terminal:return
  pid=s.get('driver_PID')
  if pid:
   try:os.kill(pid,0)
   except ProcessLookupError:
    print(json.dumps(dict(status='FAILED_DRIVER_DISAPPEARED',PID=pid)),flush=True);return
  time.sleep(60)
if __name__=='__main__':main()
