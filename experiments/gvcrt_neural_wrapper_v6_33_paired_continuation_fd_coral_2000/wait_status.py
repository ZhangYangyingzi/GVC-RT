"""Hourly concise status or immediate terminal notification; no torch/directory scans."""
from pathlib import Path
import time,json,subprocess,sys
root=Path(__file__).resolve().parent;last=0
while True:
 p=root/'pipeline_status.json'
 if p.exists():
  s=json.loads(p.read_text());now=time.time()
  if now-last>=3600 or s.get('terminal'):
   subprocess.run([sys.executable,str(root/'check_once.py')]);last=now
  if s.get('terminal'):break
 time.sleep(30)
