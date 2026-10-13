"""Read only small state. No torch, recursion, subprocess, or scheduling."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
if __name__=='__main__':
 for n in ('launch.json','status.json','publication_status.json'):
  p=root/n
  if p.exists():
   d=json.loads(p.read_text());print(n,json.dumps({k:d[k] for k in ('PID','session','status','phase','counts','heartbeat','calculation_status','report_status','integrity_status','publication_status','commit','error') if k in d}))
