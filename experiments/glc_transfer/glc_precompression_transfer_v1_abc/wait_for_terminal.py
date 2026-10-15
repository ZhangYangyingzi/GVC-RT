"""Read-only event monitor; no model/API calls and no duplicate driver launch."""
import json,time,os
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def main():
 last=None;hour=None
 while True:
  s=read(R/'status.json');phase=s.get('phase');terminal=s.get('terminal',False);h=int(time.time()//1800)
  if phase!=last or h!=hour or terminal:
   updates={b:read(R/f'branches/{b}/status.json').get('updates',0) for b in ('B_rgb_P','C_latent_P')}
   cal=len(list((R/'calibration').glob('*_P.json')))
   print(json.dumps(dict(status=s,training=updates,calibration_measurements=cal,evaluation_valid_receipts=len(list((R/'results/points').glob('*.json'))),publication=read(R/'publication_status.json').get('phase')),ensure_ascii=False),flush=True);last=phase;hour=h
  if terminal:return
  pid=s.get('PID')
  if pid:
   try:os.kill(pid,0)
   except ProcessLookupError:print('BLOCKED: DRIVER_DISAPPEARED',pid,flush=True);return
  time.sleep(30)
if __name__=='__main__':main()
