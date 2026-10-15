"""Read-only milestone watcher, never starts/stops workers or edits state."""
from pathlib import Path
import json,time
R=Path(__file__).resolve().parent
def main():
 last=None
 while True:
  s=json.loads((R/'status.json').read_text());updates={m:json.loads((R/f'branches/{m}/status.json').read_text())['updates'] for m in ('B_rgb_P','C_latent_P')};milestone=min(updates.values())//100
  if milestone!=last or min(updates.values())>=900 or s.get('terminal'):
   print(json.dumps(dict(event='READY_FOR_DRIVER_HANDOFF' if min(updates.values())>=900 else 'PROGRESS',updates=updates,status=s)),flush=True);last=milestone
  if min(updates.values())>=900 or s.get('terminal'):return
  time.sleep(30)
if __name__=='__main__':main()
